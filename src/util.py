from __future__ import print_function

from contextlib import contextmanager

import torch
import numpy as np

import copy
import os

import time

from logic import Logic

from constraints import Constraint

from contextlib import ContextDecorator

from PIL import Image

def compute_mean_std(directory: str, transform=None):
    means = []
    stds = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            image = Image.open(os.path.join(directory, file))

            if transform:
                image = transform(image)

            image_np = np.array(image) / 255.

            if len(image_np.shape) == 2:
                means.append(np.mean(image_np))
                stds.append(np.std(image_np))
            elif len(image_np.shape) == 3:
                r, g, b = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]

                means.append([np.mean(r), np.mean(g), np.mean(b)])
                stds.append([np.std(r), np.std(g), np.std(b)])
        
    return np.mean(means, axis=0).astype(np.float32), np.mean(stds, axis=0).astype(np.float32)

@contextmanager
def maybe(context_manager, flag: bool):
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None

class Stopwatch(ContextDecorator):
    def __init__(self, jobname: str):
        self.jobname = jobname

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, _exc_type: None, _exc_value: None, _exc_traceback: None):
        self.end = time.time()
        self.elapsed = self.end - self.start

        if not self.jobname:
            print(f'Elapsed time: {self.elapsed:.2f} s')
        else:
            print(f'[{self.jobname}] Elapsed time: {self.elapsed:.2f} s')

# GradNorm (https://arxiv.org/abs/1711.02257) based on https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
class GradNorm(ContextDecorator):
    def __init__(self, device: torch.device, model: torch.nn.Module):
        self.model = model
        self.initial_loss = 0.
        self.device = device

    def safe_div(self, x, y):
        y = np.asarray(y)
        return x / np.where(y == 0., np.finfo(y.dtype).eps, y)

    def weighted_loss(self, batch_index: int, ce_loss: torch.Tensor, dl_loss: torch.Tensor, alpha: float):
        task_loss = torch.stack([ce_loss, dl_loss])
        weighted_loss = torch.mul(self.model.loss_weights, task_loss)

        if batch_index == 1:
            self.initial_loss = task_loss.detach().cpu().numpy()

        total_loss = torch.sum(weighted_loss)
        total_loss.backward(retain_graph=True)

        self.model.loss_weights.grad = None

        W = self.model.last_shared_layer
        norms = torch.stack([torch.norm(torch.mul(self.model.loss_weights[i], torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)[0])) for i in range(2)])

        loss_ratio = self.safe_div(task_loss.detach().cpu().numpy(), self.initial_loss)
        inverse_train_rate = self.safe_div(loss_ratio, np.mean(loss_ratio))

        mean_norm = np.mean(norms.detach().cpu().numpy())
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False, device=self.device)
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

        self.model.loss_weights.grad = torch.autograd.grad(grad_norm_loss, self.model.loss_weights)[0]

        return total_loss

    def renormalise(self):
        normalise_coeff = 2 / torch.sum(self.model.loss_weights.data, dim=0)
        self.model.loss_weights.data *= normalise_coeff

        print(f'loss_weights={self.model.loss_weights.data}')

    def __enter__(self):
        return self

    def __exit__(self, _exc_type: None, _exc_value: None, _exc_traceback: None):
        self.renormalise()

# based on https://github.com/oscarknagg/adversarial/blob/master/adversarial/functional.py
class PGD:
    def __init__(self, device: torch.device, steps: int, mean: tuple[float, float, float], std: tuple[float, float, float], gamma: float = 64/255):
        self.device = device
        self.steps = steps
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
        self.gamma = gamma

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def random_perturbation(self, x: torch.tensor, eps: float):
        perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
        perturbation = torch.sign(perturbation) * eps

        return x + perturbation

    @torch.enable_grad
    def attack(self, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, logic: Logic, constraint: Constraint, eps: float):
        model = copy.deepcopy(model)
        model.eval()

        adv = self.denormalise(inputs.clone().detach().requires_grad_(True).to(inputs.device))

        # random uniform start
        adv = self.random_perturbation(adv, eps)
        adv.requires_grad_(True)

        for _ in range(self.steps):
            _adv = adv.clone().detach().requires_grad_(True)

            loss, _ = constraint.eval(model, inputs, self.normalise(_adv), labels, logic, train=True, skip_sat=True)
            loss.backward()

            with torch.no_grad():
                gradients = _adv.grad.sign() * self.gamma
                adv += gradients

            # project back into l_norm ball and correct range
            adv = torch.max(torch.min(adv, inputs + eps), inputs - eps)
            adv = torch.clamp(adv, min=0, max=1)

        return self.normalise(adv.detach())