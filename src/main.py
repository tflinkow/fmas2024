from __future__ import print_function

from collections import namedtuple

import argparse

import time
import os
import csv

import sys

import numpy as np

import matplotlib.pyplot as plt

import onnx

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data

from torchvision import transforms

from logic import Logic
from dl2 import DL2
from fuzzy_logics import *

from constraints import *
from dataset import XYDataset, get_orig_x, get_orig_y
from models import LegoNet

from util import GradNorm, PGD, maybe, compute_mean_std

EpochInfo = namedtuple('EpochInfo', 'constr_acc pred_loss constr_loss pred_loss_weight constr_loss_weight')

std = None
mean = None

def denormalise(x: torch.Tensor) -> torch.Tensor:
    global std
    global mean
    return x * std + mean

def visualize(image, true_label, prediction, filename: str):
    image = image.cpu().numpy().squeeze()
    image = denormalise(image)

    true_label = true_label.cpu().numpy()
    prediction = prediction.cpu().detach().numpy()

    true_x = get_orig_x(true_label[0], image.shape[1])
    true_y = get_orig_y(true_label[1], image.shape[1])

    pred_x = get_orig_x(prediction[0], image.shape[1])
    pred_y = get_orig_y(prediction[1], image.shape[1])

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')

    plt.scatter(true_x, true_y, color='green', label=f'True x={true_x:}, y={true_y}')
    plt.scatter(pred_x, pred_y, color='red', marker='x', label=f'Predicted x={pred_x}, y={pred_y}')
    
    plt.legend()
    # plt.show()

    plt.tight_layout()
    plt.savefig(os.path.join('visualisation', filename), bbox_inches='tight')
    plt.close()

def train(epoch: int, model: torch.nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer, pgd: PGD, logic: Logic, constraint: Constraint, alpha: float, is_baseline: bool) -> EpochInfo:
    avg_pred_loss = torch.tensor(0., device=device)
    avg_constr_acc, avg_constr_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)

    model.train()

    with maybe(GradNorm(device, model), not is_baseline) as grad_norm:
        for batch_index, (data, target) in enumerate(train_loader, start=1):
            inputs, labels = data.to(device), target.to(device)

            outputs = model(inputs)
            ce_loss = F.mse_loss(outputs, labels)

            adv = pgd.attack(model, inputs, labels, logic, constraint, constraint.eps)

            with maybe(torch.no_grad(), is_baseline):
                dl_loss, sat = constraint.eval(model, inputs, adv, labels, logic, train=True)

            loss = ce_loss if is_baseline else grad_norm.weighted_loss(batch_index, ce_loss, dl_loss, alpha)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            avg_pred_loss += ce_loss
            avg_constr_acc += sat
            avg_constr_loss += dl_loss

    return EpochInfo(
        constr_acc=avg_constr_acc.item() / float(batch_index),
        pred_loss=avg_pred_loss.item() / float(batch_index),
        constr_loss=avg_constr_loss.item() / float(batch_index),
        pred_loss_weight=model.loss_weights[0].item(),
        constr_loss_weight=model.loss_weights[1].item()
    )

def test(epoch: int, model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader, pgd: PGD, logic: Logic, constraint: Constraint, is_baseline: bool) -> EpochInfo:
    test_ce_loss, test_dl_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    constr = torch.tensor(0., device=device)

    model.eval()

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(test_loader, start=1):
            inputs, labels = data.to(device), target.to(device)

            outputs = model(inputs)
            ce_loss = F.mse_loss(outputs, labels)

            adv = pgd.attack(model, inputs, labels, logic, constraint, constraint.eps)

            dl_loss, sat = constraint.eval(model, inputs, adv, labels, logic, train=False)

            test_ce_loss += ce_loss
            test_dl_loss += dl_loss

            constr += sat            

            if batch_index == 1:
                visualize(inputs[0], labels[0], outputs[0], f'vis_{logic.name if not is_baseline else "Baseline"}_epoch_{epoch}')
                visualize(adv[0], labels[0], model(adv)[0], f'adv_{logic.name if not is_baseline else "Baseline"}_epoch_{epoch}')

    return EpochInfo(
        constr_acc=constr.item() / len(test_loader.dataset),
        pred_loss=test_ce_loss.item() / len(test_loader.dataset),
        constr_loss=test_dl_loss.item() / len(test_loader.dataset),
        pred_loss_weight=None,
        constr_loss_weight=None
    )

def main():
    logics: list[Logic] = [
        DL2(),
        GoedelFuzzyLogic(),
        KleeneDienesFuzzyLogic(),
        LukasiewiczFuzzyLogic(),
        ReichenbachFuzzyLogic(),
        GoguenFuzzyLogic(),
        ReichenbachSigmoidalFuzzyLogic(),
        YagerFuzzyLogic()
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-set', type=str, choices=['lego'], default='lego')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pgd-steps', type=int, default=25)
    parser.add_argument('--pgd-gamma', type=float, default=float(48/255))
    parser.add_argument('--logic', type=str, default=None, choices=[l.name for l in logics])
    parser.add_argument('--constraint', type=str, help='Robustness(eps: float, delta: float)', default='Robustness(eps=0.1, delta=0.1)')
    parser.add_argument('--reports-dir', type=str, default='../reports')
    parser.add_argument('--grad-norm-alpha', type=float, default=0.1)
    args = parser.parse_args()

    kwargs = { 'batch_size': args.batch_size }

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')

        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        kwargs.update({ 'num_workers': 4, 'pin_memory': True })
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.logic == None:
        logic = logics[0] # need some logic loss for PGD even for baseline
        baseline = True
    else:
        logic = next(l for l in logics if l.name == args.logic)
        baseline = False

    def Robustness(eps: float, delta: float) -> RobustnessConstraint:
        return RobustnessConstraint(torch.tensor(eps, device=device), torch.tensor(delta, device=device))

    constraint: Constraint = eval(args.constraint)

    if args.data_set == 'lego':
        global mean
        global std

        mean, std = compute_mean_std('../data/lego', transform=transforms.Grayscale(num_output_channels=1))
        print(f'mean={mean}, std={std}')

        transform = transforms.Compose([
            transforms.Resize((112, 112)), # images are 224x224, but this helps reduce network size
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset = XYDataset('../data/lego', transform=transform)

        num_test = int(.1 * len(dataset))
        dataset_train, dataset_test = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

        model = LegoNet().to(device)

        print(f'total number of model parameters: {sum(p.numel() for p in model.parameters())}')

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True, **kwargs)

    # save test set data to numpy for DNNV
    output_dir = '../data_np'
    os.makedirs(output_dir, exist_ok=True)

    for i, (images, _) in enumerate(test_loader):
        images_np = images.numpy()
    
        for j in range(images_np.shape[0]):
            image_np = np.expand_dims(images_np[j], axis=1)
            npy_filename = os.path.join(output_dir, f'image_{i * images_np.shape[0] + j}.npy')
            np.save(npy_filename, image_np)

    pgd = PGD(device, args.pgd_steps, mean, std, args.pgd_gamma)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.reports_dir, exist_ok=True)

    if isinstance(constraint, RobustnessConstraint):
        folder = 'robustness'

    file_name = f'{args.reports_dir}/{folder}/{args.data_set}/{logic.name if not baseline else "Baseline"}.csv'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    os.makedirs('visualisation', exist_ok=True)

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        csvfile.write(f'#{sys.argv}\n')
        writer.writerow(['Epoch', 'Train-P-Loss', 'Train-C-Loss', 'Train-P-Loss-Weight', 'Train-C-Loss-Weight', 'Train-C-Acc', 'Test-P-Loss', 'Test-C-Loss', 'Test-C-Acc', 'Time'])

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                train_info = train(epoch, model, device, train_loader, optimizer, pgd, logic, constraint, args.grad_norm_alpha, is_baseline=baseline)
                train_time = time.time() - start

                print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not baseline else "Baseline"} \t TRAIN \t C-Acc: {train_info.constr_acc:.4f}\t P-Loss: {train_info.pred_loss:.4f}\t DL-Loss: {train_info.constr_loss:.4f}\t Time (Train) [s]: {train_time:.1f}')
            else:
                train_info = EpochInfo(0., 0., 0., 1., 1.)
                train_time = 0.

            test_info = test(epoch, model, device, test_loader, pgd, logic, constraint, is_baseline=baseline)
            test_time = time.time() - start - train_time

            writer.writerow([epoch, \
                             train_info.pred_loss, train_info.constr_loss, train_info.pred_loss_weight, train_info.constr_loss_weight, train_info.constr_acc, \
                             test_info.pred_loss, test_info.constr_loss, test_info.constr_acc, \
                             train_time])

            print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not baseline else "Baseline"} \t TEST \t C-Acc: {test_info.constr_acc:.4f}\t P-Loss: {test_info.pred_loss:.4f}\t DL-Loss: {test_info.constr_loss:.4f}\t Time (Test) [s]: {test_time:.1f}')
            print(f'===')

    torch.save(model.state_dict(), f'model_xy_{logic.name if not baseline else "Baseline"}.pth')

    torch.onnx.export(
        model.eval(),
        torch.randn(args.batch_size, 1, 112, 112, requires_grad=True).to(device=device),
        f'model_xy_{logic.name if not baseline else "Baseline"}.onnx',
        do_constant_folding=True,
        input_names=["input"],
        output_names=['output'],
        dynamic_axes={'input': { 0: 'batch_size' }, 'output': { 0: 'batch_size' }},
    )
    
    onnx_model = onnx.load(f'model_xy_{logic.name if not baseline else "Baseline"}.onnx')
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()