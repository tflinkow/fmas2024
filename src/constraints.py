import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Callable

from boolean_logic import Logic

from boolean_logic import *
from fuzzy_logics import FuzzyLogic

class Constraint(ABC):
    def __init__(self, eps: torch.Tensor):
        self.eps = eps
        self.boolean_logic = BooleanLogic()

    def get_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(outputs, dim=1)

    @abstractmethod
    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor) -> Callable[[Logic], torch.Tensor]:
        pass

    # usage:
    # loss, sat = eval()
    # where sat returns whether the constraint is satisfied or not
    def eval(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor, logic: Logic, train: bool, skip_sat: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        constraint = self.get_constraint(model, inputs, adv, labels)

        loss = constraint(logic)
        sat = constraint(self.boolean_logic).float() if not skip_sat else None

        assert not torch.isnan(loss).any()

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss

        if train:
            return torch.mean(loss), torch.mean(sat) if not skip_sat else None
        else:
            return torch.sum(loss), torch.sum(sat) if not skip_sat else None

class RobustnessConstraint(Constraint):
    def __init__(self, eps: torch.Tensor, delta: torch.Tensor):
        super().__init__(eps)
        self.delta = delta

    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.tensor]:
        return lambda l: l.LEQ(torch.linalg.vector_norm(model(adv) - model(inputs), ord=float('inf'), dim=1), self.delta)