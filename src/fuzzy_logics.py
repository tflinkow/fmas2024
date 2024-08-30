import torch

from logic import Logic

def safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / torch.where(y == 0., torch.finfo(y.dtype).eps, y)

def safe_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0., torch.full_like(x, torch.finfo(x.dtype).eps), x)

def safe_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.pow(safe_zero(x), y)

class FuzzyLogic(Logic):
    def __init__(self, name: str):
        super().__init__(name)

    def LEQ(self, x, y):
        return 1. - safe_div(torch.clamp(x - y, min=0.), (torch.abs(x) + torch.abs(y)))

    def NOT(self, x):
        return 1. - x

class GoedelFuzzyLogic(FuzzyLogic):
    def __init__(self, name='GD'):
        super().__init__(name)

    def AND(self, x, y):
        return torch.minimum(x, y)

    def OR(self, x, y):
        return torch.maximum(x, y)

    def IMPL(self, x, y):
        return torch.where(x < y, 1., y)

class KleeneDienesFuzzyLogic(GoedelFuzzyLogic):
    def __init__(self):
        super().__init__(name='KD')

    def IMPL(self, x, y):
        return Logic.IMPL(self, x, y)

class LukasiewiczFuzzyLogic(GoedelFuzzyLogic):
    def __init__(self):
        super().__init__(name='LK')

    def AND(self, x, y):
        return torch.maximum(torch.zeros_like(x), x + y - 1.)

    def OR(self, x, y):
        return torch.minimum(torch.ones_like(x), x + y)

    def IMPL(self, x, y):
        return Logic.IMPL(self, x, y)

class ReichenbachFuzzyLogic(FuzzyLogic):
    def __init__(self, name='RC'):
        super().__init__(name)

    def AND(self, x, y):
        return x * y

    def OR(self, x, y):
        return x + y - x * y

class GoguenFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self):
        super().__init__(name='GG')

    def IMPL(self, x, y):
        return torch.where(torch.logical_or(x <= y, x == 0.), torch.tensor(1., device=x.device), safe_div(y, x))

class ReichenbachSigmoidalFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self, s=9.0):
        super().__init__(name='RCS')
        self.s = s

    def IMPL(self, x, y):
        exp = torch.exp(torch.tensor(self.s / 2, device=x.device))

        numerator = (1. + exp) * torch.sigmoid(self.s * super().IMPL(x, y) - self.s/2) - 1.
        denominator = exp - 1.

        I_s = torch.clamp(safe_div(numerator, denominator), max=1.)

        return I_s

class YagerFuzzyLogic(FuzzyLogic):
    def __init__(self, p=2):
        super().__init__(name='YG')
        self.p = p

    def AND(self, x, y):
        return torch.clamp(1. - safe_pow( safe_pow(1. - x, self.p) + safe_pow(1. - y, self.p), 1. / self.p), min=0.)

    def OR(self, x, y):
        return torch.clamp(safe_pow( safe_pow(x, self.p) + safe_pow(y, self.p), 1. / self.p) , max=1.)

    def IMPL(self, x, y):
        return torch.where(torch.logical_and(x == 0., y == 0.), torch.ones_like(x), safe_pow(y, x))