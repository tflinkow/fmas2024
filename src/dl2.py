import torch

from logic import Logic
    
class DL2(Logic):
    def __init__(self):
        super().__init__(name='DL2')

    def LEQ(self, x, y):
        return torch.clamp(x - y, min=0.)

    def NOT(self, x):
        # technically, negation is not supported in DL2, but this allows to use base class implication definition
        return 1. - x

    def AND(self, x, y):
        return x + y

    def OR(self, x, y):
        return x * y