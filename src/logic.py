import torch

from abc import ABC, abstractmethod

class Logic(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def AND(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def OR(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.OR(self.NOT(x), y)

    def EQUIV(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        return self.AND(self.IMPL(P, Q), self.IMPL(Q, P))