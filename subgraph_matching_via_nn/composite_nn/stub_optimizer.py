from typing import Optional, Callable, overload
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class StubOptimizer(Optimizer):
    def __init__(self, params: ParamsT, kwargs=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(params=params, defaults=kwargs)

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass

