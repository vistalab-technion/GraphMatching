from numpy import iterable
from torch import Tensor
from torch.optim.sgd import SGD


class PGM(SGD):
    def __init__(self,
                 params,
                 proxs,
                 lr: float = 0.2,
                 momentum: float = 0,
                 dampening: float = 0,
                 nesterov: bool = False):

        if momentum != 0:
            raise ValueError("momentum is not supported")
        if dampening != 0:
            raise ValueError("dampening is not supported")
        if nesterov != 0:
            raise ValueError("nesterov is not supported")

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0,
                      nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError(
                "Invalid length of argument proxs: {} instead of {}".format(len(proxs),
                                                                            len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, lamb, closure=None):
        # this performs a gradient step
        # optionally with momentum or nesterov acceleration
        self.param_groups[0]['params']
        super().step(closure=closure)

        for group in self.param_groups:
            prox = group['prox']

            # here we apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(z=p.data, lamb=lamb)
