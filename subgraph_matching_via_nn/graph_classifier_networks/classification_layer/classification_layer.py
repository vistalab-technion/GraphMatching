from torch import nn
import torch
import torch.nn.functional as F


class BaseClassificationLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self, init_value=None):
        pass


class IdentityClassificationLayer(BaseClassificationLayer):

    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        return x



class SigmoidClassificationLayer(BaseClassificationLayer):

    def __init__(self, default_temp: float = 1.0,
                 learnable_temp: bool = False):
        super().__init__()
        self._default_temp = default_temp
        self._temp_param = nn.Parameter(
            torch.Tensor([default_temp]))
        if not learnable_temp:
            self._temp_param.requires_grad = False

    def forward(self, A, x):
        return torch.sigmoid(self._temp_param * x)

    def init_weights(self, init_value=None):
        if init_value is None:
            self._temp_param.data.fill_(
                self._default_temp)
        else:
            self._temp_param.data = init_value


class SoftmaxClassificationLayer(BaseClassificationLayer):

    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        return F.softmax(x, dim=0)


class TopkSoftmaxClassificationLayer(SigmoidClassificationLayer):

    def __init__(self, k: int, default_temp: float = 0.001,
                 learnable_temp: bool = False):
        super().__init__(default_temp=default_temp, learnable_temp=learnable_temp)
        self._k = k

    def softmax_w(self, x, w):
        logw = torch.log(w + 1E-12)  # Use 1E-12 to prevent numeric problem
        x = (x + logw) / self._temp_param
        x = x - torch.max(x)
        return torch.exp(x) / torch.sum(torch.exp(x))

    def forward(self, A, x):
        y = torch.zeros_like(x)
        for i in range(self._k):
            x1 = self.softmax_w(x, w=(1 - y))
            y = y + x1

        return y / y.sum()


class SquaredNormalizedClassificationLayer(BaseClassificationLayer):

    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        x = x ** 2
        x = x / x.sum()
        return x
