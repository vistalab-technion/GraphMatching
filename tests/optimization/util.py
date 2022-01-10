import torch
from torch import Tensor
import numpy as np


def double_centering(x):
    n = x.shape[0]
    row_sum = torch.sum(x, dim=0).unsqueeze(0)
    col_sum = torch.sum(x, dim=1).unsqueeze(1)
    all_sum = sum(col_sum)
    dc_x = x - (1 / n) * (
            Tensor.repeat(row_sum, n, 1) + Tensor.repeat(col_sum, 1, n)) + (
                   1 / n ** 2) * all_sum
    return dc_x


def set_diag_zero(x):
    ind = np.diag_indices(x.shape[0])
    x[ind[0], ind[1]] = torch.zeros(x.shape[0])

    return x

