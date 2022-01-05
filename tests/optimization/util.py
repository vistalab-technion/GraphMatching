import torch
from torch import Tensor


def double_centering(x):
    n = x.shape[0]
    row_sum = torch.sum(x, dim=0).unsqueeze(0)
    col_sum = torch.sum(x, dim=1).unsqueeze(1)
    all_sum = sum(col_sum)
    dc_x = x - (1 / n) * (
            Tensor.repeat(row_sum, n, 1) + Tensor.repeat(col_sum, 1, n)) + (
                   1 / n ** 2) * all_sum
    return dc_x
