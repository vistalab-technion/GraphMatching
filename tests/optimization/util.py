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


def init_groundtruth(full_graph, part_graph,part_nodes,diag_val):
    n=len(full_graph.nodes())

    #set ground truth v
    v = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        if i not in part_nodes:
            v[i,i]=diag_val

    v=v.requires_grad_()

    #set ground truth E
    E = torch.zeros([n, n], dtype=torch.float64)
    for edge in part_graph.edges():
        u=edge[0]
        v=edge[1]
        if u in part_nodes and v not in part_nodes:
            E[u,v]=-1.0
            E[v,u]=1.0
            E[u,u]=E[u,u]-1.0
            E[v, v] = E[v, v] - 1.0

    E = E.requires_grad_()

    return v,E

