import torch
from torch import diag, tensor


def laplacian(A):
    L = diag(A.sum(dim=1)) - A
    return L


def graph_edit_matrix(A, v):
    # scale = 0.5
    # e = A * (v - v.T) ** 2
    # E = 0.5 * (torch.tanh(scale * (e - 0.5)) + 1)
    E = A * (v - v.T) ** 2
    # E = torch.tanh(10 * E)
    # Another option:
    # E = A * self.squared_distance_matrix_based_on_kernel(v)
    return E


def squared_distance_matrix_based_on_kernel(v):
    # Calculate the kernel K = w @ w.T via outer product
    K = v @ v.T

    # Extract diagonal elements of K
    diag_K = diag(K)
    E = (diag_K[:, None] - 2 * K + diag_K[None, :])

    return E


def hamiltonian(A, v, diagonal_scale):
    E = graph_edit_matrix(A, v)
    H = laplacian(A - E) + diagonal_scale * diag(v.squeeze())
    return H
