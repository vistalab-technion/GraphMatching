import torch


def edited_Laplacian(A, v):
    E = A * (v - v.T) ** 2
    A_edited = A - E
    L_edited = torch.diag(A_edited.sum(axis=1)) - A_edited
    return L_edited


# regularization terms

def spectral_reg(A, w, params):
    L_edited = edited_Laplacian(A, 1 - w * (params["m"]))
    reg = torch.norm(L_edited @ w, p=2) ** 2
    return reg


def graph_total_variation(A, w, params):
    # todo: should check why it doesn't give 0 with gt mask
    L = torch.diag(A.sum(dim=1)) - A
    total_variation = (0.5 * w.T @ L @ w).squeeze()
    return total_variation


def graph_entropy(A, w, params):
    H = - w.T @ torch.log(w)
    return H


def binary_penalty(A, w, params):
    # reg = torch.norm(w * (1/params["m"] - w), p=2) ** 2
    reg = torch.sum(w * (1 / params["m"] - w) ** 2)
    return reg
