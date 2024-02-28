import torch

# regularization terms
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix, laplacian, hamiltonian


def spectral_reg(A, w, params):
    v = 1 - w * (params["m"])
    E = graph_edit_matrix(A, v)
    L_edited = laplacian(A - E)
    reg = torch.norm(L_edited @ v, p=2) ** 2
    return reg


def graph_total_variation(A, w, params):
    # todo: should check why it doesn't give 0 with gt mask
    L = torch.diag(A.sum(dim=1)) - A
    total_variation = (0.5 * w.T @ L @ w).squeeze()
    return total_variation

def graph_nodes_number(A, w, params):
    scaler = params['scaler']
    raw_diff = torch.sum(w) - params['m']
    reg = torch.pow(raw_diff / scaler, 2)
    return reg

def graph_edges_number(A, w, params):
    scaler = params['scaler']
    raw_diff = torch.sum(w * params['m'] * A * w.T * params['m']) / 2 - params['n']
    reg = torch.pow(raw_diff / scaler, 2)
    return reg

def graph_entropy(A, w, params):
    H = - w.T @ torch.log(w)
    return H


def binary_penalty(A, w, params):
    scaler = params['scaler']
    reg = torch.sum(((w ** 2) / scaler * (((w-1/params['m']) / scaler) ** 2)))
    return reg


def log_barrier_penalty(A, w, params):
    v = 1 - w * (params["m"])
    evals, evecs = torch.linalg.eigh(hamiltonian(A, v, params["diagonal_scale"]))
    lambda_second = evals[1]
    c = params['second_eig']
    if lambda_second <= c:
        return torch.inf  # or some large value to signify the penalty
    else:
        return -torch.log(lambda_second - params['second_eig'])
