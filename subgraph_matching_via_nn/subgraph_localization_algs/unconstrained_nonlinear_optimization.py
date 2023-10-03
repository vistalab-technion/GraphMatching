from typing import Optional

import networkx as nx
import torch
from livelossplot import PlotLosses
from torch import optim
import numpy as np
from tqdm import tqdm

from subgraph_matching_via_nn.composite_nn.composite_nn import CompositeNeuralNetwork
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import \
    EmbeddingMetricNetwork
from subgraph_matching_via_nn.graph_processors.graph_processors import \
    BaseGraphProcessor, GraphProcessor
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix, laplacian, \
    hamiltonian
from subgraph_matching_via_nn.utils.utils import uniform_dist


def nn_subgraph_localization(G: nx.graph,
                             G_sub: nx.graph,
                             composite_nn: CompositeNeuralNetwork,
                             embedding_metric_nn: EmbeddingMetricNetwork,
                             params: dict,
                             graph_processor: Optional[BaseGraphProcessor] =
                             GraphProcessor(),
                             dtype=torch.double):
    # Create your optimizer
    lr = params['lr']
    x0 = params.get("x0", None)
    liveloss = PlotLosses(mode='notebook')

    # preprocess the graphs, e.g. to get a line-graph
    G = graph_processor.pre_process(G)
    G_sub = graph_processor.pre_process(G_sub)
    A = torch.tensor(nx.to_numpy_array(G)).type(dtype)
    A_sub = torch.tensor(nx.to_numpy_array(G_sub)).type(dtype)
    embeddings_sub = composite_nn.embed(A=A_sub.detach().type(dtype),
                                        w=uniform_dist(A_sub.shape[0]).detach())

    # Set the model to training mode
    composite_nn.train()

    # set solver
    solver_type = params.get("solver_type", None)
    if solver_type == 'gd':
        optimizer = optim.SGD(params=composite_nn.parameters(), lr=lr)
    elif solver_type == 'lbfgs':
        optimizer = optim.LBFGS(params=composite_nn.parameters(), lr=lr, max_iter=5,
                                max_eval=None,
                                tolerance_grad=1e-07,
                                tolerance_change=1e-09,
                                history_size=10,
                                line_search_fn=None)
    else:
        raise ValueError(f"Unknown optimizer choice: {solver_type}")

    def evaluate_objective(A, x0, composite_nn, embedding_metric_nn, params):
        embeddings_full, w = composite_nn(A, x0, params)
        data_term = embedding_metric_nn(embeddings_full=embeddings_full,
                                        embeddings_subgraph=embeddings_sub)  # + regularization

        reg_term = torch.stack(
            [reg_param * reg_term(A, w, params) for reg_param, reg_term in
             zip(params["reg_params"], params["reg_terms"])]).sum()
        loss = data_term + reg_term
        return loss, data_term, reg_term

    def closure():
        optimizer.zero_grad()  # Zero out the previous gradient
        loss, _, _ = evaluate_objective(A, x0, composite_nn, embedding_metric_nn,
                                        params)
        loss.backward()
        return loss

    # run iterations
    for iteration in tqdm(range(params["maxiter"])):  # TODO: add stopping condition
        optimizer.step(closure)
        if iteration % params['k_update_plot'] == 0:
            with torch.no_grad():
                loss, _, _ = evaluate_objective(A, x0, composite_nn,
                                                embedding_metric_nn,
                                                params)
                liveloss.update({'loss': loss.item()})
                liveloss.send()

    # print final loss
    with torch.no_grad():
        loss, data_term, reg_term = evaluate_objective(A, x0, composite_nn,
                                                       embedding_metric_nn,
                                                       params)
        print(f"Iteration {iteration}, Data: {data_term.item()}")
        print(f"Iteration {iteration}, Reg: {reg_term.item()}")
        print(f"Iteration {iteration}, Data + rho * Reg: {loss.item()}")


# regularization terms

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


def graph_entropy(A, w, params):
    H = - w.T @ torch.log(w)
    return H


def binary_penalty(A, w, params):
    # reg = torch.norm(w * (1/params["m"] - w), p=2) ** 2
    reg = torch.sum(w * (1 / params["m"] - w) ** 2)
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
