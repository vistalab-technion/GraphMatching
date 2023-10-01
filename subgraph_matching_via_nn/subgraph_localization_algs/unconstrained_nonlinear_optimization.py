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
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix, laplacian
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
    optimizer = optim.SGD(composite_nn.parameters(), lr=lr)
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

    for iteration in tqdm(range(params["maxiter"])):  # TODO: add stopping condition
        embeddings_full, w = composite_nn(A, x0, params)
        loss = embedding_metric_nn(embeddings_full=embeddings_full,
                                   embeddings_subgraph=embeddings_sub)  # + regularization

        reg = torch.stack([reg_param * reg_term(A, w, params) for reg_param, reg_term in
                           zip(params["reg_params"], params["reg_terms"])]).sum()
        full_loss = loss + reg

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        if iteration % params['k_update_plot'] == 0:
            liveloss.update({'loss': loss.item()})
            liveloss.send()

    print(f"Iteration {iteration}, Loss: {loss.item()}")
    print(f"Iteration {iteration}, Reg: {reg.item()}")
    print(f"Iteration {iteration}, Loss + rho * Reg: {full_loss.item()}")


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
