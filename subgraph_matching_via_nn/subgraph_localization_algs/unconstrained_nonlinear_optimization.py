from typing import Optional

import networkx as nx
import torch
from livelossplot import PlotLosses
from torch import optim
import numpy as np
from subgraph_matching_via_nn.composite_nn.compiste_nn import CompositeNeuralNetwork
from subgraph_matching_via_nn.graph_metric_networks.graph_matric_nn import \
    GraphMetricNetwork
from subgraph_matching_via_nn.graph_processors.graph_processors import \
    BaseGraphProcessor, GraphProcessor
from subgraph_matching_via_nn.utils.utils import uniform_dist


def nn_subgraph_localization(G: nx.graph,
                             G_sub: nx.graph,
                             composite_nn: CompositeNeuralNetwork,
                             graph_metric_nn: GraphMetricNetwork,
                             params: dict,
                             graph_processor: Optional[BaseGraphProcessor] =
                             GraphProcessor(),
                             dtype=torch.double):
    # Create your optimizer
    lr = params['lr']
    reg_term = params['reg_term']
    optimizer = optim.SGD(composite_nn.parameters(), lr=lr)
    x0 = params.get("x0", None)
    liveloss = PlotLosses(mode='notebook')

    # preprocess the graphs, e.g. to get a line-graph
    G = graph_processor.pre_process(G)
    G_sub = graph_processor.pre_process(G_sub)
    A = torch.tensor(nx.to_numpy_array(G)).type(dtype)
    A_sub = torch.tensor(nx.to_numpy_array(G_sub)).type(dtype)
    embedding_sub = composite_nn.embedding_network(A=A_sub.detach().type(dtype),
                                                   w=uniform_dist(
                                                       A_sub.shape[0]).detach())
    # Set the model to training mode
    composite_nn.train()
    for iteration in range(params["maxiter"]): # TODO: add stopping condition
        embedding_full, w = composite_nn(A, x0)
        loss = graph_metric_nn(embedding_full=embedding_full,
                               embedding_subgraph=embedding_sub)  # + regularization

        reg = reg_term(A, w, params)
        full_loss = loss + params["reg_param"] * reg

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        if iteration % params['k_update_plot'] == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            print(f"Iteration {iteration}, Reg: {reg.item()}")
            print(f"Iteration {iteration}, Loss + rho * Reg: {full_loss.item()}")
            liveloss.update({'loss': loss.item()})
            liveloss.send()


def edited_Laplacian(A, w):
    A_edited = A - A * ((w - w.T) ** 2)
    L = (torch.diag(A_edited.sum(axis=1)) - A_edited)
    return L



# regularization terms

def spectral_reg(A, w, params):
    L_edited = edited_Laplacian(A, w * (params["m"] ** 2))
    reg = w.T @ L_edited @ w
    return reg


def graph_total_variation(A, w, params):
    # todo: should check why it doesn't give 0 with gt mask
    diff = torch.abs(torch.matmul(A, w))
    total_variation = torch.sum(diff)
    return total_variation


def graph_entropy(A, w, params):
    H = - w.T @ torch.log(w)
    return H


def binary_penalty(A, w, params):
    # reg = torch.norm(w * (1/params["m"] - w), p=2) ** 2
    reg = torch.sum(w * (1 / params["m"] - w) ** 2)
    return reg
