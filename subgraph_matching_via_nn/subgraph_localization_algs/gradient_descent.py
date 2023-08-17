import torch
from livelossplot import PlotLosses
from torch import optim
import numpy as np
from subgraph_matching_via_nn.composite_nn.compiste_nn import CompositeNeuralNetwork
from subgraph_matching_via_nn.graph_metric_networks.graph_matric_nn import GraphMetricNetwork


def nn_subgraph_localization(A: torch.Tensor,
                             subgraph_embedding_nn: CompositeNeuralNetwork,
                             graph_metric_nn: GraphMetricNetwork,
                             params: dict):
    # Create your optimizer
    lr = params['lr']
    optimizer = optim.SGD(subgraph_embedding_nn.parameters(), lr=lr)

    liveloss = PlotLosses(mode='notebook')
    # Set the desired figure size (width, height)
    x0 = params.get("x0", None)
    embedding_sub = params['embedding_gt']
    for iteration in range(params["maxiter"]):
        # Set the model to training mode
        subgraph_embedding_nn.train()
        embedding_full, w = subgraph_embedding_nn(A, x0)
        loss = graph_metric_nn(embedding_full=embedding_full,
                               embedding_subgraph=embedding_sub)  # + regularization

        reg = binary_penalty(A, w, params)
        full_loss = loss + params["reg_param"] * reg

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        if iteration % params['k_update_plot'] == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            liveloss.update({'loss': loss.item()})
            liveloss.send()


def edited_Laplacian(A, w):
    A_edited = A - A * ((w - w.T) ** 2)
    L = (torch.diag(A_edited.sum(axis=1)) - A_edited)
    return L


def spectral_reg(A, w, params):
    L_edited = edited_Laplacian(A, w * (params["m"] ** 2))
    reg = w.T @ L_edited @ w
    return reg


def graph_total_variation(A, w, params):
    diff = torch.abs(torch.matmul(A, w))
    total_variation = torch.sum(diff)
    return total_variation


def graph_entropy(A, w, params):
    H = - w.T @ torch.log(w)
    return H


def binary_penalty(A, w, params):
    #reg = torch.norm(w * (1/params["m"] - w), p=2) ** 2
    reg = torch.sum(w * (1/params["m"] - w) ** 2)
    return reg


