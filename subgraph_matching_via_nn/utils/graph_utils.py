import torch
from torch import diag, tensor
import networkx as nx
import numpy as np


def get_node_indicator(G: nx.graph, G_sub: nx.graph):
    """
    Create node indicator for G_sub in G (assuming G_sub was extracted from G)

    :param G: A networkx graph
    :param G_sub: A networkx sub-graph of G
    :return: w_indicator - a vector with w[i] ==1 if node i of G is a node in G_sub
    , otherwise w_indicator[i]==0.
    """
    # Set the indices corresponding to the subgraph nodes to 1
    subgraph_node_indices = [list(G.nodes()).index(node) for node in G_sub.nodes()]
    # subgraph_node_indices = list(G_sub.nodes())
    w_indicator = np.zeros(len(G.nodes()))
    w_indicator[subgraph_node_indices] = 1.0
    return w_indicator


def get_edge_indicator(G: nx.graph, G_sub: nx.graph):
    """
    Create edge indicator for G_sub in G (assuming G_sub was extracted from G)

    :param G: A networkx graph
    :param G_sub: A networkx sub-graph of G
    :return: edge_indicator - dict with values
    edge_indicator[(i,j)] == edge_indicator[(j,i)] ==1 if (i,j) is an edge of G_sub,
    and 0 otherwise.
    """
    # edge_indicator = \
    #     {(min(u, v), max(u, v)): 1 if (min(u, v), max(u, v))
    #                                   in G_sub.edges() else 0 for u, v in G.edges()}

    edge_indicator = \
        {(u, v): 1 if (u, v) in G_sub.edges() else 0 for (u, v) in G.edges()}
    # Create symmetric adjacency matrix
    num_nodes = len(G.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for (i, j), val in edge_indicator.items():
        adj_matrix[i][j] = val
        adj_matrix[j][i] = val  # Ensure it's symmetric

    return edge_indicator, adj_matrix


def node_indicator_from_edge_indicator(G: nx.graph, edge_indicator):
    # Create node incident vector
    w = [0] * len(G.nodes())
    for node in G.nodes():
        # Get incident edges for node
        incident_edges = [(node, neighbor) if (node, neighbor) in list(G.edges()) else
                          (neighbor, node) for neighbor in
                          G.neighbors(node)]

        # Calculate average of edge_indicator values for the incident edges
        edge_indicator_values = [edge_indicator[edge] for edge in incident_edges]
        node_indicator_value = 0
        if len(edge_indicator_values) != 0:
            node_indicator_value = max(edge_indicator_values)

        w[list(G.nodes).index(node)] = float(node_indicator_value)
    return w


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
