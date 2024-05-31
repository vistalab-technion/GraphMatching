from typing import Dict, Tuple
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

    return edge_indicator


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


def get_normalized_node_indicator(raw_node_indicator, dtype):
    gt_indicator_tensor = torch.tensor(raw_node_indicator)[:, None].type(dtype)
    gt_indicator_tensor = gt_indicator_tensor / gt_indicator_tensor.sum()

    return gt_indicator_tensor

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

def adjacency_matrix_to_edges(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                edges.append((i, j))
    return edges

def total_edges_between_nodes_of_degrees(edge_weight_matrix, in_degrees, degree_i, degree_j):
    nodes_of_degree_i = np.argwhere(in_degrees==degree_i)
    nodes_of_degree_j = np.argwhere(in_degrees==degree_j)
    if len(nodes_of_degree_i) == 0:
        return 0
    if len(nodes_of_degree_j) == 0:
        return 0

    matched_edges_weights = edge_weight_matrix[nodes_of_degree_i].T[nodes_of_degree_j]

    if len(matched_edges_weights) == 0:
        return 0

    return matched_edges_weights.sum()

def joint_degree_matrix(A, edge_weight_matrix=None):
    if edge_weight_matrix is None:
        edge_weight_matrix = A
    else:
        edge_weight_matrix = edge_weight_matrix * A # zero out entries of edges not in A

    # out_degrees = np.sum(A, axis=1).reshape(-1)
    in_degrees = np.sum(A, axis=0).reshape(-1)

    n = A.shape[0]

    output_matrix = np.array([
        np.array([total_edges_between_nodes_of_degrees(edge_weight_matrix, in_degrees, degree_i, degree_j) for degree_j in range(n)])
        for degree_i in range(n)

    ])

    return output_matrix


def create_weighted_adjacency_matrix_from_edge_mask(edge_mask: Dict[Tuple[int, int], float], num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for edge, mask_val in edge_mask.items():
        source, target = edge
        adjacency_matrix[source][target] = adjacency_matrix[target][source] = mask_val
    return adjacency_matrix


def create_weighted_adjacency_matrix_from_node_mask(node_mask: Dict[int, float], num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for node, mask_val in node_mask.items():
        adjacency_matrix[node, :] = mask_val * np.ones(num_nodes)
        adjacency_matrix[:, node] = mask_val * np.ones(num_nodes)
    return adjacency_matrix