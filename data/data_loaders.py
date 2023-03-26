import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from problem.spectral_subgraph_localization import lap_from_adj, \
    ProximalSubgraphIsomorphismSolver


def load_data(target, pattern):



    G_target = nx.read_adjlist(target)
    G_pattern=nx.read_adjlist(pattern)
    A = torch.tensor(nx.to_numpy_matrix(G_target))
    A_sub = torch.tensor(nx.to_numpy_matrix(G_pattern))


    return A, A_sub