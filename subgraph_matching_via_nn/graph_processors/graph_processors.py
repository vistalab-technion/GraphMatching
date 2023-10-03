from logging import exception
from typing import Optional

import kmeans1d
import networkx as nx
import numpy as np

from subgraph_matching_via_nn.utils.graph_utils import laplacian
from subgraph_matching_via_nn.utils.utils import NP_DTYPE, top_m


class BaseGraphProcessor:
    def __init__(self):
        super().__init__()

    def pre_process(self, graph, w):
        pass

    def post_process(self, graph, w):
        pass


class GraphProcessor(BaseGraphProcessor):

    def __init__(self, params: dict = {'to_undirected': None, 'to_line': False}):
        super().__init__()
        self._to_line = params.get("to_line", None)
        self._to_undirected = params.get("to_undirected", None)

    def pre_process(self, graph: nx.Graph, edge_indicator=None, node_indicator=None):
        # performing a sequence of operations on the graph as a pre-process
        if self._to_undirected is not None:
            if self._to_undirected == 'symmetrize':
                graph = nx.to_undirected(graph)
            else:
                raise exception(f"{self._to_undirected} not supported yet")
        if self._to_line:
            graph = nx.line_graph(graph)
            if edge_indicator is not None:
                node_indicator = np.array([edge_indicator[edge] for edge in
                                           graph.nodes()])

        if edge_indicator is not None:
            return graph, node_indicator
        else:
            return graph

    @staticmethod
    def binarize(graph: nx.graph, w: np.array, params, type='top_m'):
        if type == 'k_means':
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == 'top_m':
            w_th = top_m(w, params["m"])
        elif type == 'quantile':
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        elif type == 'diffusion':
            A = (nx.adjacency_matrix(graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            # Eigenvalue decomposition of the Laplacian
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            k = 10

            # Generate k logarithmically spaced values of t from a large to small value
            max_t = 100  # Change this to your desired maximum value
            min_t = 0.01  # Change this to your desired minimum value
            t_values = np.logspace(np.log10(max_t), np.log10(min_t), k)

            w_th = w
            for t in t_values:
                # Apply the heat kernel using matrix exponentiation
                heat_matrix = eigenvectors @ np.diag(
                    np.exp(-t * eigenvalues)) @ eigenvectors.T
                heat_w = heat_matrix @ w

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])

        w_th = w_th / w_th.sum()
        w_th_dict = dict(zip(graph.nodes(), w_th))
        return w_th_dict
