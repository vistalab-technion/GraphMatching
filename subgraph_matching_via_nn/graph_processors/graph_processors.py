from logging import exception
from typing import Optional

import kmeans1d
import networkx as nx
import numpy as np
import scipy as sp

from subgraph_matching_via_nn.graph_processors.binarization_algos import \
    solve_maximum_weight_subgraph
from subgraph_matching_via_nn.utils.graph_utils import laplacian, graph_edit_matrix, \
    edge_indicator_from_node_indicator
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

    # @staticmethod
    def binarize(self, graph: nx.graph, w: np.array, params,
                 type='top_m'):
        processed_graph = self.pre_process(graph)
        if type == 'k_means':
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == 'top_m':
            w_th = top_m(w, params["num_nodes"])
        elif type == 'quantile':
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        elif type == 'diffusion':
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            # # Eigenvalue decomposition of the Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            k = 10

            # Generate k logarithmically spaced values of t from a large to small value
            max_t = 10  # Change this to your desired maximum value
            min_t = 0.01  # Change this to your desired minimum value
            t_values = np.logspace(np.log10(max_t), np.log10(min_t), k)

            w_th = w
            for t in t_values:
                # Apply the heat kernel using matrix exponentiation
                heat_matrix = eigenvectors @ np.diag(
                    np.exp(-t * eigenvalues)) @ eigenvectors.T
                heat_w = heat_matrix @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["num_nodes"])
        elif type == 'zoomout':
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            w_th = w
            for i in range(2, A.shape[0]):
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["num_nodes"])

        elif type == 'nonlinear_zoomout':
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            w_th = w
            heat_w = w
            for i in range(2, A.shape[0]):
                E = graph_edit_matrix(A, 1 - params["num_nodes"] * w_th)
                Ae = A - E

                De = np.diag(Ae.sum(axis=1))
                Le = De - Ae

                # Generalized eigenvalue decomposition of the Random Walk Laplacian
                eigenvalues, eigenvectors = np.linalg.eigh(Le)
                # eigenvalues, eigenvectors = sp.linalg.eigh(Le, De)
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["num_nodes"])
        elif type == 'mwksp':
            # todo: switch beforehand to original graph,
            #  and add maxium weighted edge subgraph algorithm

            # no need to process the graph
            processed_graph = self.pre_process(graph)

            A = (nx.adjacency_matrix(processed_graph)).toarray()
            selected_nodes, selected_edges = solve_maximum_weight_subgraph(w, A, params[
                "num_nodes"], params["num_edges"])
            print(f'{selected_nodes=}')
            print(f'{selected_edges=}')
            print(
                f'requested: n_nodes = {params["num_nodes"]}, n_edges : {params["num_edges"]}')
            print(
                f'found: n_nodes = {len(selected_nodes)}, n_edges : {len(selected_edges)}')
            w_th = np.zeros([len(processed_graph.nodes()), 1])
            w_th[selected_nodes] = 1.0
        else:
            w_th = w

        w_th = w_th / w_th.sum()

        if self._to_line:
            w_th_dict = dict(zip(processed_graph.nodes(), w_th))
        else:
            if type != 'mwksp':
                w_th_dict = edge_indicator_from_node_indicator(graph, w_th)
            else:
                w_th_dict = {edge: (1 if edge in selected_edges else 0)
                             for edge in graph.edges()}

        return w_th_dict
