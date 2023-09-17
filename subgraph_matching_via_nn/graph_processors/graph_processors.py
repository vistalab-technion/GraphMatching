from logging import exception
from typing import Optional

import kmeans1d
import networkx as nx
import numpy as np

from subgraph_matching_via_nn.utils.utils import NP_DTYPE


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
            indices_of_top_m = np.argsort(w, axis=0)[-params["m"]:]  # top m
            w_th = np.zeros_like(w, dtype=NP_DTYPE)
            w_th[indices_of_top_m] = 1
        elif type == 'quantile':
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        w_th = w_th / w_th.sum()
        w_th_dict = dict(zip(graph.nodes(), w_th))

        return w_th_dict
