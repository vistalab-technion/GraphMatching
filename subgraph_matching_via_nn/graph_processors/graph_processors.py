from logging import exception
from typing import Optional

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

    def __init__(self,params : dict  =  {'to_undirected' : None, 'to_line':  False}):
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


