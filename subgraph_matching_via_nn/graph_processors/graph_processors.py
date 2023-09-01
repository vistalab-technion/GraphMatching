from logging import exception

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

    def __init__(self,
                 to_bipartite : bool = False,
                 to_line: bool = False):
        super().__init__()
        self._to_line = to_line
        self._to_bipartite = to_bipartite

    def pre_process(self, graph: nx.graph, edge_indicator = None):
        # performing a sequence of operations on the graph as a pre-process
        if self._to_bipartite:
            raise exception(f"to biparatite not supported yet")
        if self._to_line:
            graph = nx.line_graph(graph)
            if edge_indicator is not None:
                node_indicator_line = np.array([edge_indicator[edge] for edge in
                                                graph.nodes()])
                return graph, node_indicator_line
            else:
                return graph


