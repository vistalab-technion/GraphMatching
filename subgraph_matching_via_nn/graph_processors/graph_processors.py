from logging import exception
import networkx as nx
import numpy as np
import scipy as sp

from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.utils.graph_utils import laplacian, graph_edit_matrix
from subgraph_matching_via_nn.utils.utils import NP_DTYPE, top_m


class BaseGraphProcessor:
    def __init__(self):
        super().__init__()

    def pre_process(self, sub_graph: SubGraph):
        pass

    def post_process(self, graph, w):
        pass


class GraphProcessor(BaseGraphProcessor):

    def __init__(self, params: dict = {'to_undirected': None, 'to_line': False}):
        super().__init__()
        self._to_line = params.get("to_line", None)
        self._to_undirected = params.get("to_undirected", None)

    def pre_process(self, sub_graph: SubGraph):
        processed_G = sub_graph.G
        edge_indicator = sub_graph.edge_indicator
        original_node_indicator = sub_graph.node_indicator
        is_line_graph = False

        # performing a sequence of operations on the graph as a pre-process
        if self._to_undirected is not None:
            if self._to_undirected == 'symmetrize':
                processed_G = nx.to_undirected(processed_G)
            else:
                raise exception(f"{self._to_undirected} not supported yet")
        if self._to_line:
            is_line_graph = True
            processed_G = nx.line_graph(processed_G)
            if edge_indicator is not None:
                edge_indicator = original_node_indicator

        if edge_indicator is not None:
            G_sub_as_sub_graph = SubGraph(sub_graph.G_sub, None)
            processed_G_sub = self.pre_process(G_sub_as_sub_graph)
            sub_graph = SubGraph(processed_G, processed_G_sub, is_line_graph=is_line_graph, original_graph=sub_graph.G,
                                 original_subgraph=sub_graph.G_sub)
            return sub_graph
        else:
            return processed_G
