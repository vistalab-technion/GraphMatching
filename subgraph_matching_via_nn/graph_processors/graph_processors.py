from logging import exception
import networkx as nx
import numpy as np

from subgraph_matching_via_nn.data.sub_graph import SubGraph


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
        node_indicator = sub_graph.node_indicator
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
                node_indicator = np.array([edge_indicator[edge] for edge in
                                           processed_G.nodes()])

        if edge_indicator is not None:
            G_sub_as_sub_graph = SubGraph(sub_graph.G_sub, None, None, None)
            processed_G_sub = self.pre_process(G_sub_as_sub_graph)
            sub_graph = SubGraph(processed_G, processed_G_sub, node_indicator, edge_indicator, is_line_graph=is_line_graph)
            return sub_graph
        else:
            return processed_G
