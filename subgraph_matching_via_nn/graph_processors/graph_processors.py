import networkx as nx

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

    def pre_process(self, graph: nx.graph, w = None):
        # performing a sequence of operations on the graph as a pre-process
        if self._to_bipartite:
            pass
        if self._to_line:
            return nx.line_graph(graph)
        else:
            return graph

    def post_process(self, graph: nx.graph, w = None):
        # performing a sequence of operations on the graph as a post-process
        if self._to_bipartite:
            pass
        if self._to_line:
            return nx.line_graph(graph)
