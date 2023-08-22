import networkx as nx
from networkx import graph


def graph_to_line_graph(G):
    line_graph = nx.line_graph(G)
    return line_graph


# class MyGraph(graph):
#     def __init__(self):
#         super().__init__()
#
#     def graph_to_line_graph(self):
#         pass
#
#     def graph_to_line_graph(self):