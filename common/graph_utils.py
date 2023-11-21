import networkx as nx


def relabel_graph_nodes_by_contiguous_order(g: nx.Graph, copy):
    nodes_map = dict(zip(list(g.nodes), [i for i in range(len(g))]))
    graph = nx.relabel_nodes(g, nodes_map, copy=copy)
    return graph