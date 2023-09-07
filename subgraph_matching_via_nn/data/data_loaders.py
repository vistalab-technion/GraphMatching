from logging import exception

import numpy as np
import networkx as nx

from subgraph_matching_via_nn.utils.utils import get_node_indicator, get_edge_indicator


def load_graph(type: str = 'random',
               loader_params: dict = {'graph_size': 30, 'subgraph_size': 10}):
    """
    :param loader_params: dict of parameters for loader
    :param type: 'random' - random graph
                 'example' some fixed example
    :return:  G - networkx graph object
              G_sub - subgraph
              node_indicator - node indicator (i.e., 1 for nodes of G_sub that are in G)
              edge_indicator -
              edge indicator (i.e., dict[(i,j)]= 1 for edges of G_sub that are in G)
    """
    if type == 'random':

        graph_size = loader_params["graph_size"]
        subgraph_size = loader_params["subgraph_size"]

        # Set the size of the graph and the subgraph
        n = graph_size  # Number of nodes in the graph
        m = subgraph_size  # Number of nodes in the subgraph

        # Generate a random adjacency matrix A
        A_upper = np.triu(np.random.randint(0, 2, size=(n, n)), k=1)
        A = A_upper + A_upper.T

        # Set diagonal elements to zero to remove self-loops
        np.fill_diagonal(A, 0)

        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(A)
        # Generate a random subset of nodes for the subgraph
        subgraph_nodes = np.random.choice(G.nodes(), size=m, replace=False)

        # Create the subgraph by keeping only the edges that connect the selected subset of nodes
        G_sub = G.subgraph(subgraph_nodes)

        node_indicator = get_node_indicator(G=G, G_sub=G_sub)
        edge_indicator, subgraph_adj_matrix = get_edge_indicator(G=G, G_sub=G_sub)

    elif type == 'example':
        # A fixed synthetic example

        # generate the adjacency matrices
        circuit_edges = [(0, 1), (5, 10), (10, 11), (0, 2), (0, 3), (0, 4), (1, 5),
                         (2, 6), (6, 12), (6, 13), (6, 14), (6, 15), (6, 7), (4, 8),
                         (8, 9), (0, 6)]
        subcircuit_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (
            4, 8)]  # the subgraph has the nodes 0 and 6 but not the edge (0,6)

        # Create the graph from the adjacency matrices
        G = nx.from_edgelist(circuit_edges)
        G_sub = G.edge_subgraph(subcircuit_edges)

        node_indicator = get_node_indicator(G=G, G_sub=G_sub)
        edge_indicator, subgraph_adj_matrix = get_edge_indicator(G=G, G_sub=G_sub)

    elif 'subcircuit':
        import pickle

        def remove_isolated_nodes_from_graph(graph):
            isolated_nodes_indices = list(nx.isolates(graph))
            graph.remove_nodes_from(isolated_nodes_indices)

        g_full_path = loader_params['g_full_path']
        g_sub_path = loader_params['g_sub_path']

        G = pickle.load(open(g_full_path, 'rb'))
        G_sub = pickle.load(open(g_sub_path, 'rb'))

        remove_isolated_nodes_from_graph(G_sub)

        node_indicator = get_node_indicator(G=G, G_sub=G_sub)
        edge_indicator, subgraph_adj_matrix = get_edge_indicator(G=G, G_sub=G_sub)

        pass
    else:
        raise exception(f"type = {type} not supported")

    return G, G_sub, node_indicator, edge_indicator

#
# # Set the size of the graph and the subgraph
# n = 20  # Number of nodes in the graph (for random graph)
# m = 7  # Number of nodes in the subgraph (for random graph)
# seed = 10  # for plotting
# loader_params = {'graph_size': 30, 'subgraph_size': 10}
# loader_params['g_full_path'] = '/Users/amitboy/PycharmProjects/GraphMatching/subgraph_matching_via_nn/data/subcircuits/compound1/comp1_32_full_graph.p'
# loader_params['g_sub_path'] = '/Users/amitboy/PycharmProjects/GraphMatching/subgraph_matching_via_nn/data/subcircuits/compound1/comp1_32_subgraph0.p'
# G, G_sub, w_indicator, edge_indicator =\
#     load_graph(type='subcircuit', loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'
#
# print('done')
