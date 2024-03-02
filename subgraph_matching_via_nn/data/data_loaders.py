import pickle
from logging import exception, error

import numpy as np
import networkx as nx

from subgraph_matching_via_nn.graph_generators.util import generate_random_tree, \
    sample_connected_subgraph, generate_wheel_graph, generate_random_graph, \
    generate_graph_with_unique_degrees
from subgraph_matching_via_nn.utils.graph_utils import get_node_indicator, \
    get_edge_indicator

GRAPH_TYPES = ['random', 'random_tree', 'wheel', 'unique_degree', 'example',
               'subcircuit']


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
    if type in GRAPH_TYPES:

        graph_size = loader_params["graph_size"]
        subgraph_size = loader_params["subgraph_size"]

        # Set the size of the graph and the subgraph
        n = graph_size  # Number of nodes in the graph
        m = subgraph_size  # Number of nodes in the subgraph

        # TODO: add this to the choice, change to enum
        if type == 'random':
            G = generate_random_graph(n)
        elif type == 'random_tree':
            G = generate_random_tree(n)
        elif type == 'wheel':
            G = generate_wheel_graph(n)
        elif type == 'unique_degree':
            G = generate_graph_with_unique_degrees(n)
        else:
            error('error, graph type not supported')

        # Generate a random subset of nodes for the subgraph
        # subgraph_nodes = np.random.choice(G.nodes(), size=m, replace=False)
        subgraph_nodes = sample_connected_subgraph(G=G, m=m)

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

    elif type == 'subcircuit':

        def remove_isolated_nodes_from_graph(graph):
            isolated_nodes_indices = list(nx.isolates(graph))
            graph.remove_nodes_from(isolated_nodes_indices)

        g_full_path = loader_params['data_path'] + loader_params['g_full_path']
        g_sub_path = loader_params['data_path'] + loader_params['g_sub_path']

        G = pickle.load(open(g_full_path, 'rb'))
        G_sub = pickle.load(open(g_sub_path, 'rb'))

        remove_isolated_nodes_from_graph(G_sub)

        node_indicator = get_node_indicator(G=G, G_sub=G_sub)
        edge_indicator, subgraph_adj_matrix = get_edge_indicator(G=G, G_sub=G_sub)

    else:
        raise exception(f"type = {type} not supported")

    return G, G_sub, node_indicator, edge_indicator
