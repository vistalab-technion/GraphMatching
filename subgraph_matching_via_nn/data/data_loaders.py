import pickle
from logging import exception
import networkx as nx
import numpy as np
import torch

from subgraph_matching_via_nn.data.graph_constants import GraphConstants
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.graph_generators.util import generate_random_tree, \
    sample_connected_subgraph, generate_wheel_graph, generate_random_graph
from subgraph_matching_via_nn.utils.utils import extract_node_features_from_graph, set_node_features_for_graph


def load_graph(type: str = 'random',
               loader_params: dict = {'graph_size': 30, 'subgraph_size': 10}):
    """
    :param loader_params: dict of parameters for loader
    :param type: 'random' - random graph
                 'example' some fixed example
    :return: SubGraph
    """
    if type == 'random':

        graph_size = loader_params["graph_size"]
        subgraph_size = loader_params["subgraph_size"]

        # Set the size of the graph and the subgraph
        n = graph_size  # Number of nodes in the graph
        m = subgraph_size  # Number of nodes in the subgraph

        # TODO: add this to the choice, change to enum
        G = generate_random_graph(n)
        # G = generate_random_tree(n)
        # G = generate_wheel_graph(n)

        # Generate a random subset of nodes for the subgraph
        # subgraph_nodes = np.random.choice(G.nodes(), size=m, replace=False)
        subgraph_nodes = sample_connected_subgraph(G=G, m=m)

        # Create the subgraph by keeping only the edges that connect the selected subset of nodes
        G_sub = G.subgraph(subgraph_nodes)

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

    elif type == 'subcircuit':

        def remove_isolated_nodes_from_graph(graph):
            isolated_nodes_indices = list(nx.isolates(graph))
            graph.remove_nodes_from(isolated_nodes_indices)

        def one_hot_encode_node_features(graph, unique_features, feature_name):
            node_features = extract_node_features_from_graph(graph, feature_name)
            feature_map = {feature: np.eye(len(unique_features))[i] for i, feature in enumerate(unique_features)}
            feature_map[np.nan] = np.full((len(unique_features)), np.nan) #handle missing node

            encoded_features = torch.from_numpy(np.array([feature_map[feature] for feature in node_features])).to(
                dtype=torch.double)
            set_node_features_for_graph(graph, feature_name, encoded_features)

        g_full_path = loader_params['data_path'] + loader_params['g_full_path']
        g_sub_path = loader_params['data_path'] + loader_params['g_sub_path']

        G = pickle.load(open(g_full_path, 'rb'))
        G_sub = pickle.load(open(g_sub_path, 'rb'))

        remove_isolated_nodes_from_graph(G_sub)

        G.remove_edges_from(nx.selfloop_edges(G))
        G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

        feature_name = GraphConstants.NODE_GATE_TYPE_ATTRIBUTE_NAME

        # TODO: this should be replaced with all the possible catergories list, to make sure the encodings are consistent
        unique_features = list(set(extract_node_features_from_graph(G, feature_name)))

        one_hot_encode_node_features(G, unique_features, feature_name)
        one_hot_encode_node_features(G_sub, unique_features, feature_name)

    else:
        raise exception(f"type = {type} not supported")

    return SubGraph(G, G_sub)

#
# # Set the size of the graph and the subgraph
# n = 20  # Number of nodes in the graph (for random graph)
# m = 7  # Number of nodes in the subgraph (for random graph)
# seed = 10  # for plotting
# loader_params = {'graph_size': 30, 'subgraph_size': 10}
# loader_params['g_full_path'] = '/Users/amitboy/PycharmProjects/GraphMatching/subgraph_matching_via_nn/data/subcircuits/compound1/comp1_32_full_graph.p'
# loader_params['g_sub_path'] = '/Users/amitboy/PycharmProjects/GraphMatching/subgraph_matching_via_nn/data/subcircuits/compound1/comp1_32_subgraph0.p'
# sub_graph =\
#     load_graph(type='subcircuit', loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'
#
# print('done')
