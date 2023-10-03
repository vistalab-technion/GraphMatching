import random
import networkx as nx
import numpy as np


def generate_random_graph(n):
    G = nx.erdos_renyi_graph(n=n, p=0.5)
    return G


def generate_barbell_graph(m1, m2):
    """
    Generate a barbell graph.

    Parameters:
    - m1: Number of nodes in the two complete graphs.
    - m2: Number of nodes in the path connecting the two complete graphs.

    Returns:
    - A networkx Graph object representing the barbell graph.
    """

    # Use built-in function to create a barbell graph
    G = nx.barbell_graph(m1, m2)

    return G


def generate_wheel_graph(n):
    """
    Generate a wheel graph.

    Parameters:
    - n: Number of nodes in the graph. This includes the central node and the nodes of the cycle.

    Returns:
    - A networkx Graph object representing the wheel graph.
    """

    # Use built-in function to create a wheel graph
    G = nx.wheel_graph(n)

    return G

def generate_random_tree(n):
    G = nx.Graph()
    nodes = list(range(n))
    random.shuffle(nodes)

    # Start by adding one node
    G.add_node(nodes.pop())

    # While there are still nodes left, connect each to a random node in the growing tree
    while nodes:
        target = random.choice(list(G.nodes()))
        G.add_edge(nodes.pop(), target)

    return G


def sample_connected_subgraph(G, m):
    if m > len(G) or m <= 0:
        raise ValueError("Invalid sample size.")

    nodes = list(G.nodes())

    max_attempts = 1000
    for _ in range(max_attempts):
        sampled_nodes = np.random.choice(nodes, size=m, replace=False)
        subgraph = G.subgraph(sampled_nodes)
        if nx.is_connected(subgraph):
            return sampled_nodes

    raise ValueError("Couldn't find a connected subgraph after many attempts.")
