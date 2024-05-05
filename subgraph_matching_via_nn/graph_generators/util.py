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


def generate_random_join_degree_graph(n):
    # Randomly generate a degree sequence
    degree_sequence = np.random.randint(1, n, size=n)

    # Generate a random joint degree matrix
    while True:
        joint_degree_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i, n):
                max_degree = min(degree_sequence[i], degree_sequence[j])
                joint_degree_matrix[i, j] = np.random.randint(0, max_degree + 1)
                joint_degree_matrix[j, i] = joint_degree_matrix[
                    i, j]  # Symmetric matrix

        # Check if the joint degree matrix is valid
        row_sums = np.sum(joint_degree_matrix, axis=1)
        col_sums = np.sum(joint_degree_matrix, axis=0)
        if np.array_equal(row_sums, degree_sequence) and np.array_equal(col_sums,
                                                                        degree_sequence):
            break

    # Generate a graph using the joint degree matrix
    G = nx.generators.joint_degree_graph(joint_degree_matrix)
    return G


def generate_graph_with_unique_degrees(n):
    k = 7
    degree_sequence = [i + 1 for i in range(n)]
    degree_sequence[0] = k
    sum_of_degrees = sum(degree_sequence)
    if sum_of_degrees % 2 != 0:
        degree_sequence[k] = degree_sequence[k] + 1
    # degree_sequence = [k if i == 0 else i for i in range(n)]
    print(degree_sequence)
    G = nx.configuration_model(degree_sequence)
    # Remove self-loops
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

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
