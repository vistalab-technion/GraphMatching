import networkx as nx
import numpy as np
from abc import ABC, abstractmethod


def generate_random_degree_sequence(n):
    # Generate random number of nodes for each degree value
    degree_counts = np.random.randint(0, n + 1, size=n + 1)

    # Create the degree sequence
    degree_sequence = []
    for degree, count in enumerate(degree_counts):
        degree_sequence.extend([degree] * count)

    return degree_sequence


# Set the maximum degree value
n = 5


def generate_graph_with_degree_sequence(degree_sequence):
    # Create an empty graph
    graph = nx.Graph()

    # Create a list of nodes based on the degree sequence
    nodes = []
    for i, degree in enumerate(degree_sequence):
        nodes.extend([i] * degree)

    # Add nodes to the graph
    graph.add_nodes_from(nodes)

    # Shuffle the node list randomly
    np.random.shuffle(nodes)

    # Connect the nodes randomly based on their degrees
    for i in range(len(nodes)):
        node = nodes[i]
        neighbors = list(graph.nodes())[:i]
        np.random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor != node and graph.degree(neighbor) < degree_sequence[neighbor]:
                graph.add_edge(node, neighbor)
                break

    return graph




class BaseGraphGenerator(ABC):
    """
    A base class for graph generator. Objects from this class contain some method to
     generate graphs
    """

    def __init__(self):
        super().__init__()


    def generate(self, num_graphs : int):
        """"
        a method to generate

        :param num_graphs: number of graphs to generate
        """
        pass