import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import norm
import seaborn as sns

TORCH_DTYPE = torch.float64
NP_DTYPE = np.float64

def plot_indicator(w_list, labels):
    # Sort the flattened array independently
    idx = np.argsort(w_list[0], axis=0)
    sorted_w_list = [w[idx].squeeze(-1) for w in w_list]
    # sorted_w_list = [np.sort(w, axis=0) for w in w_list]

    # Plotting the sorted tensors
    markers = ['o', 's', '^']  # List of markers for different tensors
    colors = ['red', 'blue', 'green']  # List of colors for different tensors

    # Plot each tensor with a different marker, color, and label
    for i, w in enumerate(sorted_w_list):
        plt.plot(range(len(w)), w, marker=markers[i], color=colors[i], label=labels[i])

    plt.xticks(range(len(w_list[0])), range(len(w_list[0])))
    plt.title('Permuted solutions according to sorted gt')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def plot_degree_distribution(graph, n_moments=4, ax=None):
    # Get the degree sequence of the graph
    n = graph.number_of_nodes()
    degree_sequence = [degree / n for _, degree in graph.degree()]

    # Plot the degree distribution
    sns.histplot(degree_sequence, bins='auto', stat='density',
                 label='Normalized Degree Histogram', ax=ax)

    # Fit a distribution to the data
    mu, std = norm.fit(degree_sequence)
    x = np.linspace(min(degree_sequence), max(degree_sequence), 100)
    y = norm.pdf(x, mu, std)
    sns.lineplot(x=x, y=y, color='red', label='Fitted Normal Distribution', ax=ax)

    # Plot the KDE
    sns.kdeplot(degree_sequence, label='KDE', bw_adjust=0.5, ax=ax)

    ax.set_xlabel('Degree')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Compute the first k moments
    moments = []
    for i in range(1, n_moments + 1):
        moments.append(np.mean(np.power(degree_sequence, i)))

    return moments


def uniform_dist(n):
    x = torch.ones(n, 1, dtype=TORCH_DTYPE)
    return x / x.sum()


def get_node_indicator(G: nx.graph, G_sub: nx.graph):
    """
    Create node indicator for G_sub in G (assuming G_sub was extracted from G)

    :param G: A networkx graph
    :param G_sub: A networkx sub-graph of G
    :return: w_indicator - a vector with w[i] ==1 if node i of G is a node in G_sub
    , otherwise w_indicator[i]==0.
    """
    # Set the indices corresponding to the subgraph nodes to 1
    subgraph_node_indices = [list(G.nodes()).index(node) for node in G_sub.nodes()]
    # subgraph_node_indices = list(G_sub.nodes())
    w_indicator = np.zeros(len(G.nodes()))
    w_indicator[subgraph_node_indices] = 1.0
    return w_indicator


def get_edge_indicator(G: nx.graph, G_sub: nx.graph):
    """
    Create edge indicator for G_sub in G (assuming G_sub was extracted from G)

    :param G: A networkx graph
    :param G_sub: A networkx sub-graph of G
    :return: edge_indicator - dict with values
    edge_indicator[(i,j)] == edge_indicator[(j,i)] ==1 if (i,j) is an edge of G_sub,
    and 0 otherwise.
    """
    edge_indicator = \
        {(min(u, v), max(u, v)): 1 if (min(u, v), max(u, v))
                                      in G_sub.edges() else 0 for u, v in G.edges()}

    # Create symmetric adjacency matrix
    num_nodes = len(G.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for (i, j), val in edge_indicator.items():
        adj_matrix[i][j] = val
        adj_matrix[j][i] = val  # Ensure it's symmetric

    return edge_indicator, adj_matrix


def node_indicator_from_edge_indicator(G: nx.graph, edge_indicator):
    # Create node incident vector
    w = [0] * len(G.nodes())
    for node in G.nodes():
        # Get incident edges for node
        incident_edges = [(min(node, neighbor), max(node, neighbor)) for neighbor in
                          G.neighbors(node)]

        # Calculate average of edge_indicator values for the incident edges
        avg_value = max([edge_indicator[edge] for edge in incident_edges])

        w[list(G.nodes).index(node)] = float(avg_value)
    return w


def plot_graph_with_colors(G: nx.graph, G_sub: nx.graph, node_indicator=None,
                           edge_indicator: dict = None, title: str = '', ax=None,
                           colorbar: bool = True, seed: int = 42,
                           draw_labels: bool = False):
    # Define the colors for the colormap
    colors = ['red', 'yellow', 'green']  # Red to Green
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)

    if edge_indicator is not None:
        norm = mcolors.Normalize(vmin=min(edge_indicator.values()),
                                 vmax=max(edge_indicator.values()))
        edge_probabilities = [norm(edge_indicator[(u, v)]) for u, v in
                              G.edges()]
        node_probabilities = norm(node_indicator_from_edge_indicator(G=G,
                                                                     edge_indicator=edge_indicator))
    else:
        if node_indicator is not None:
            # Normalize w to match the colormap range
            norm = mcolors.Normalize(vmin=min(node_indicator), vmax=max(node_indicator))

            # Generate a list of colors for nodes based on w values
            node_probabilities = norm(node_indicator)
            edge_probabilities = [norm((node_indicator[list(G.nodes).index(u)] +
                                        node_indicator[list(G.nodes).index(v)]) / 2.0)
                                  for u, v in
                                  G.edges()]

        else:

            # Generate a list of colors for nodes, red for subgraph nodes and green
            # for the rest
            node_probabilities = [1.0 if node in G_sub.nodes() else 0.0 for node in
                                  G.nodes()]
            # Generate a list of colors for edges, red for subgraph edges and green
            # for the rest
            edge_probabilities = [1.0 if edge in G_sub.edges() else 0.0 for edge in
                                  G.edges()]

    node_colors = cmap(node_probabilities)
    edge_colors = cmap(edge_probabilities)

    # Set a fixed seed for the layout algorithm
    pos = nx.spring_layout(G, seed=seed)  # Layout algorithm for graph visualization

    # Draw the graph with node and edge colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    # Add node numbers as labels
    if draw_labels:
        node_labels = {node: node for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8,
                                font_color='blue', ax=ax)

    ax.set_axis_off()  # Turn off the axis

    # Add colorbar
    if edge_indicator is not None or node_indicator is not None and colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
    ax.set_title(title)
