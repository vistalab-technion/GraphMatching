from typing import Union

import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import norm
import seaborn as sns

from subgraph_matching_via_nn.utils.graph_utils import \
    node_indicator_from_edge_indicator

TORCH_DTYPE = torch.float64
NP_DTYPE = np.float64


def plot_indicator(w_list, labels, ax):
    # Sort the flattened array independently
    idx = np.argsort(w_list[0], axis=0)
    sorted_w_list = [w[idx].squeeze(-1) for w in w_list]
    # sorted_w_list = [np.sort(w, axis=0) for w in w_list]

    # Plotting the sorted tensors
    markers = ['o', 's', '^']  # List of markers for different tensors
    marker_sizes = [7, 5, 5]
    colors = ['green', 'cyan', 'red']  # List of colors for different tensors

    # Plot each tensor with a different marker, color, and label
    for i, w in enumerate(sorted_w_list):
        ax.plot(range(len(w)), w, marker=markers[i], color=colors[i], label=labels[i],
                markersize=marker_sizes[i], linestyle='None')

    plt.xticks(range(len(w_list[0])), range(len(w_list[0])))
    ax.set_title('Permuted solutions according to sorted gt')
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.legend()
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
    if not np.all(x == x[0]):
        y = norm.pdf(x, mu, std)
    else:
        y = np.ones_like(x)
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


def top_m(w, m):
    indices_of_top_m = np.argsort(w, axis=0)[-m:]  # top m
    w_th = np.zeros_like(w, dtype=NP_DTYPE)
    w_th[indices_of_top_m] = 1
    return w_th


def uniform_dist(n):
    x = torch.ones(n, 1, dtype=TORCH_DTYPE)
    return x / x.sum()


def plot_graph_with_colors(G: nx.graph,
                           distribution: Union[dict, np.ndarray] = None,
                           title: str = '',
                           ax=None,
                           colorbar: bool = True, seed: int = 42,
                           draw_labels: bool = False):
    """

    :param G: graph
    :param G_sub: sub_graph of G
    :param distribution: either a node distribution (numpy array) or and edge
    distribution
    (dict of {edge tuple : distribution value})
    :param title: title for the plot
    :param ax: axis for the plot
    :param colorbar: flag for showing colorbar
    :param seed: random seed to make sure the embedding is always the same
    :param draw_labels: flag for whether to draw node labels or not
    :return:
    """
    # TODO: instead of passing both edge_indicator and node_indicator, infer it from
    #  the data_type
    # Define the colors for the colormap
    colors = ['red', 'blue', 'green']  # Red to Green
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)

    # infer if this a node indicator or an edge indicator
    if type(distribution) is np.ndarray:
        # Normalize w to match the colormap range
        if min(distribution) == max(distribution):
            norm = mcolors.Normalize(vmin=0, vmax=max(distribution))
        else:
            norm = mcolors.Normalize(vmin=min(distribution), vmax=max(distribution))

        # Generate a list of colors for nodes based on w values
        node_probabilities = norm(distribution)
        edge_probabilities = [norm((distribution[list(G.nodes).index(u)] +
                                    distribution[list(G.nodes).index(v)]) / 2.0)
                              for u, v in
                              G.edges()]
    elif type(distribution) is dict:
        edge_indicator = distribution
        if min(edge_indicator.values()) == max(edge_indicator.values()):
            norm = mcolors.Normalize(vmin=0,
                                     vmax=max(edge_indicator.values()))
        else:
            norm = mcolors.Normalize(vmin=min(edge_indicator.values()),
                                     vmax=max(edge_indicator.values()))
        edge_probabilities = [norm(edge_indicator[(u, v)]) for u, v in
                              G.edges()]
        node_probabilities = norm(node_indicator_from_edge_indicator(G=G,
                                                                     edge_indicator=edge_indicator))
    elif distribution is None:
        # Generate a list of colors for nodes, red for subgraph nodes and green
        # for the rest
        node_probabilities = [1.0 for node in G.nodes()]
        edge_probabilities = [1.0 for edge in G.edges()]

        # node_probabilities = [1.0 if node in G.nodes() else 0.0 for node in
        #                       G.nodes()]
        # Generate a list of colors for edges, red for subgraph edges and green
        # for the rest
        # edge_probabilities = [1.0 if edge in G.edges() else 0.0 for edge in
        #                       G.edges()]
    else:
        raise ("Only NumPy array or a dictionary are supported.")

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
    if distribution is not None and colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
    ax.set_title(title)
