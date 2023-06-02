import torch

DTYPE = torch.float64
import networkx as nx
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
from torch import Tensor, stack
from livelossplot import PlotLosses
import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns


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


def plot_graph_with_colors(G, G_sub, w=None, title='', ax=None,
                           colorbar=True, seed=42, draw_labels = False):
    if w is None:
        # Generate a list of colors for nodes, red for subgraph nodes and green for the rest
        node_colors = ['green' if node in G_sub.nodes() else 'red' for node in
                       G.nodes()]

        # Generate a list of colors for edges, red for subgraph edges and green for the rest
        edge_colors = ['green' if edge in G_sub.edges() else 'red' for edge in
                       G.edges()]
    else:
        # Define the colors for the colormap
        colors = ['red', 'yellow', 'green']  # Red to Green
        cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)

        # Normalize w to match the colormap range
        norm = mcolors.Normalize(vmin=min(w), vmax=max(w))

        # Generate a list of colors for nodes based on w values
        node_colors = cmap(norm(w))
        # Generate a list of colors for edges based on the nodes they connect
        edge_colors = [cmap(norm((w[list(G.nodes).index(u)] +
                                  w[list(G.nodes).index(v)]) / 2.0)) for u, v in
                       G.edges()]

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
    if w is not None and colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
    ax.set_title(title)


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
    x = torch.ones(n, 1, dtype=DTYPE)
    return x / x.sum()


def align_subgraph_edges_with_full_graph_edges(sub_graph: nx.Graph, full_graph: nx.Graph):
    edges_to_add = []
    for node_i, node_j in full_graph.edges:
        if node_i in sub_graph.nodes:
            if not sub_graph.has_edge(node_i, node_j):
                edges_to_add.append((node_i, node_j))
    return edges_to_add