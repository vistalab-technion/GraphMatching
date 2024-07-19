import numpy as np
from matplotlib import pyplot as plt

from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.utils.utils import plot_graph_with_colors, plot_degree_distribution


class PlotServices:

    def __init__(self, seed: int):
        self.seed = seed

    def plot_graph_alongside_subgraph(self, sub_graph:SubGraph, is_show_plot=True, n_subplots=2):
        assert n_subplots >= 2
        G = sub_graph.G
        G_sub = sub_graph.G_sub

        fig, axes = plt.subplots(1, n_subplots, figsize=(16, 4))

        plot_graph_with_colors(G=G, distribution=sub_graph.original_edge_distribution_indicator, ax=axes[0],
                               colorbar=False, title='Full graph', seed=self.seed, draw_labels=True)
        plot_graph_with_colors(G=G_sub, ax=axes[1], colorbar=False,
                               title='Sub-graph', seed=self.seed, draw_labels=True)

        if is_show_plot:
            plt.show()

        return axes

    def plot_graph_alongside_subgraph_with_degree_distribution(self, sub_graph: SubGraph, n_moments: int):
        axes = self.plot_graph_alongside_subgraph(sub_graph, is_show_plot=False, n_subplots=3)

        # Plot degree distribution and compute first 4 moments
        moments = plot_degree_distribution(sub_graph.G_sub, n_moments=n_moments, ax=axes[2])

        plt.show()
        print(f"First {n_moments} moments: {[f'{value:.4f}' for value in moments]}")

    def plot_subgraph_indicators(self, G, to_line: bool, indicator_name_to_object_map: dict, is_show=True):
        fig, axes = plt.subplots(1, len(indicator_name_to_object_map), figsize=[18, 4])

        axes_counter = 0
        for indicator_name, indicator_obj in indicator_name_to_object_map.items():
            indicator_obj = self.get_w_indicator_from_w_indicator_dict(indicator_obj, to_line)
            plot_graph_with_colors(G=G, distribution=indicator_obj,
                                   title=indicator_name,
                                   ax=axes[axes_counter], seed=self.seed)
            axes_counter += 1

        if is_show:
            plt.show()

    def get_w_indicator_from_w_indicator_dict(self, w_dict, to_line):
        if to_line:
            return w_dict
        else:
            sorted_dict_items = sorted(w_dict.items(), reverse=False, key=lambda item: item[0])
            return np.array([sorted_dict_item[1] for sorted_dict_item in sorted_dict_items])

    def plot_subgraph_gt_vs_initial_indicators(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, w_init, gt_indicator):
        G = sub_graph.G
        w_init_dict = dict(zip(processed_sub_graph.G.nodes(), w_init))

        fig, axes = plt.subplots(1, 2, figsize=[18, 4])

        to_line = processed_sub_graph.is_line_graph
        w_init_indicator = self.get_w_indicator_from_w_indicator_dict(w_init_dict, to_line)

        plot_graph_with_colors(G=G, title='gt', distribution=gt_indicator, ax=axes[0], seed=self.seed)
        plot_graph_with_colors(G=G, title='w_init', distribution=w_init_indicator, ax=axes[1], seed=self.seed)

        plt.show()


def plot_mask_gt_vs_mask_values(graph, gt_edges, marked_edges, disabled_edges, edge_mask_dicts, step_number):
    # Get the nodes and sort them to ensure consistent matrix indices
    nodes = sorted(graph.nodes())
    n = len(nodes)

    # Create a figure and axis for the plot
    n_subplots = len(edge_mask_dicts)
    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(16 * n_subplots, 16))

    if n_subplots > 1:
        axes = axes.flat
    else:
        axes = [axes]

    for i in range(n_subplots):
        ax = axes[i]

        ax.set_axis_off()

        # Initialize the matrix with None
        matrix = [[None for _ in range(n)] for _ in range(n)]
        # Fill in the matrix with scores from the graph
        edge_mask_dict = edge_mask_dicts[i]
        for edge, mask_score in edge_mask_dict.items():
            i, j = edge
            matrix[i][j] = "{:.2e}".format(mask_score.item())

        # Create a table plot
        table = ax.table(cellText=matrix, loc='center', cellLoc='center', colLabels=nodes, rowLabels=nodes,
                         colColours=["palegreen"] * n, rowColours=["palegreen"] * n)
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.2)

        # Mark the Ground Truth edges
        for u, v in gt_edges:
            i, j = nodes.index(u), nodes.index(v)
            cell = table[(i + 1, j)]
            cell.set_facecolor("#56b5fd")  # Set GT cell color
        for u, v in marked_edges:
            i, j = nodes.index(u), nodes.index(v)
            cell = table[(i + 1, j)]
            cell.set_text_props(weight='bold')
        for u, v in disabled_edges:
            i, j = nodes.index(u), nodes.index(v)
            cell = table[(i + 1, j)]
            # cell.set_text_props(weight='bold')
            text = cell.get_text().get_text()
            # Adding strikethrough characters
            strikethrough_text = ''.join([char + '\u0336' for char in text])
            cell.get_text().set_text(strikethrough_text)

    graph_file_name = "mask gt vs marked in binarization %d.png" % step_number
    plt.savefig(graph_file_name, format="PNG")