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

    def plot_subgraph_indicators(self, G, to_line: bool, indicator_name_to_object_map: dict):
        fig, axes = plt.subplots(1, len(indicator_name_to_object_map), figsize=[18, 4])

        axes_counter = 0
        for indicator_name, indicator_obj in indicator_name_to_object_map.items():
            indicator_obj = self.get_w_indicator_from_w_indicator_dict(indicator_obj, to_line)
            plot_graph_with_colors(G=G, distribution=indicator_obj,
                                   title=indicator_name,
                                   ax=axes[axes_counter], seed=self.seed)
            axes_counter += 1

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

