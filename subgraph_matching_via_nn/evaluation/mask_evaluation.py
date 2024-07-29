import math
import sys
from abc import abstractmethod, ABC
from os import cpu_count
from typing import Dict
import multiprocessing as mp
import networkx as nx
import numpy as np
from overrides import overrides
from common.graph_utils import SubGraphGenerator
from common.logger import TimeLogging
from metrics.accuracy_metrics import evaluate_binary_classifier
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_metrics_constants import MaskMetricsConstants
from subgraph_matching_via_nn.utils.graph_utils import get_node_indicator_given_subgraph_nodes


def induced_subgraph(full_graph, binary_w):
    sampled_nodes = np.nonzero(binary_w.reshape(-1)).flatten().tolist()
    # sampled_nodes = np.nonzero(binary_w)[0]

    full_graph_nodes = list(full_graph.nodes)
    sampled_nodes = [full_graph_nodes[i] for i in sampled_nodes]

    subgraph = full_graph.subgraph(sampled_nodes)
    # print(subgraph.nodes)
    # print(subgraph.edges)
    return subgraph


def evaluate_mask_performance(w_bin, gt_node_distribution_processed):
    return evaluate_binary_classifier(y_pred =
                               (w_bin/max(w_bin)).astype(bool),
                               y_true= (gt_node_distribution_processed/max(gt_node_distribution_processed)).squeeze().numpy().astype(bool))


def compute_relative_hausdorff_similarity_to_target_subgraph(full_graph, target_subgraph, actual_subgraph,
                                                             k_subgraph_nodes=None):
    if k_subgraph_nodes is None:
        actual_subgraph_nodes = actual_subgraph.nodes
    else:
        actual_subgraph_nodes = k_subgraph_nodes

    shortest_path_distances = {}
    for source_node in actual_subgraph_nodes:
        shortest_path_distances[source_node] = nx.single_source_shortest_path_length(full_graph, source_node)

    max_distance = max(min([shortest_path_distances[node][target_node] for target_node in target_subgraph])
                       for node in actual_subgraph_nodes)

    diameter = nx.diameter(full_graph)
    rel_distance = max_distance / diameter
    return 1-rel_distance, shortest_path_distances


def measure_k_subgraph_num_correct_nodes(graph, gt_binary_mask, k_subgraph_original_nodes):
    k_subgraph_binary_mask = get_node_indicator_given_subgraph_nodes(graph, k_subgraph_original_nodes)
    num_correct_nodes = (k_subgraph_binary_mask * gt_binary_mask).sum()
    return num_correct_nodes


class CDFScoreService:

    @staticmethod
    def calculate_cdf(histogram_map, is_complement=False):
        # Step 1: Sort the histogram keys
        sorted_keys = sorted(histogram_map.keys(), reverse=is_complement)

        # Step 2: Calculate the cumulative count
        cumulative_counts = []
        cumulative_sum = 0
        for key in sorted_keys:
            cumulative_sum += histogram_map[key]
            cumulative_counts.append(cumulative_sum)

        # Total number of samples
        total_count = cumulative_sum

        # Step 3: Normalize to get the CDF
        cdf = [count / total_count for count in cumulative_counts]

        return {key: cdf_val for key, cdf_val in zip(sorted_keys, cdf)}

    @staticmethod
    def get_cdf_score(ordered_cdf_map: Dict, x: float, is_complement: bool):
        cdf_score = 0

        basic_key_comparison_condition = lambda x, key: x < key
        key_comparison_condition = basic_key_comparison_condition
        if is_complement:
            key_comparison_condition = lambda x, key: not basic_key_comparison_condition(x, key)

        for key, cdf_value in ordered_cdf_map.items():
            if key_comparison_condition(x, key):
                break
            cdf_score = cdf_value

        return cdf_score


class ConnectedInducedSubgraphCDFScore(ABC):
    # CDF of metric among k connected induced subgraphs,
    # to show usefulness (for testing vs random choise connected k subgraph)

    def __init__(self, sub_graph: SubGraph):
        self.sub_graph = sub_graph
        self.is_complement_score = False

    def _log_cdf_histogram(self, ordered_cdf_map):
        pass

    def calculate(self, cdf_query_value):
        gt_subgraph = self.sub_graph.G_sub
        graph = self.sub_graph.G

        subgraph_size = len(gt_subgraph)
        # TODO: refactor to avoid multiple computations (save in state, allow setting from outside)
        k_subgraphs, k_subgraphs_original_nodes = SubGraphGenerator.generate_k_subgraphs(graph, k=subgraph_size,
                                                                                         is_parallel=True)

        metric_val_to_num_k_subgraphs_map = self.__calculate_subgraph_metric_histogram(k_subgraphs,
                                                                                       k_subgraphs_original_nodes)

        # compute CDF score
        ordered_cdf_map = CDFScoreService.calculate_cdf(metric_val_to_num_k_subgraphs_map,
                                                        is_complement=self.is_complement_score)
        self._log_cdf_histogram(ordered_cdf_map)

        cdf_score = CDFScoreService.get_cdf_score(ordered_cdf_map, cdf_query_value, is_complement=self.is_complement_score)

        return cdf_score

    @abstractmethod
    def _calculate_subgraph_metric(self, k_subgraphs_chunk, k_subgraph_original_nodes_chunk):
        pass

    # cannot be private, as it is used by Pool.starmap
    def _measure_k_subgraph_metric_for_chunk(self, chunk_index, k_subgraphs_chunk, k_subgraph_original_nodes_chunk):
        curr_time = TimeLogging.log_time(None, "enter _measure_k_subgraph_metric_for_chunk")

        metric_vals = self._calculate_subgraph_metric(k_subgraphs_chunk, k_subgraph_original_nodes_chunk)

        curr_time = TimeLogging.log_time(curr_time, f"Chunk #{chunk_index} finished, "
                                                    f"chunk size={len(k_subgraphs_chunk)}")
        sys.stdout.flush()
        return metric_vals

    def __calculate_subgraph_metric_histogram(self, k_subgraphs, k_subgraphs_original_nodes, is_parallel=True):
        n = len(k_subgraphs_original_nodes)
        curr_time = TimeLogging.log_time(None, f"enter __calculate_subgraph_metric_histogram (total of {n} graphs)")

        cpu_num = int(cpu_count())
        if is_parallel:
            chunks_amount = cpu_num
        else:
            chunks_amount = 1

        chunk_size = int(math.ceil(n / chunks_amount))
        chunk_size = min(chunk_size, 4 * 8_192)  # cap to avoid memory and timeout issues
        # according to chunk size, recalculate chnuks_amount
        chunks_amount = int(math.ceil(n / chunk_size))

        original_nodes_chunks = [k_subgraphs_original_nodes[i * chunk_size: min(i * chunk_size + chunk_size, n)] for i in
                  range(chunks_amount)]
        subgraph_chunks = [k_subgraphs[i * chunk_size: min(i * chunk_size + chunk_size, n)] for i in
                  range(chunks_amount)]

        if is_parallel:
            with mp.Pool(processes=cpu_num) as pool:
                # execute tasks in order
                metric_lists = pool.starmap(self._measure_k_subgraph_metric_for_chunk,
                                                   zip(range(chunks_amount),
                                                       subgraph_chunks,
                                                       original_nodes_chunks))
            metric_list = [e for lst in metric_lists for e in lst]
        else:
            metric_list = self._measure_k_subgraph_metric_for_chunk(1, subgraph_chunks[0], original_nodes_chunks[0])

        curr_time = TimeLogging.log_time(curr_time, "finished metric evaluation for all k-subgraphs")

        metric_val_to_num_k_subgraphs_map = {}
        for metric_val in metric_list:
            if metric_val in metric_val_to_num_k_subgraphs_map:
                metric_val_to_num_k_subgraphs_map[metric_val] += 1
            else:
                metric_val_to_num_k_subgraphs_map[metric_val] = 1

        _ = TimeLogging.log_time(curr_time, "finished __calculate_subgraph_metric_histogram")

        return metric_val_to_num_k_subgraphs_map


class ConnectedInducedSubgraphOverlapWithGTNodesCDFScore(ConnectedInducedSubgraphCDFScore):
    # CDF of correctly classified c out of GT k subgraph nodes

    @overrides
    def _log_cdf_histogram(self, ordered_cdf_map):
        print(f"overlap CDF: {ordered_cdf_map}")

    @overrides
    def _calculate_subgraph_metric(self, k_subgraphs_chunk, k_subgraph_original_nodes_chunk):
        graph = self.sub_graph.G
        gt_binary_mask = self.sub_graph.node_indicator

        chunk_num_correct_nodes = [
            measure_k_subgraph_num_correct_nodes(graph, gt_binary_mask, k_subgraph_original_nodes)
            for k_subgraph_original_nodes in k_subgraph_original_nodes_chunk]
        return chunk_num_correct_nodes


class ConnectedInducedSubgraphRelativeHausdorffSimilarityCDFScore(ConnectedInducedSubgraphCDFScore):
    # CDF of relative Hausdorff distance to GT k subgraph

    def __init__(self, sub_graph: SubGraph):
        super(ConnectedInducedSubgraphRelativeHausdorffSimilarityCDFScore, self).__init__(sub_graph)
        self.is_complement_score = False

    @overrides
    def _log_cdf_histogram(self, ordered_cdf_map):
        print(f"{MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME} map: {ordered_cdf_map}")

    @overrides
    def _calculate_subgraph_metric(self, k_subgraphs_chunk, k_subgraph_original_nodes_chunk):
        graph = self.sub_graph.G
        target_graph = self.sub_graph.G_sub

        chunk_metric_vals = [
            compute_relative_hausdorff_similarity_to_target_subgraph(graph, target_subgraph=target_graph,
                                                                     actual_subgraph=k_subgraph,
                                                                     k_subgraph_nodes=k_subgraph_original_nodes)[0]
            for k_subgraph, k_subgraph_original_nodes in zip(k_subgraphs_chunk, k_subgraph_original_nodes_chunk)]
        return chunk_metric_vals
