import itertools
import math
import sys
from os import cpu_count
from typing import Dict
import multiprocessing as mp
import networkx as nx
import numpy as np

from common.graph_utils import SubGraphGenerator
from common.logger import TimeLogging
from metrics.accuracy_metrics import evaluate_binary_classifier
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.utils.graph_utils import get_node_indicator_given_subgraph_nodes


def evaluate_mask_performance(w_bin, gt_node_distribution_processed):
    return evaluate_binary_classifier(y_pred =
                               (w_bin/max(w_bin)).astype(bool),
                               y_true= (gt_node_distribution_processed/max(gt_node_distribution_processed)).squeeze().numpy().astype(bool))


def compute_relative_hausdorff_distance_to_target_subgraph(full_graph, target_subgraph, actual_subgraph):
    shortest_path_distances = {}
    for source_node in actual_subgraph:
        shortest_path_distances[source_node] = nx.single_source_shortest_path_length(full_graph, source_node)

    max_distance = max(shortest_path_distances[node][target_node] for node in actual_subgraph for target_node in target_subgraph)

    diameter = nx.diameter(full_graph)
    alpha = max_distance / diameter
    return alpha, shortest_path_distances


def calculate_cdf(histogram_map):
    # Step 1: Sort the histogram keys
    sorted_keys = sorted(histogram_map.keys())

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

    return {key:cdf_val for key, cdf_val in zip(sorted_keys, cdf)}


def get_cdf_score(ordered_cdf_map: Dict, x: float):
    cdf_score = 0
    for key, cdf_value in ordered_cdf_map.items():
        if x < key:
            break
        cdf_score = cdf_value

    return cdf_score


def connected_induced_subgraph_cdf_score(sub_graph: SubGraph, correctly_captured_nodes_number):
    # CDF of correctly classified c out of k connected induced subgraphs,
    # to show usefulness (for testing vs random choise connected k subgraph)

    graph = sub_graph.G
    gt_subgraph = sub_graph.G_sub
    gt_binary_mask = sub_graph.node_indicator

    subgraph_size = len(gt_subgraph)
    _, k_subgraphs_original_nodes = SubGraphGenerator.generate_k_subgraphs(graph, k=subgraph_size, is_parallel=True)

    # go through all k subgraphs, and for each check mark how many captured gt nodes
    num_correct_nodes_to_num_k_subgraphs_map = \
        measure_k_subgraphs_nodes_against_gt_mask(graph, k_subgraphs_original_nodes, gt_binary_mask)

    # compute CDF score
    ordered_cdf_map = calculate_cdf(num_correct_nodes_to_num_k_subgraphs_map)
    print(f"CDF: {ordered_cdf_map}")
    cdf_score = get_cdf_score(ordered_cdf_map, correctly_captured_nodes_number)

    return cdf_score


def measure_k_subgraph_num_correct_nodes(graph, gt_binary_mask, k_subgraph_original_nodes):
    k_subgraph_binary_mask = get_node_indicator_given_subgraph_nodes(graph, k_subgraph_original_nodes)
    num_correct_nodes = (k_subgraph_binary_mask * gt_binary_mask).sum()
    return num_correct_nodes


def measure_k_subgraph_num_correct_nodes_for_chunk(graph, gt_binary_mask, chunk_index, k_subgraph_original_nodes_chunk):
    curr_time = TimeLogging.log_time(None, "enter measure_k_subgraph_num_correct_nodes_for_chunk")

    chunk_num_correct_nodes = [measure_k_subgraph_num_correct_nodes(graph, gt_binary_mask, k_subgraph_original_nodes)
            for k_subgraph_original_nodes in k_subgraph_original_nodes_chunk]

    curr_time = TimeLogging.log_time(curr_time, f"Chunk #{chunk_index} finished, "
                                                f"chunk size={len(k_subgraph_original_nodes_chunk)}")
    sys.stdout.flush()
    return chunk_num_correct_nodes


def measure_k_subgraphs_nodes_against_gt_mask(full_graph, k_subgraphs_original_nodes, gt_binary_mask, is_parallel=True):
    n = len(k_subgraphs_original_nodes)
    curr_time = TimeLogging.log_time(None, f"enter measure_k_subgraphs_nodes_against_gt_mask (total of {n} graphs)")

    cpu_num = int(cpu_count())
    if is_parallel:
        chunks_amount = cpu_num
    else:
        chunks_amount = 1

    chunk_size = int(math.ceil(n / chunks_amount))
    chunks = [k_subgraphs_original_nodes[i * chunk_size: min(i * chunk_size + chunk_size, n)] for i in
              range(chunks_amount)]

    if is_parallel:
        with mp.Pool(processes=cpu_num) as pool:
            # execute tasks in order
            num_correct_nodes_lists_list = pool.starmap(measure_k_subgraph_num_correct_nodes_for_chunk,
                                               zip(itertools.repeat(full_graph),
                                                   itertools.repeat(gt_binary_mask),
                                                   range(chunks_amount),
                                                   chunks))
        num_correct_nodes_list = [e for lst in num_correct_nodes_lists_list for e in lst]
    else:
        num_correct_nodes_list = measure_k_subgraph_num_correct_nodes_for_chunk(full_graph, gt_binary_mask, 1, chunks[0])

    curr_time = TimeLogging.log_time(curr_time, "finished correct nodes counting")

    num_correct_nodes_to_num_k_subgraphs_map = {}
    for num_correct_nodes in num_correct_nodes_list:
        if num_correct_nodes in num_correct_nodes_to_num_k_subgraphs_map:
            num_correct_nodes_to_num_k_subgraphs_map[num_correct_nodes] += 1
        else:
            num_correct_nodes_to_num_k_subgraphs_map[num_correct_nodes] = 1

    _ = TimeLogging.log_time(curr_time, "finished measure_k_subgraphs_nodes_against_gt_mask")

    return num_correct_nodes_to_num_k_subgraphs_map


def is_connected_subgraph(full_graph, binary_w):
    sampled_nodes = np.nonzero(binary_w.reshape(-1)).flatten().tolist()
    # sampled_nodes = np.nonzero(binary_w)[0]
    subgraph = full_graph.subgraph(sampled_nodes)
    print(subgraph.nodes)
    print(subgraph.edges)
    return subgraph, nx.is_connected(subgraph)
