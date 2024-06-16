import networkx as nx

from metrics.accuracy_metrics import evaluate_binary_classifier


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