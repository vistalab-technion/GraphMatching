from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_evaluation import evaluate_mask_performance, induced_subgraph, \
    measure_k_subgraph_num_correct_nodes, ConnectedInducedSubgraphOverlapWithGTNodesCDFScore, \
    compute_relative_hausdorff_distance_to_target_subgraph, ConnectedInducedSubgraphRelativeHausdorffDistanceCDFScore


class LocalizationInferenceScorerBase(ABC):
    @abstractmethod
    def score(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, normalized_binarized_solution: np.array):
        pass


class FBetaLocalizationInferenceScorer(LocalizationInferenceScorerBase):

    def score(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, normalized_binarized_solution: np.array):
        return evaluate_mask_performance(normalized_binarized_solution.detach().cpu().numpy(), processed_sub_graph.w_gt)


class ConnectivityLocalizationInferenceScorer(LocalizationInferenceScorerBase):

    def score(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, normalized_binarized_solution: np.array):
        integer_binarized_solution = normalized_binarized_solution * (normalized_binarized_solution > 0).sum()

        actual_subgraph = induced_subgraph(processed_sub_graph.G, integer_binarized_solution)
        is_connected = nx.is_connected(actual_subgraph)
        return is_connected


class OverlapNodesCDFLocalizationInferenceScorer(LocalizationInferenceScorerBase):

    def score(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, normalized_binarized_solution: np.array):
        integer_binarized_solution = normalized_binarized_solution * (normalized_binarized_solution > 0).sum()
        actual_subgraph = induced_subgraph(processed_sub_graph.G, integer_binarized_solution)
        gt_binary_mask = processed_sub_graph.node_indicator

        correctly_captured_nodes_number = measure_k_subgraph_num_correct_nodes(processed_sub_graph.G,
                                                                               gt_binary_mask,
                                                                               list(actual_subgraph.nodes))

        overlap_cdf_scorer = ConnectedInducedSubgraphOverlapWithGTNodesCDFScore(sub_graph)
        overlap_cdf_score = overlap_cdf_scorer.calculate(correctly_captured_nodes_number)
        return correctly_captured_nodes_number, overlap_cdf_score


class RelativeHausdorffDistanceCDFLocalizationInferenceScorer(LocalizationInferenceScorerBase):

    def score(self, sub_graph: SubGraph, processed_sub_graph: SubGraph, normalized_binarized_solution: np.array):
        integer_binarized_solution = normalized_binarized_solution * (normalized_binarized_solution > 0).sum()
        actual_subgraph = induced_subgraph(processed_sub_graph.G, integer_binarized_solution)

        hausdorff_relative_distance, shortest_path_distances_from_target = compute_relative_hausdorff_distance_to_target_subgraph(
            processed_sub_graph.G, processed_sub_graph.G_sub, actual_subgraph)

        hausdorff_relative_distance_cdf_scorer = ConnectedInducedSubgraphRelativeHausdorffDistanceCDFScore(
            sub_graph)
        hausdorff_relative_distance_cdf_score = hausdorff_relative_distance_cdf_scorer.calculate(
            hausdorff_relative_distance)

        return hausdorff_relative_distance, hausdorff_relative_distance_cdf_score