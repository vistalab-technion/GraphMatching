import numpy as np
import torch

from subgraph_matching_via_nn.composite_nn.localization_state_replayer import ReplayableLocalizationState
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.localization_inference_scorer import FBetaLocalizationInferenceScorer, \
    ConnectivityLocalizationInferenceScorer, OverlapNodesCDFLocalizationInferenceScorer, \
    RelativeHausdorffDistanceCDFLocalizationInferenceScorer
from subgraph_matching_via_nn.graph_classifier_networks.greedy_search_node_classifier_schemes import \
    GreedySearchNodeClassifierSchemes
from subgraph_matching_via_nn.mask_binarization.indicator_dsitribution_binarizer import IndicatorBinarizationType, \
    IndicatorBinarizationBootType
from subgraph_matching_via_nn.training.PairSampleInfo import PairSampleBase
from subgraph_matching_via_nn.utils.graph_utils import get_normalized_node_indicator
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class LocalizationInference:

    def __init__(self, sub_graph, processed_sub_graph, original_reference_subgraph, reference_subgraph,
                 composite_solver, params, plot_services):

        self.sub_graph = sub_graph
        self.processed_sub_graph = processed_sub_graph
        self.original_reference_subgraph = original_reference_subgraph
        self.reference_subgraph = reference_subgraph

        self.composite_solver = composite_solver
        self.params = params
        self.to_line = self.params['to_line']
        self.plot_services = plot_services

        self.f_beta_scorer = FBetaLocalizationInferenceScorer()
        self.connectivity_scorer = ConnectivityLocalizationInferenceScorer()
        self.overlap_nodes_cdf_scorer = OverlapNodesCDFLocalizationInferenceScorer()
        self.rel_hausdorff_distance_cdf_scorer = RelativeHausdorffDistanceCDFLocalizationInferenceScorer()

    def binarize_and_check_for_negative_example(self, w_all, w_star, binarization_type, series_binarization_func):
        processed_G = self.processed_sub_graph.G

        # binarization
        greedy_search_node_classifier_schemes = GreedySearchNodeClassifierSchemes(self.composite_solver, self.sub_graph,
                                                                                  self.processed_sub_graph,
                                                                                  self.reference_subgraph,
                                                                                  self.original_reference_subgraph)

        # this code should align with the binarization API we have, refactor in the future
        if binarization_type == IndicatorBinarizationType.greedy_stepwise:
            best_quantile = self.params['greedy_search_node_classifier_best_quantile']
            worst_quantile = self.params['greedy_search_node_classifier_worst_quantile']
            w_rounded = greedy_search_node_classifier_schemes.greedy_stepwise_binarization(w_star, use_magnitude=True,
                                                                                           best_quantile=best_quantile,
                                                                                           worst_quantile=worst_quantile)  # TODO:extract param use_magnitude
        elif binarization_type == IndicatorBinarizationType.simulated_greedy_stepwise:
            w_rounded = greedy_search_node_classifier_schemes.simulated_stepwise_binarization(w_star,
                                                                                              use_magnitude=True)
        else:
            w_bin_dict = series_binarization_func(self.sub_graph.G, processed_G, w_all, w_star, self.params,
                                             IndicatorBinarizationBootType.OptimalElement, binarization_type)
            w_rounded = np.array(list(w_bin_dict.values()))
        print(f"rounded W = {w_rounded}")

        # check if GT solution was found
        reference_sub_graph = SubGraph(self.processed_sub_graph.G, self.reference_subgraph, is_line_graph=self.to_line,
                                       original_graph=self.sub_graph.G, original_subgraph=self.sub_graph.G_sub)
        reference_gt_indicator = reference_sub_graph.w_gt
        is_negative_example = (w_rounded != reference_gt_indicator.detach().cpu().numpy()).any()

        return is_negative_example, w_rounded

    def single_round_subgraph_inference(self, binarization_type, w_all, series_binarization_func):
        w_star = self.composite_solver.init_mask_and_solve_one_round(self.sub_graph, self.original_reference_subgraph)
        # check for convergence
        if w_star is None:
            # no localization
            return False, None

        is_negative_example, w_rounded = self.binarize_and_check_for_negative_example(w_all, w_star, binarization_type,
                                                                                      series_binarization_func)

        return True, (is_negative_example, w_rounded, w_star)

    def collect_negative_pair_sample(self, w_star, device):
        w_example = torch.tensor(w_star)  # w_rounded
        w_example = w_example.to(device=device)

        # collect negative + positive example pairs
        converged_subgraph = AnnotatedGraph(self.processed_sub_graph.G, label=None, node_attributes=w_example.reshape(-1),
                                            device=device)
        pair_sample_info = PairSampleBase(
            masked_graph=converged_subgraph,
            subgraph=AnnotatedGraph(self.reference_subgraph, label=None, device=device),
            is_negative_sample=True,
            localization_state_object=ReplayableLocalizationState(self.processed_sub_graph, w_example)
        )

        return pair_sample_info

    def experiment_scores_evaluation(self, binarized_solution):
        normalized_binarized_solution = get_normalized_node_indicator(binarized_solution, dtype=TORCH_DTYPE)\
            .reshape(-1, 1)

        # F-beta score
        f_beta_score = self.f_beta_scorer.score(self.sub_graph, self.processed_sub_graph, normalized_binarized_solution)
        print(f"F-beta score = {f_beta_score}")

        # connectivity
        is_connected = self.connectivity_scorer.score(self.sub_graph, self.processed_sub_graph, normalized_binarized_solution)
        print(f"is_connected_subgraph = {is_connected}")

        # CDF scores
        correctly_captured_nodes_number, overlap_cdf_score = self.overlap_nodes_cdf_scorer.score(
            self.sub_graph, self.processed_sub_graph, normalized_binarized_solution)
        print(f"overlap nodes number= {correctly_captured_nodes_number}")
        print(f"overlap nodes distance CDF score = {overlap_cdf_score}")

        # Relative Hausdorff
        hausdorff_relative_distance, hausdorff_relative_distance_cdf_score = self.rel_hausdorff_distance_cdf_scorer.score(
            self.sub_graph, self.processed_sub_graph, normalized_binarized_solution)
        print(f"Relative Hausdorff distance = {hausdorff_relative_distance}")
        print(f"Relative Hausdorff distance CDF score = {hausdorff_relative_distance_cdf_score}")

        # show result
        processed_G = self.processed_sub_graph.G
        indicator_name_to_object_map = {
            'w_star': dict(zip(processed_G.nodes(), np.array(normalized_binarized_solution))),
            'gt sub': dict(zip(processed_G.nodes(), np.array(self.processed_sub_graph.node_indicator)))
        }
        self.plot_services.plot_subgraph_indicators(self.sub_graph.G, self.to_line, indicator_name_to_object_map)

        variable_dict = {'f_beta_score': f_beta_score, 'is_connected': is_connected,
                         'correctly_captured_nodes_number': correctly_captured_nodes_number,
                         'overlap_cdf_score': overlap_cdf_score, 'hausdorff_relative_distance': hausdorff_relative_distance,
                         'hausdorff_relative_distance_cdf_score': hausdorff_relative_distance_cdf_score,
                         'binarized_solution': binarized_solution}

        return variable_dict