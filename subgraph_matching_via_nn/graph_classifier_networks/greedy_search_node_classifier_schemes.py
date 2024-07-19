import math

import networkx as nx
import numpy as np
import torch
from subgraph_matching_via_nn.composite_nn.composite_solver import BaseCompositeSolver
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_debugger import MaskDebugger
from subgraph_matching_via_nn.graph_classifier_networks.greedy_search_node_classifier_schemes_services import \
    GreedySearchNodeClassifierSchemesServices
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE
from subgraph_matching_via_nn.utils.graph_utils import is_neighbor


class GreedySearchNodeClassifierSchemes:

    def __init__(self, composite_solver: BaseCompositeSolver, sub_graph: SubGraph,
                                     processed_sub_graph: SubGraph, reference_subgraph: nx.Graph,
                                     original_reference_subgraph: nx.Graph):
        self.composite_solver = composite_solver
        self.sub_graph = sub_graph
        self.processed_sub_graph = processed_sub_graph
        self.reference_subgraph = reference_subgraph
        self.original_reference_subgraph = original_reference_subgraph

        self.chosen_nodes_indices = []
        self.discarded_nodes_indices = []
        self.current_binarized_mask = None

    def __single_best_node_step(self, init_w_star, use_magnitude, step_number):
        chosen_subgraph_node_index = \
            GreedySearchNodeClassifierSchemesServices.get_most_prominent_mask_node(step_number, init_w_star,
                                                                                   self.chosen_nodes_indices,
                                                                                   self.discarded_nodes_indices,
                                                                                   self.composite_solver, self.sub_graph,
                                                                                   self.processed_sub_graph,
                                                                                   self.reference_subgraph,
                                                                                   use_magnitude=use_magnitude)

        self.chosen_nodes_indices.append(chosen_subgraph_node_index)
        self.current_binarized_mask[chosen_subgraph_node_index] = 1

        # Fix the chosen node mask entry (set the mask entry to 1 regardless to the following mask training iterations)
        self.composite_solver.composite_nn.ignore_w_indices(self.chosen_nodes_indices + self.discarded_nodes_indices)

        # solve the localization loop
        w_star = self.composite_solver.init_mask_and_solve_one_round(self.sub_graph, self.original_reference_subgraph)

        # check for convergence
        if w_star is None:
            # no localization
            self.current_binarized_mask = None
            return False

        return True

    # TODO: this code should align with the binarization API we have, refactor in the future
    def greedy_stepwise_binarization(self, init_w_star: np.ndarray, use_magnitude: bool,
                                     best_quantile: float = 1, worst_quantile: float = 0):
        self.current_binarized_mask = np.zeros(init_w_star.shape)

        # The first inference iteration for the current trial produced @init_w_star
        k = len(self.processed_sub_graph.G_sub.nodes)
        n = len(self.processed_sub_graph.G.nodes)
        max_worst_quantile = (n - k) / n

        # coerce quantiles
        assert 0 <= worst_quantile <= max_worst_quantile <= best_quantile <= 1
        # first step quantiles collection
        best_nodes_number = math.ceil((1 - best_quantile) * n)
        worst_nodes_number = math.floor(worst_quantile * n)

        # for the first step, discard worst nodes and fix best nodes
        best_nodes_indices, worst_nodes_indices = GreedySearchNodeClassifierSchemesServices.get_best_nodes_and_worst_nodes(init_w_star, self.chosen_nodes_indices, [], self.composite_solver,
                                           self.processed_sub_graph, self.reference_subgraph, use_magnitude=use_magnitude,
                                           best_nodes_num=best_nodes_number, worst_nodes_num=worst_nodes_number)

        self.discarded_nodes_indices = worst_nodes_indices
        for best_subgraph_node_index in best_nodes_indices:
            self.current_binarized_mask[best_subgraph_node_index] = 1
            self.chosen_nodes_indices.append(best_subgraph_node_index)

        num_steps = k - best_nodes_number
        w_star = init_w_star

        for step_number in range(num_steps):
            is_valid_step = self.__single_best_node_step(init_w_star, use_magnitude, step_number)

            GreedySearchNodeClassifierSchemesServices.measure_stepwise_binarization_progress(self.sub_graph,
                                                                                             self.processed_sub_graph,
                                                                                       num_steps, step_number,
                                                                                       w_star, self.current_binarized_mask,
                                                                                       self.chosen_nodes_indices)

            if not is_valid_step:
                break

        self.composite_solver.composite_nn.ignore_w_indices([])  # restore state of solver
        self.chosen_nodes_indices = []

        return self.current_binarized_mask

    # TODO: this code should align with the binarization API we have, refactor in the future
    def simulated_stepwise_binarization(self,
                                        init_w_star: np.ndarray, use_magnitude: bool):
        graph_size = len(self.processed_sub_graph.G.nodes)
        current_binarized_mask = np.zeros(graph_size)

        num_steps = len(self.processed_sub_graph.G_sub.nodes)
        chosen_nodes_indices = []

        for step_number in range(num_steps):

            candidate_mask_node_losses = torch.ones(graph_size) * float("inf")
            candidate_w_stars = [None for i in range(graph_size)]

            if step_number == 0:
                # first step should be lighter, since big graphs would have too many candidates, computation wise
                w_star = init_w_star
                min_candidate_mask_node_index = \
                    GreedySearchNodeClassifierSchemesServices.get_most_prominent_mask_node(step_number, w_star,
                                                                                     chosen_nodes_indices,
                                                                                           [],
                                                                                     self.composite_solver, self.sub_graph,
                                                                                     self.processed_sub_graph,
                                                                                     self.reference_subgraph,
                                                                                     use_magnitude=use_magnitude)
                candidate_mask_node_losses[min_candidate_mask_node_index] = 0 #fake loss, just to be smaller than other entries

            else:

                # go over candidate nodes
                for candidate_mask_node_index in range(graph_size):
                    if len(chosen_nodes_indices) != 0:
                        if candidate_mask_node_index in chosen_nodes_indices:
                            continue
                        if not is_neighbor(self.processed_sub_graph.G, candidate_mask_node_index, chosen_nodes_indices):
                            continue

                    # Fix the chosen node mask entry (set the mask entry to 1 regardless to the following mask training iterations)
                    self.composite_solver.composite_nn.ignore_w_indices(chosen_nodes_indices + [candidate_mask_node_index])

                    # solve the localization loop
                    w_star = self.composite_solver.init_mask_and_solve_one_round(self.sub_graph, self.original_reference_subgraph)
                    # check for convergence
                    if w_star is None:
                        # no localization
                        continue
                    candidate_w_stars[candidate_mask_node_index] = w_star
                    w_star = torch.tensor(w_star, requires_grad=True)
                    candidate_mask_node_loss = self.composite_solver.solve_using_external_params(w_star, self.processed_sub_graph.A_full,
                                                                               SubGraph(self.reference_subgraph).A_full,
                                                                                            A_node_features=self.processed_sub_graph.A_node_features,
                                                                                            A_sub_node_features=self.processed_sub_graph.A_sub_node_features,
                                                                               embedding_networks=self.composite_solver.composite_nn.embedding_networks,
                                                                               dtype=TORCH_DTYPE)
                    candidate_mask_node_losses[candidate_mask_node_index] = candidate_mask_node_loss.item()

                    # restore state of solver
                    self.composite_solver.composite_nn.ignore_w_indices(chosen_nodes_indices)

                min_candidate_mask_node_index = torch.argmin(candidate_mask_node_losses).item()

            if candidate_mask_node_losses[min_candidate_mask_node_index] == float("inf"):
                # no node could be chosen
                current_binarized_mask = None
                break

            chosen_subgraph_node_index = min_candidate_mask_node_index
            min_loss_w_star = candidate_w_stars[chosen_subgraph_node_index]

            MaskDebugger.debug_mask_scores(self.sub_graph, self.processed_sub_graph, chosen_nodes_indices,
                                                             step_number,
                                                             chosen_subgraph_node_index,
                                                             [candidate_mask_node_losses])

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1

            GreedySearchNodeClassifierSchemesServices.measure_stepwise_binarization_progress(self.sub_graph, self.processed_sub_graph,
                                                                                       num_steps, step_number,
                                                                                       min_loss_w_star, current_binarized_mask,
                                                                                       chosen_nodes_indices)

        self.composite_solver.composite_nn.ignore_w_indices([])  # restore state of solver

        return current_binarized_mask
