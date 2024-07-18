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

    @staticmethod
    # TODO: this code should align with the binarization API we have, refactor in the future
    def greedy_stepwise_binarization(composite_solver: BaseCompositeSolver, sub_graph: SubGraph,
                                     processed_sub_graph: SubGraph, reference_subgraph: nx.Graph,
                                     original_reference_subgraph: nx.Graph,
                                     init_w_star: np.ndarray, use_magnitude: bool):
        # The first inference iteration for the current trial produced @init_w_star

        w_star = init_w_star
        current_binarized_mask = np.zeros(init_w_star.shape)

        num_steps = len(processed_sub_graph.G_sub.nodes)
        chosen_nodes_indices = []
        for step_number in range(num_steps):

            chosen_subgraph_node_index = \
                GreedySearchNodeClassifierSchemesServices.get_most_prominent_mask_node(step_number, w_star, chosen_nodes_indices,
                                                                                 composite_solver, sub_graph, processed_sub_graph,
                                                                                 reference_subgraph,
                                                                                 use_magnitude=use_magnitude)

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1

            GreedySearchNodeClassifierSchemesServices.measure_stepwise_binarization_progress(sub_graph, processed_sub_graph,
                                                                                       num_steps, step_number,
                                                                                       w_star, current_binarized_mask,
                                                                                       chosen_nodes_indices)


            # Fix the chosen node mask entry (set the mask entry to 1 regardless to the following mask training iterations)
            composite_solver.composite_nn.ignore_w_indices(chosen_nodes_indices)

            # solve the localization loop
            w_star = composite_solver.init_mask_and_solve_one_round(sub_graph, original_reference_subgraph)

            # check for convergence
            if w_star is None:
                # no localization
                current_binarized_mask = None
                break

        composite_solver.composite_nn.ignore_w_indices([])  # restore state of solver

        return current_binarized_mask

    @staticmethod
    # TODO: this code should align with the binarization API we have, refactor in the future
    def simulated_stepwise_binarization(composite_solver: BaseCompositeSolver, sub_graph: SubGraph,
                                     processed_sub_graph: SubGraph, reference_subgraph: nx.Graph,
                                     original_reference_subgraph: nx.Graph,
                                        init_w_star: np.ndarray, use_magnitude: bool):
        graph_size = len(processed_sub_graph.G.nodes)
        current_binarized_mask = np.zeros(graph_size)

        num_steps = len(processed_sub_graph.G_sub.nodes)
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
                                                                                     composite_solver, sub_graph,
                                                                                     processed_sub_graph,
                                                                                     reference_subgraph,
                                                                                     use_magnitude=use_magnitude)
                candidate_mask_node_losses[min_candidate_mask_node_index] = 0 #fake loss, just to be smaller than other entries

            else:

                # go over candidate nodes
                for candidate_mask_node_index in range(graph_size):
                    if len(chosen_nodes_indices) != 0:
                        if candidate_mask_node_index in chosen_nodes_indices:
                            continue
                        if not is_neighbor(processed_sub_graph.G, candidate_mask_node_index, chosen_nodes_indices):
                            continue

                    # Fix the chosen node mask entry (set the mask entry to 1 regardless to the following mask training iterations)
                    composite_solver.composite_nn.ignore_w_indices(chosen_nodes_indices + [candidate_mask_node_index])

                    # solve the localization loop
                    w_star = composite_solver.init_mask_and_solve_one_round(sub_graph, original_reference_subgraph)
                    # check for convergence
                    if w_star is None:
                        # no localization
                        continue
                    candidate_w_stars[candidate_mask_node_index] = w_star
                    w_star = torch.tensor(w_star, requires_grad=True)
                    candidate_mask_node_loss = composite_solver.solve_using_external_params(w_star, processed_sub_graph.A_full,
                                                                               SubGraph(reference_subgraph).A_full,
                                                                                            A_node_features=processed_sub_graph.A_node_features,
                                                                                            A_sub_node_features=processed_sub_graph.A_sub_node_features,
                                                                               embedding_networks=composite_solver.composite_nn.embedding_networks,
                                                                               dtype=TORCH_DTYPE)
                    candidate_mask_node_losses[candidate_mask_node_index] = candidate_mask_node_loss.item()

                    # restore state of solver
                    composite_solver.composite_nn.ignore_w_indices(chosen_nodes_indices)

                min_candidate_mask_node_index = torch.argmin(candidate_mask_node_losses).item()

            if candidate_mask_node_losses[min_candidate_mask_node_index] == float("inf"):
                # no node could be chosen
                current_binarized_mask = None
                break

            chosen_subgraph_node_index = min_candidate_mask_node_index
            min_loss_w_star = candidate_w_stars[chosen_subgraph_node_index]

            MaskDebugger.debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices,
                                                             step_number,
                                                             chosen_subgraph_node_index,
                                                             [candidate_mask_node_losses])

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1

            GreedySearchNodeClassifierSchemesServices.measure_stepwise_binarization_progress(sub_graph, processed_sub_graph,
                                                                                       num_steps, step_number,
                                                                                       min_loss_w_star, current_binarized_mask,
                                                                                       chosen_nodes_indices)

        composite_solver.composite_nn.ignore_w_indices([])  # restore state of solver

        return current_binarized_mask
