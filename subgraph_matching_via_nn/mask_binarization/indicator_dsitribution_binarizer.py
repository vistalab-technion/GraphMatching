import os
from enum import Enum
import kmeans1d
import networkx as nx
import numpy as np
import scipy as sp
import torch
from matplotlib import pyplot as plt

from subgraph_matching_via_nn.composite_nn.composite_solver import BaseCompositeSolver
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_evaluation import evaluate_mask_performance
from subgraph_matching_via_nn.mask_binarization.LP_binarization import solve_maximum_weight_subgraph
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix, is_neighbor
from subgraph_matching_via_nn.utils.plot_services import PlotServices, plot_mask_gt_vs_mask_values
from subgraph_matching_via_nn.utils.utils import NP_DTYPE, top_m, TORCH_DTYPE


class IndicatorBinarizationBootType(Enum):
    OptimalElement = 0,
    SeriesNormalizedMean = 1,
    SeriesMedianOfBinarizedElements = 2,


class IndicatorBinarizationType(Enum):
    KMeans = 0,
    TopK = 1,
    Quantile = 2,
    Diffusion = 3,
    zoomout = 4,
    nonlinear_zoomout = 5,
    mwksp = 6,
    greedy_stepwise_mwksp = 7,
    simulated_greedy_stepwise = 8,


class IndicatorDistributionBinarizer:

    @staticmethod
    def __mark_irrelevant_mask_node_entries(graph, chosen_nodes_indices, mask_scores, is_arg_max_score_mode):
        irrelevant_mask_score_value = float("inf")
        if is_arg_max_score_mode:
            irrelevant_mask_score_value = float("-inf")

        # should choose nodes only reachable by 1-hop from current nodes, which are not already chosen

        if len(chosen_nodes_indices) == 0:
            return

        for mask_node_index, _ in enumerate(mask_scores):
            if (mask_node_index in chosen_nodes_indices) \
                    or \
                    (not is_neighbor(graph, mask_node_index, chosen_nodes_indices)):
                mask_scores[mask_node_index] = irrelevant_mask_score_value

    @staticmethod
    def debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices, step_number,
                          current_chosen_subgraph_node_index, edge_scores_lists):

        if not processed_sub_graph.is_line_graph:
            print("Skipping debug_mask_scores, as it is not supported for non line graphs")
            return

        # debug scores visually
        all_processed_nodes = list(processed_sub_graph.G.nodes)
        chosen_nodes = [all_processed_nodes[i] for i in chosen_nodes_indices + [current_chosen_subgraph_node_index]]

        gt_edges = list(sub_graph.G_sub.edges)

        edge_mask_dicts = [dict(zip(all_processed_nodes, edge_scores)) for edge_scores in edge_scores_lists]

        plot_mask_gt_vs_mask_values(sub_graph.G, gt_edges=gt_edges, marked_edges=chosen_nodes,
                                    edge_mask_dicts=edge_mask_dicts,
                                    step_number=step_number)

    @staticmethod
    def get_most_prominent_mask_node(step_number, w_star, chosen_nodes_indices, composite_solver,
                                     sub_graph, processed_sub_graph, reference_subgraph, use_magnitude: bool):
        magnitude_w_star_copy = np.copy(w_star)
        IndicatorDistributionBinarizer.__mark_irrelevant_mask_node_entries(processed_sub_graph.G,
                                                                           chosen_nodes_indices, magnitude_w_star_copy,
                                                                           is_arg_max_score_mode=True)
        magnitude_chosen_subgraph_node_index = np.argmax(magnitude_w_star_copy.reshape(-1))

        grad_w_star_copy = torch.tensor(w_star, requires_grad=True)
        updated_w_star_loss = composite_solver.solve_using_external_params(grad_w_star_copy, processed_sub_graph.A_full,
                                                                           SubGraph(reference_subgraph).A_full,
                                                                           A_node_features=processed_sub_graph.A_node_features,
                                                                           A_sub_node_features=processed_sub_graph.A_sub_node_features,
                                                                           embedding_networks=composite_solver.composite_nn.embedding_networks,
                                                                           dtype=TORCH_DTYPE)

        w_mask_grad = torch.autograd.grad(updated_w_star_loss, grad_w_star_copy, retain_graph=True, create_graph=True,
                                          allow_unused=True)[0]

        IndicatorDistributionBinarizer.__mark_irrelevant_mask_node_entries(processed_sub_graph.G,
                                                                           chosen_nodes_indices, w_mask_grad,
                                                                           is_arg_max_score_mode=False)
        grad_chosen_subgraph_node_index = torch.argmin(w_mask_grad.reshape(-1)).item()

        # decide on the most prominent mask node entry (according to grad/how binary it is/magnitude)
        if use_magnitude:
            chosen_subgraph_node_index = magnitude_chosen_subgraph_node_index
        else:
            chosen_subgraph_node_index = grad_chosen_subgraph_node_index

        IndicatorDistributionBinarizer.debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices,
                                                         step_number,
                                                         chosen_subgraph_node_index,
                                                         [magnitude_w_star_copy, w_mask_grad])

        return chosen_subgraph_node_index

    @staticmethod
    def measure_stepwise_binarization_progress(sub_graph, processed_sub_graph, num_steps, step_number,
                                               w_star, current_binarized_mask, chosen_nodes_indices):
        seed = 10  # for plotting
        plot_services = PlotServices(seed)

        print(f"Node #{step_number + 1} out of {num_steps}, was chosen during greedy binarization.{os.linesep}"
              f"Current mask is: {current_binarized_mask}.{os.sep}Order of selected nodes is {chosen_nodes_indices}")

        # plot temporary localized graph vs gt graph
        w_parial_binarized_dict = dict(zip(processed_sub_graph.G.nodes(), current_binarized_mask))
        w_gt_dict = dict(zip(processed_sub_graph.G.nodes(), np.array(processed_sub_graph.w_gt)))
        indicator_name_to_object_map = {'w_partial_binarized': w_parial_binarized_dict, 'gt sub': w_gt_dict}
        plot_services.plot_subgraph_indicators(sub_graph.G, processed_sub_graph.is_line_graph,
                                               indicator_name_to_object_map, is_show=False)
        graph_file_name = "greedy_stepwise_binarization step %d.png" % step_number
        plt.savefig(graph_file_name, format="PNG")

        # logging
        with open("localization cell output.txt", "a") as myfile:
            myfile.write(f"Ref subgraph:{os.linesep}"
                         f"w_star={w_star}{os.linesep}current w_rounded={current_binarized_mask}{os.linesep}")
            if step_number == num_steps - 1:
                myfile.write(
                    f"{evaluate_mask_performance(current_binarized_mask, processed_sub_graph.w_gt)}{os.linesep}")
                return True

        return False

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
                IndicatorDistributionBinarizer.get_most_prominent_mask_node(step_number, w_star, chosen_nodes_indices,
                                                                            composite_solver, sub_graph, processed_sub_graph,
                                                                            reference_subgraph,
                                                                            use_magnitude=use_magnitude)

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1

            IndicatorDistributionBinarizer.measure_stepwise_binarization_progress(sub_graph, processed_sub_graph,
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
                    IndicatorDistributionBinarizer.get_most_prominent_mask_node(step_number, w_star,
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

            IndicatorDistributionBinarizer.debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices,
                                                             step_number,
                                                             chosen_subgraph_node_index,
                                                             [candidate_mask_node_losses])

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1

            IndicatorDistributionBinarizer.measure_stepwise_binarization_progress(sub_graph, processed_sub_graph,
                                                                                  num_steps, step_number,
                                                                                  min_loss_w_star, current_binarized_mask,
                                                                                  chosen_nodes_indices)

        composite_solver.composite_nn.ignore_w_indices([])  # restore state of solver

        return current_binarized_mask

    @staticmethod
    def binarize(original_graph: nx.graph, processed_graph: nx.graph, w: np.array, params, type: IndicatorBinarizationType, as_dict:bool=True):
        if type == IndicatorBinarizationType.KMeans:
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == IndicatorBinarizationType.TopK:
            w_th = top_m(w, params["m"])
        elif type == IndicatorBinarizationType.Quantile:
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        elif type == IndicatorBinarizationType.Diffusion:
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            # # Eigenvalue decomposition of the Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            k = 10

            # Generate k logarithmically spaced values of t from a large to small value
            max_t = 10  # Change this to your desired maximum value
            min_t = 0.01  # Change this to your desired minimum value
            t_values = np.logspace(np.log10(max_t), np.log10(min_t), k)

            w_th = w
            for t in t_values:
                # Apply the heat kernel using matrix exponentiation
                heat_matrix = eigenvectors @ np.diag(
                    np.exp(-t * eigenvalues)) @ eigenvectors.T
                heat_w = heat_matrix @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])
        elif type == IndicatorBinarizationType.zoomout:
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            w_th = w
            for i in range(2, A.shape[0]):
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])

        elif type == IndicatorBinarizationType.nonlinear_zoomout:
            A = (nx.adjacency_matrix(processed_graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            w_th = w
            heat_w = w
            for i in range(2, A.shape[0]):
                E = graph_edit_matrix(A, 1 - params["m"] * w_th)
                Ae = A - E

                De = np.diag(Ae.sum(axis=1))
                Le = De - Ae

                # Generalized eigenvalue decomposition of the Random Walk Laplacian
                eigenvalues, eigenvectors = np.linalg.eigh(Le)
                # eigenvalues, eigenvectors = sp.linalg.eigh(Le, De)
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])
        elif type == IndicatorBinarizationType.mwksp:
            num_nodes = params['m']
            num_edges = params['n']

            A = (nx.adjacency_matrix(original_graph)).toarray()

            selected_nodes_map, selected_edges_map = solve_maximum_weight_subgraph(w, A, num_nodes, num_edges)
            print(f'requested: n_nodes = {num_nodes}, n_edges : {num_edges}')
            print(f'found: n_nodes = {len(selected_nodes_map)}, n_edges : {len(selected_edges_map)}')

            # convert resulting mask W to the format the processed graph is working with (in terms of line graph format)
            is_working_on_node_mask = (len(w) == A.shape[0])
            if is_working_on_node_mask:
                pass
            else:
                # if working on a line graph, convert the result edges mask to the node mask we are working on
                selected_nodes_map = selected_edges_map

            w_th = np.zeros([len(processed_graph.nodes()), 1])
            for node_index, node in enumerate(processed_graph.nodes()):
                mask_val = selected_nodes_map.get(node, None)
                if mask_val is None:
                    continue
                w_th[node_index] = mask_val
        else:
            w_th = w

        w_th = w_th / w_th.sum()

        if as_dict:
            return dict(zip(processed_graph.nodes(), w_th))
        return w_th

    @staticmethod
    def from_indicators_series_to_binary_indicator(original_G, processed_G, w_all, w_star, params,
                                                   series_binarization_type: IndicatorBinarizationBootType,
                                                   element_binarization_type: IndicatorBinarizationType):

        binarize = IndicatorDistributionBinarizer.binarize

        if series_binarization_type == IndicatorBinarizationBootType.OptimalElement:
            return binarize(original_G, processed_G, w_star, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesNormalizedMean:
            w_boot = np.mean(np.array(w_all), axis=0)
            w_boot = binarize(original_G, processed_G, w_boot, params, None, as_dict=False)
            return binarize(original_G, processed_G, w_boot, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesMedianOfBinarizedElements:
            return binarize(original_G, processed_G, np.median(
                np.array([list(binarize(original_G, processed_G, w, params, element_binarization_type).values()) for w in w_all]), axis=0),
                            params, element_binarization_type)
        else:
            raise ValueError(f"Unsupported series binarization type: {series_binarization_type}")
