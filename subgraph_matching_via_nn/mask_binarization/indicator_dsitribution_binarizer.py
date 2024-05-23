import os
from enum import Enum
import kmeans1d
import networkx as nx
import numpy as np
import scipy as sp
import torch

from subgraph_matching_via_nn.composite_nn.composite_solver import BaseCompositeSolver
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_evaluation import evaluate_mask_performance
from subgraph_matching_via_nn.mask_binarization.LP_binarization import solve_maximum_weight_subgraph
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix
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


class IndicatorDistributionBinarizer:

    @staticmethod
    def get_most_prominent_mask_node(w_star, chosen_nodes_indices, composite_solver,
                                     processed_sub_graph, reference_subgraph, use_magnitude: bool):
        # decide on the most prominent mask node entry (according to grad/how binary it is/magnitude)
        w_star_copy = np.copy(w_star)

        if use_magnitude:
            # don't take into account already chosen nodes
            for mask_node_index, _ in enumerate(w_star):
                if mask_node_index in chosen_nodes_indices:
                    w_star_copy[mask_node_index] = float("-inf")

            chosen_subgraph_node_index = np.argmax(w_star_copy.reshape(-1))
            return chosen_subgraph_node_index

        w_star_copy = torch.tensor(w_star, requires_grad=True)
        updated_w_star_loss = composite_solver.solve_using_external_params(w_star_copy, processed_sub_graph.A_full,
                                                                           SubGraph(reference_subgraph).A_full,
                                                                           embedding_networks=composite_solver.composite_nn.embedding_networks,
                                                                           dtype=TORCH_DTYPE)

        w_mask_grad = torch.autograd.grad(updated_w_star_loss, w_star_copy, retain_graph=True, create_graph=True,
                                          allow_unused=True)[0]

        # don't take into account already chosen nodes
        for mask_node_index in range(len(w_mask_grad)):
            if mask_node_index in chosen_nodes_indices:
                w_mask_grad[mask_node_index] = float("inf")
        chosen_subgraph_node_index = torch.argmin(w_mask_grad.reshape(-1)).item()

        return chosen_subgraph_node_index

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
                IndicatorDistributionBinarizer.get_most_prominent_mask_node(w_star, chosen_nodes_indices,
                                                                            composite_solver, processed_sub_graph,
                                                                            reference_subgraph,
                                                                            use_magnitude=use_magnitude)

            chosen_nodes_indices.append(chosen_subgraph_node_index)
            current_binarized_mask[chosen_subgraph_node_index] = 1
            print(f"Node #{step_number + 1} out of {num_steps}, was chosen during greedy binarization.{os.linesep}"
                  f"Current mask is: {current_binarized_mask}.{os.sep}Order of selected nodes is {chosen_nodes_indices}")

            # logging
            with open("localization cell output.txt", "a") as myfile:
                myfile.write(f"Ref subgraph:{os.linesep}"
                             f"w_star={w_star}{os.linesep}current w_rounded={current_binarized_mask}{os.linesep}")
                if step_number == num_steps - 1:
                    myfile.write(f"{evaluate_mask_performance(current_binarized_mask, processed_sub_graph.w_gt)}{os.linesep}")
                    break

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
    def binarize(graph: nx.graph, w: np.array, params, type: IndicatorBinarizationType, as_dict:bool=True):
        if type == IndicatorBinarizationType.KMeans:
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == IndicatorBinarizationType.TopK:
            w_th = top_m(w, params["m"])
        elif type == IndicatorBinarizationType.Quantile:
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        elif type == IndicatorBinarizationType.Diffusion:
            A = (nx.adjacency_matrix(graph)).toarray()
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
            A = (nx.adjacency_matrix(graph)).toarray()
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
            A = (nx.adjacency_matrix(graph)).toarray()
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

            A = (nx.adjacency_matrix(graph)).toarray()
            selected_nodes, selected_edges = solve_maximum_weight_subgraph(w, A, num_nodes, num_edges)
            print(f'requested: n_nodes = {num_nodes}, n_edges : {num_edges}')
            print(f'found: n_nodes = {len(selected_nodes)}, n_edges : {len(selected_edges)}')
            w_th = np.zeros([len(graph.nodes()), 1])
            w_th[selected_nodes] = 1.0
        else:
            w_th = w

        w_th = w_th / w_th.sum()

        if as_dict:
            return dict(zip(graph.nodes(), w_th))
        return w_th

    @staticmethod
    def from_indicators_series_to_binary_indicator(processed_G, w_all, w_star, params,
                                                   series_binarization_type: IndicatorBinarizationBootType,
                                                   element_binarization_type: IndicatorBinarizationType):

        binarize = IndicatorDistributionBinarizer.binarize

        if series_binarization_type == IndicatorBinarizationBootType.OptimalElement:
            return binarize(processed_G, w_star, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesNormalizedMean:
            w_boot = np.mean(np.array(w_all), axis=0)
            w_boot = binarize(processed_G, w_boot, params, None, as_dict=False)
            return binarize(processed_G, w_boot, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesMedianOfBinarizedElements:
            return binarize(processed_G, np.median(
                np.array([list(binarize(processed_G, w, params, element_binarization_type).values()) for w in w_all]), axis=0),
                            params, element_binarization_type)
        else:
            raise ValueError(f"Unsupported series binarization type: {series_binarization_type}")
