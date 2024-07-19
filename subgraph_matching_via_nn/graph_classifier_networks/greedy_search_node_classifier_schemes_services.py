import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.mask_debugger import MaskDebugger
from subgraph_matching_via_nn.utils.graph_utils import is_neighbor
from subgraph_matching_via_nn.utils.plot_services import PlotServices
from subgraph_matching_via_nn.evaluation.mask_evaluation import evaluate_mask_performance
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class GreedySearchNodeClassifierSchemesServices:

    @staticmethod
    def __mark_irrelevant_mask_node_entries(graph, chosen_nodes_indices, discarded_nodes_indices, mask_scores, is_arg_max_score_mode):
        irrelevant_mask_score_value = float("inf")
        if is_arg_max_score_mode:
            irrelevant_mask_score_value = float("-inf")

        # should choose nodes only reachable by 1-hop from current nodes, which are not already chosen
        already_marked_nodes_indices = chosen_nodes_indices + discarded_nodes_indices
        if len(chosen_nodes_indices) == 0:
            return

        for mask_node_index, _ in enumerate(mask_scores):
            if (mask_node_index in already_marked_nodes_indices) \
                    or \
                    (not is_neighbor(graph, mask_node_index, chosen_nodes_indices)):
                mask_scores[mask_node_index] = irrelevant_mask_score_value

    @staticmethod
    def get_relevant_mask_nodes_scores(w_star, chosen_nodes_indices, discarded_nodes_indices, composite_solver,
                                       processed_sub_graph, reference_subgraph):
        magnitude_w_star_copy = np.copy(w_star)
        GreedySearchNodeClassifierSchemesServices.__mark_irrelevant_mask_node_entries(processed_sub_graph.G,
                                                                           chosen_nodes_indices, discarded_nodes_indices, magnitude_w_star_copy,
                                                                           is_arg_max_score_mode=True)

        grad_w_star_copy = torch.tensor(w_star, requires_grad=True)
        updated_w_star_loss = composite_solver.solve_using_external_params(grad_w_star_copy, processed_sub_graph.A_full,
                                                                           SubGraph(reference_subgraph).A_full,
                                                                           A_node_features=processed_sub_graph.A_node_features,
                                                                           A_sub_node_features=processed_sub_graph.A_sub_node_features,
                                                                           embedding_networks=composite_solver.composite_nn.embedding_networks,
                                                                           dtype=TORCH_DTYPE)

        w_mask_grad = torch.autograd.grad(updated_w_star_loss, grad_w_star_copy, retain_graph=True, create_graph=True,
                                          allow_unused=True)[0]

        GreedySearchNodeClassifierSchemesServices.__mark_irrelevant_mask_node_entries(processed_sub_graph.G,
                                                                           chosen_nodes_indices, discarded_nodes_indices, w_mask_grad,
                                                                           is_arg_max_score_mode=False)

        return magnitude_w_star_copy, w_mask_grad

    @staticmethod
    def get_most_prominent_mask_node(step_number, w_star, chosen_nodes_indices, discarded_nodes_indices, composite_solver,
                                       sub_graph, processed_sub_graph, reference_subgraph, use_magnitude: bool):
        magnitude_w_star_copy, w_mask_grad = GreedySearchNodeClassifierSchemesServices.get_relevant_mask_nodes_scores(
            w_star, chosen_nodes_indices, discarded_nodes_indices, composite_solver,
            processed_sub_graph, reference_subgraph)

        # decide on the most prominent mask node entry (according to grad/how binary it is/magnitude)
        if use_magnitude:
            chosen_subgraph_node_index = np.argmax(magnitude_w_star_copy.reshape(-1))
        else:
            chosen_subgraph_node_index = torch.argmin(w_mask_grad.reshape(-1)).item()

        MaskDebugger.debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices, discarded_nodes_indices,
                                                         step_number,
                                                         chosen_subgraph_node_index,
                                                         [magnitude_w_star_copy, w_mask_grad])

        return chosen_subgraph_node_index

    @staticmethod
    def get_best_nodes_and_worst_nodes(w_star, chosen_nodes_indices, discarded_nodes_indices, composite_solver,
                                       processed_sub_graph, reference_subgraph, use_magnitude: bool,
                                       best_nodes_num, worst_nodes_num):
        magnitude_w_star_copy, w_mask_grad = GreedySearchNodeClassifierSchemesServices.get_relevant_mask_nodes_scores(
            w_star, chosen_nodes_indices, discarded_nodes_indices, composite_solver,
            processed_sub_graph, reference_subgraph)

        # decide on the best mask node entries and the worst ones (according to grad/how binary it is/magnitude)
        n = len(processed_sub_graph.G)
        if best_nodes_num == 0:
            top_k_slice_start_index = 0
            top_k_slice_end_index = 0
        elif best_nodes_num == n:
            top_k_slice_start_index = 0
            top_k_slice_end_index = n
        else:
            top_k_slice_start_index = n - best_nodes_num
            top_k_slice_end_index = n

        if use_magnitude:
            magnitude_array_sorted_indices = np.argsort(magnitude_w_star_copy.reshape(-1))
            # Top-k items
            top_k_indices = magnitude_array_sorted_indices[top_k_slice_start_index: top_k_slice_end_index]
            # Worst-m items
            worst_m_indices = magnitude_array_sorted_indices[:worst_nodes_num]
        else:
            grad_tensor = w_mask_grad.reshape(-1)
            # Top-k items
            top_k_indices = torch.topk(-grad_tensor, best_nodes_num).indices
            # Worst-m items
            worst_m_indices = torch.topk(grad_tensor, worst_nodes_num).indices

        return top_k_indices.tolist(), worst_m_indices.tolist()

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

