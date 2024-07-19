from subgraph_matching_via_nn.utils.plot_services import plot_mask_gt_vs_mask_values


class MaskDebugger:

    @staticmethod
    def debug_mask_scores(sub_graph, processed_sub_graph, chosen_nodes_indices, discarded_nodes_indices, step_number,
                          current_chosen_subgraph_node_index, edge_scores_lists):

        if not processed_sub_graph.is_line_graph:
            print("Skipping debug_mask_scores, as it is not supported for non line graphs")
            return

        # debug scores visually
        all_processed_nodes = list(processed_sub_graph.G.nodes)
        chosen_nodes = [all_processed_nodes[i] for i in chosen_nodes_indices + [current_chosen_subgraph_node_index]]
        discarded_nodes = [all_processed_nodes[i] for i in discarded_nodes_indices]

        gt_edges = list(sub_graph.G_sub.edges)

        edge_mask_dicts = [dict(zip(all_processed_nodes, edge_scores)) for edge_scores in edge_scores_lists]

        plot_mask_gt_vs_mask_values(sub_graph.G, gt_edges=gt_edges, marked_edges=chosen_nodes, disabled_edges=discarded_nodes,
                                    edge_mask_dicts=edge_mask_dicts,
                                    step_number=step_number)
