import os
import pickle
import random
from abc import abstractmethod, ABC

import networkx as nx
import numpy as np

from subgraph_matching_via_nn.composite_nn.composite_solver import build_composite_solver
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.evaluation.localization_inference import LocalizationInference
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_network_factory import \
    NodeClassifierNetworkFactory
from subgraph_matching_via_nn.graph_processors.graph_processors import GraphProcessor


class FullGraphPerturbationVsSubgraphDetectionAnalysis(ABC):
    def __init__(self, composite_solver, node_classifier_factory: NodeClassifierNetworkFactory, params, plot_services,
                 binarization_type, series_binarization_func, use_full_graph_edges: bool,
                 subgraph_instance_name: str = "General"):
        super(FullGraphPerturbationVsSubgraphDetectionAnalysis, self).__init__()
        self.composite_solver = composite_solver
        self.params = params
        self.plot_services = plot_services
        self.binarization_type = binarization_type
        self.series_binarization_func = series_binarization_func
        self.use_full_graph_edges = use_full_graph_edges
        self.node_classifier_factory = node_classifier_factory
        self.subgraph_instance_name = subgraph_instance_name

    def _create_localization_inference_instance(self, g, g_sub):
        # prepare experiment object
        sub_graph = SubGraph(g, g_sub)
        to_line = self.params['to_line']

        graph_processor = GraphProcessor(params={'to_line': to_line})  # , 'to_undirected': 'symmetrize'})
        processed_sub_graph = graph_processor.pre_process(sub_graph)

        reference_subgraph = processed_sub_graph.G_sub
        original_reference_subgraph = sub_graph.G_sub

        params = self.params
        processed_G = processed_sub_graph.G

        node_classifier_network = self.node_classifier_factory.create(processed_G)
        composite_solver = build_composite_solver(self.composite_solver.composite_nn.embedding_networks,
                                                  self.composite_solver.embedding_metric_nn, node_classifier_network,
                                                  graph_processor, params, params['device'])

        localization_inference = LocalizationInference(sub_graph, processed_sub_graph, original_reference_subgraph,
                                                       reference_subgraph, composite_solver, params,
                                                       self.plot_services)
        return localization_inference

    def run_experiment(self, localization_inference_instance):
        # run inference
        w_all = []

        is_converged, single_round_inference_result = localization_inference_instance.single_round_subgraph_inference(
            self.binarization_type, w_all, self.series_binarization_func)

        # check for convergence
        if not is_converged:
            # no localization
            return None

        return single_round_inference_result

    def evaluate_experiment_scores(self, localization_inference_instance, single_round_inference_result):
        if single_round_inference_result is None:
            return float("nan")

        _, w_rounded, w_star = single_round_inference_result

        scores_map = localization_inference_instance.experiment_scores_evaluation(binarized_solution=w_rounded)
        return scores_map

    @abstractmethod
    def _get_experiment_header(self):
        pass

    def _save_results(self, results_map):
        experiment_header = self._get_experiment_header()
        with open(f"{experiment_header}_{self.subgraph_instance_name}.txt", 'wb') as f:
            pickle.dump(results_map, f)

    @abstractmethod
    def _perturbation_stoppage_criteria(self, curr_subgraph: nx.Graph):
        pass

    def __add_node_to_subgraph(self, subgraph: nx.Graph, node_id, full_graph):
        subgraph.add_node(node_id)

        if full_graph is None:
            return

        # add node features according to full graph
        # pick full graph node in random, and take the features from it
        random_full_graph_node_id = list(full_graph.nodes)[random.randint(0, len(full_graph) - 1)]
        random_node_attributes = full_graph.nodes(data=True)[random_full_graph_node_id]

        new_node_attributes = subgraph.nodes(data=True)[node_id]

        for feature_name, feature_val in random_node_attributes.items():
            new_node_attributes[feature_name] = feature_val

    def __perturb_subgraph_via_new_edges(self, subgraph):
        nodes = list(subgraph.nodes)
        while True:
            new_node = max(subgraph.nodes) + 1
            self.__add_node_to_subgraph(subgraph, new_node, full_graph=None)
            subgraph.add_edge(np.random.choice(nodes), new_node)
            if self._perturbation_stoppage_criteria(subgraph):
                break

    def _perturb_subgraph(self, subgraph, full_graph=None):
        """
        Increase the diameter of the graph using nodes and edges from G,
        and then add new nodes and edges once G is fully utilized.
        """
        # TODO: use yield for better performance

        print(f"Adding random edges until the diameter increases.")
        nodes_in_graph = set(subgraph.nodes)

        if self.use_full_graph_edges:
            # Add nodes and edges from G until all are used or diameter increases
            for u, v in full_graph.edges:
                if u not in nodes_in_graph or v not in nodes_in_graph:
                    subgraph.add_edge(u, v)
                    if u not in nodes_in_graph:
                        self.__add_node_to_subgraph(subgraph, u, full_graph)
                    if v not in nodes_in_graph:
                        self.__add_node_to_subgraph(subgraph, v, full_graph)
                    if self._perturbation_stoppage_criteria(subgraph):
                        return

        self.__perturb_subgraph_via_new_edges(subgraph)

    def _localize_and_measure_detection(self, current_g_sub, original_g_sub):
        localization_inference_instance = self._create_localization_inference_instance(current_g_sub, original_g_sub)

        # Run the experiment and get the score
        experiment_result = self.run_experiment(localization_inference_instance)
        scores_map = self.evaluate_experiment_scores(localization_inference_instance, experiment_result)
        return scores_map

    def _log_message(self, log_message):
        # logging #TODO - refactor to logging service
        with open("localization cell output.txt", "a") as myfile:
            myfile.write(f"{os.linesep}{log_message}{os.linesep}")
        # print(log_message)


class DiameterVsSubgraphDetectionAnalysis(FullGraphPerturbationVsSubgraphDetectionAnalysis):

    def __init__(self, composite_solver, node_classifier_factory: NodeClassifierNetworkFactory, params, plot_services,
                 binarization_type, series_binarization_func, use_full_graph_edges: bool,
                 subgraph_instance_name: str = "General"):
        super(DiameterVsSubgraphDetectionAnalysis, self).__init__(composite_solver, node_classifier_factory, params,
                                                                  plot_services, binarization_type, series_binarization_func,
                                                                  use_full_graph_edges, subgraph_instance_name)
        self.target_diameter = -1

    def __log(self, current_diameter, max_diameter):
        log_message = f"current diameter = {current_diameter}; max diameter = {max_diameter}"
        self._log_message(log_message)

    def _get_experiment_header(self):
        return "diameter analysis"

    def _perturbation_stoppage_criteria(self, curr_subgraph: nx.Graph):
        return nx.diameter(curr_subgraph) >= self.target_diameter

    def run(self, g_full: nx.Graph, g_sub: nx.Graph, max_diameter):
        original_g_sub = g_sub
        current_graph = g_sub.copy()

        # Initialize
        current_diameter = nx.diameter(current_graph)
        results = {}

        # Increase the diameter and run experiments
        while current_diameter <= max_diameter:
            scores_map = self._localize_and_measure_detection(current_graph, original_g_sub)

            results[current_diameter] = scores_map
            self.__log(current_diameter, max_diameter)

            self.target_diameter = current_diameter + 1
            self._perturb_subgraph(current_graph, full_graph=g_full)
            current_diameter = nx.diameter(current_graph)

        self._save_results(results)

        return results


class NodesNumberVsSubgraphDetectionAnalysis(FullGraphPerturbationVsSubgraphDetectionAnalysis):

    def __init__(self, composite_solver, node_classifier_factory: NodeClassifierNetworkFactory, params, plot_services,
                 binarization_type, series_binarization_func, use_full_graph_edges: bool,
                 subgraph_instance_name: str = "General"):
        super(NodesNumberVsSubgraphDetectionAnalysis, self).__init__(composite_solver, node_classifier_factory, params,
                                                                  plot_services, binarization_type, series_binarization_func,
                                                                  use_full_graph_edges, subgraph_instance_name)
        self.target_nodes_number = -1

    def __log(self, current_n_nodes, max_n_nodes):
        log_message = f"current #nodes = {current_n_nodes}; max #nodes = {max_n_nodes}"
        self._log_message(log_message)

    def _get_experiment_header(self):
        return "nodes number analysis"

    def _perturbation_stoppage_criteria(self, curr_subgraph: nx.Graph):
        return len(curr_subgraph) >= self.target_nodes_number

    def run(self, g_full: nx.Graph, g_sub: nx.Graph, max_n_nodes):
        original_g_sub = g_sub.copy()
        current_graph = g_sub

        # Initialize
        current_n_nodes = len(current_graph)
        results = {}

        # Increase the diameter and run experiments
        while current_n_nodes <= max_n_nodes:
            scores_map = self._localize_and_measure_detection(current_graph, original_g_sub)

            results[current_n_nodes] = scores_map
            self.__log(current_n_nodes, max_n_nodes)

            self.target_nodes_number = current_n_nodes + 1
            self._perturb_subgraph(current_graph, full_graph=g_full)
            current_n_nodes = len(current_graph)

        self._save_results(results)

        return results
