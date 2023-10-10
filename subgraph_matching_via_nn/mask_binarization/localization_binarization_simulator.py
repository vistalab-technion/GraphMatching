import os
import pickle
from typing import List
import numpy as np
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import \
    EmbeddingMetricNetwork
from subgraph_matching_via_nn.composite_nn.composite_nn import CompositeNeuralNetwork
import torch
from subgraph_matching_via_nn.utils.utils import plot_indicator
from subgraph_matching_via_nn.graph_processors.graph_processors import GraphProcessor
from subgraph_matching_via_nn.data.data_loaders import load_graph
from subgraph_matching_via_nn.data.paths import *
from subgraph_matching_via_nn.data.paths import DATA_PATH
from subgraph_matching_via_nn.utils.plot_services import PlotServices
from subgraph_matching_via_nn.composite_nn.composite_solver import BaseCompositeSolver
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_network_factory import \
    NodeClassifierNetworkFactory, NodeClassifierLastLayerType, NodeClassifierNetworkType
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_network_factory import \
    GraphEmbeddingNetworkFactory, EmbeddingNetworkType
from subgraph_matching_via_nn.subgraph_localization_algs.unconstrained_nonlinear_optimization import \
    binary_penalty, graph_entropy, spectral_reg, graph_total_variation
from subgraph_matching_via_nn.mask_binarization.indicator_dsitribution_binarizer import IndicatorBinarizationType, \
    IndicatorDistributionBinarizer, IndicatorBinarizationBootType


class LocalizationResult:
    sub_graph: SubGraph
    processed_sub_graph: SubGraph
    gt_indicator: np.array
    gt_indicator_tensor: torch.tensor
    w_init: torch.tensor
    w_all: List[torch.tensor]
    w_star: torch.tensor
    params: dict
    solver: BaseCompositeSolver

class LocalizationBinarizationSimulator:
    seed = 10  # for plotting
    SOLVER_MODEL_DUMP_FILE_NAME = "solver.p"
    LOCALIZATION_OBJECT_DUMP_FILE_NAME = "localization.p"

    def __init__(self, n_graph, n_subgraph, to_line=True):
        # n_graph: Number of nodes in the graph (for random graph)
        # n_subgraph: Number of nodes in the subgraph (for random graph)

        self.n_graph = n_graph
        self.n_subgraph = n_subgraph

        self.plot_services = PlotServices(LocalizationBinarizationSimulator.seed)

        self.to_line = True
        self.graph_processor = GraphProcessor(params={'to_line': to_line})

        params = {}
        params["maxiter"] = 2000
        params["lr"] = 0.0000002  # 0.0002 is good
        params["n_moments"] = 4
        params["k_update_plot"] = 250
        params['spectral_op_type'] = 'Laplacian' # 'Laplacian'. 'Adjacency'
        params["moment_type"] = "standardized_raw"  # options: 'central' ,'raw', 'standardized_raw', 'standardized_central'
        params["reg_params"] = [0]  # reg param  # 0.02  is good
        params["reg_terms"] = [graph_total_variation]
        params["quantile_level"] = (self.n_graph-self.n_subgraph)/ self.n_subgraph

        self.params = params

    def build_localization_model(self, processed_G, sub_graph, params):
        self.node_classifier_network = NodeClassifierNetworkFactory.create_node_classifier_network(processed_G,
                                                                                              NodeClassifierLastLayerType.SquaredNormalized, NodeClassifierNetworkType.Identity, params)
        self.node_classifier_network.train_node_classifier(G_sub=sub_graph.G_sub, graph_generator=None)

        embedding_nns = GraphEmbeddingNetworkFactory.create_embedding_networks(sub_graph, params,
                                                                               [EmbeddingNetworkType.Moments, EmbeddingNetworkType.Spectral])

        self.composite_nn = CompositeNeuralNetwork(node_classifier_network=self.node_classifier_network,
                                              embedding_networks=embedding_nns)
        self.embedding_metric_nn = EmbeddingMetricNetwork(loss_fun=torch.nn.MSELoss())
        self.composite_solver = BaseCompositeSolver(self.composite_nn, self.embedding_metric_nn, self.graph_processor, params)

    def draw_sub_graph(self):
        loader_params = {'graph_size': self.n_graph,
             'subgraph_size': self.n_subgraph,
             'data_path' : DATA_PATH,
             'g_full_path': COMP1_FULL_path,
             'g_sub_path': COMP1_SUB0_path}

        sub_graph = \
            load_graph(type='random',
                       loader_params=loader_params) # type = 'random', 'example', 'subcircuit'
        processed_sub_graph = self.graph_processor.pre_process(sub_graph)

        self.params["m"] = len(processed_sub_graph.G_sub.nodes())

        return sub_graph, processed_sub_graph

    def run_localization(self, sub_graph, processed_sub_graph, num_rand_exp = 1):
        w_all = []

        for k in range(num_rand_exp):
            print(k)
            _ = self.composite_nn.init_network_with_indicator(processed_sub_graph)

            w_star = self.composite_solver.solve(G=sub_graph.G, G_sub=sub_graph.G_sub)

            # x0 = self.composite_solver.set_initial_params_based_on_previous_optimum(w_star) #TODO?

            w_all.append(w_star)

        return w_all

    def save_localization_results(self, sub_graph, processed_sub_graph, w_all, w_init, output_dir_path, output_path):
        gt_indicator = sub_graph.edge_indicator if self.to_line else sub_graph.node_indicator
        gt_indicator_tensor = self.composite_nn.init_network_with_indicator(processed_sub_graph)
        w_star = self.node_classifier_network(A = processed_sub_graph.A_full, params = self.params).detach().numpy()

        localization_result = LocalizationResult()
        localization_result.sub_graph = sub_graph
        localization_result.processed_sub_graph = processed_sub_graph
        localization_result.w_all = w_all
        localization_result.gt_indicator = gt_indicator
        localization_result.gt_indicator_tensor = gt_indicator_tensor
        localization_result.w_init = w_init
        localization_result.w_star = w_star
        localization_result.params = self.params

        solver_file_path = f"{output_dir_path}{os.sep}{LocalizationBinarizationSimulator.SOLVER_MODEL_DUMP_FILE_NAME}"
        torch.save(self.composite_solver.state_dict(), solver_file_path)
        with open(solver_file_path, 'rb') as f:
            pass

        with open(output_path, 'wb') as f:
            pickle.dump(localization_result, f)

    @staticmethod
    def load_localization_results(results_output_folder_path) -> List[LocalizationResult]:
        localization_results = []

        directory = os.fsencode(results_output_folder_path)

        for file in os.listdir(directory):
            # treat file as directory, and look for two files (by name)
            filename = os.fsdecode(file)
            base_localization_result_dir = f"{results_output_folder_path}{os.sep}{filename}"

            with open(f"{base_localization_result_dir}{os.sep}{LocalizationBinarizationSimulator.LOCALIZATION_OBJECT_DUMP_FILE_NAME}", 'rb') as f:
                localization_result = pickle.load(f)

            sub_graph = localization_result.sub_graph
            simulator = LocalizationBinarizationSimulator(n_graph=len(sub_graph.G.nodes), n_subgraph=len(sub_graph.G_sub.nodes), to_line= localization_result.processed_sub_graph.is_line_graph)
            simulator.build_localization_model(localization_result.processed_sub_graph.G, localization_result.sub_graph, localization_result.params)
            model = simulator.composite_solver

            model.load_state_dict(torch.load(
                f"{base_localization_result_dir}{os.sep}{LocalizationBinarizationSimulator.SOLVER_MODEL_DUMP_FILE_NAME}",
                map_location=torch.device('cpu')))

            localization_result.solver = model
            localization_results.append(localization_result)

        return localization_results

    def run_simulation(self, num_localizations, root_results_output_path=None):

        for k in range(num_localizations):
            sub_graph, processed_sub_graph = self.draw_sub_graph()

            self.build_localization_model(processed_sub_graph.G, sub_graph, self.params)
            w_init = self.node_classifier_network(A=processed_sub_graph.A_full,
                                                  params=self.params).detach().numpy()

            w_all = self.run_localization(sub_graph, processed_sub_graph)

            if root_results_output_path is not None:
                if not os.path.isdir(root_results_output_path):
                    os.makedirs(root_results_output_path)
                results_output_path = f"{root_results_output_path}{os.sep}{k}"
                if not os.path.isdir(results_output_path):
                    os.makedirs(results_output_path)

                localization_output_path = f"{results_output_path}{os.sep}{LocalizationBinarizationSimulator.LOCALIZATION_OBJECT_DUMP_FILE_NAME}"
                self.save_localization_results(sub_graph, processed_sub_graph, w_all, w_init, results_output_path, localization_output_path)

    @staticmethod
    def apply_performance_metrics(gt_w, all_binarized_indicators_map, performance_metric_name_to_func_map):
        for metric_name, metric_func in performance_metric_name_to_func_map.items():
            for indicator_name, indicator_dict in all_binarized_indicators_map.items():
                print(f"Metric: {metric_name}")
                estimated_w = torch.stack([torch.from_numpy(item).float() for item in list(indicator_dict.values())])

                binary_gt_w = (gt_w.reshape(1, -1) > 0).long()
                binary_estimated_w = (estimated_w.reshape(1, -1) > 0).long()

                print(f"{indicator_name}: {metric_func(binary_gt_w, binary_estimated_w)}")
                print()

    @staticmethod
    def apply_binarization_scheme(solver: BaseCompositeSolver, processed_sub_graph, sub_graph, w_all, series_binarization_type: IndicatorBinarizationBootType, element_binarization_type: IndicatorBinarizationType, params):
        processed_G = processed_sub_graph.G

        w_star = solver.composite_nn.node_classifier_network(A = processed_sub_graph.A_full, params = params).detach().numpy()
        w_star_dict = dict(zip(processed_G.nodes(), w_star))

        w_bin_dict = IndicatorDistributionBinarizer.from_indicators_series_to_binary_indicator(processed_G, w_all, w_star, params, series_binarization_type, element_binarization_type)
        return {"w_star": w_star_dict, "binarized" : w_bin_dict}

    @staticmethod
    def show_localization_results(solver: BaseCompositeSolver, processed_sub_graph: SubGraph, sub_graph: SubGraph, indicator_name_to_dict_map, params, embedding_idx=1):
        # embedding_idx = 1 # to choose which embedding to plot
        # indicator_name_to_object_map = {'w_star': w_star_dict, 'w_bin': w_bin_dict, 'w_boot_bin': w_boot_dict_bin, 'w_bin_boot': w_binarize_boot_dict}

        to_line = processed_sub_graph.is_line_graph
        plot_services = PlotServices(LocalizationBinarizationSimulator.seed)
        plot_services.plot_subgraph_indicators(sub_graph.G, to_line, indicator_name_to_dict_map)

        gt_node_distribution_processed = processed_sub_graph.w_gt
        loss, ref_loss = solver.compare(processed_sub_graph.A_full, processed_sub_graph.A_sub, gt_node_distribution_processed, A_sub_indicator=None)

        indicator_name_to_vector_map = {indicator_name: list(indicator_dict.values()) for indicator_name, indicator_dict in indicator_name_to_dict_map.items()}
        solver.compare_indicators(processed_sub_graph.A_full, indicator_name_to_vector_map, embedding_idx)

        reg_terms_names = [reg_terms.__name__ for reg_terms in params['reg_terms']]

        print(f"\n{to_line = }")
        print(f"\nloss = {loss}, reg_params = {params['reg_params']}, reg_terms = {reg_terms_names}")
        print(f"loss_ref = {ref_loss}, reg_param = {params['reg_params']}, reg_terms = {reg_terms_names}")
        print(f"\nRemark: embeddings_gt and embeddings_sub might differ if we don't transform to line graph because node indicator can at best give a superset of the edges of the subgraph.")

        indicator_name_to_dict_map["w_gt"] =\
            dict(zip(processed_sub_graph.G.nodes(), processed_sub_graph.w_gt.detach().cpu().numpy()))


        estimated_indicators_names = list(indicator_name_to_dict_map.keys())
        estimated_indicators_arrays = [np.array(list(indicator_name_to_dict_map[indicator_name].values())) for indicator_name in estimated_indicators_names]
        plot_indicator(estimated_indicators_arrays, estimated_indicators_names)