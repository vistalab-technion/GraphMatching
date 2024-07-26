import os
import pickle
import sys
# sys.path.append("/home/sanketh/DANI/GraphMatching/")
import torch.multiprocessing as mp

import torch
from debugpy.common.util import nameof
from matplotlib import pyplot as plt

from powerful_gnns.models.graphcnn import GraphCNN
from subgraph_matching_via_nn.composite_nn.composite_solver import build_composite_solver
from subgraph_matching_via_nn.composite_nn.composite_solver_optimizer_type import CompositeSolverOptimizerType
from subgraph_matching_via_nn.data.data_loaders import load_graph
from subgraph_matching_via_nn.data.paths import DATA_PATH
from subgraph_matching_via_nn.evaluation.fullgraph_perturbation_subgraph_detection_analysis import \
    NodesNumberVsSubgraphDetectionAnalysis
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_network_factory import \
    NodeClassifierNetworkType, NodeClassifierNetworkFactory, NodeClassifierLastLayerType
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_network_factory import \
    GraphEmbeddingNetworkFactory, EmbeddingNetworkType
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import MomentEmbeddingType
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import EmbeddingMetricNetwork
from subgraph_matching_via_nn.graph_processors.graph_processors import GraphProcessor
from subgraph_matching_via_nn.mask_binarization.indicator_dsitribution_binarizer import IndicatorBinarizationType, \
    IndicatorDistributionBinarizer
from subgraph_matching_via_nn.utils.plot_services import PlotServices
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


def plot_analysis_scores(results, score_name, x_title):
    fig, ax = plt.subplots(1)
    x = list(results.keys())
    y = [results[i][score_name] for i in x]
    # plot the data
    plt.xlabel(x_title)
    plt.ylabel(score_name)
    ax.plot(x, y)
    plt.show()

def load_sub_graph(loader_params):


    sub_graph = \
        load_graph(type='subcircuit',
                   loader_params=loader_params)

    return sub_graph

def build_general_params_map(sub_graph, processed_sub_graph, device, to_line, graphcnn_hidden_dim, moment_type: MomentEmbeddingType, solver_type, maxiter=1_500,
                             reg_terms=[], reg_params=[]):
    params = {}
    # params["solver_type"] = 'gd'
    params["maxiter"] = maxiter
    params['lr'] = 1e-2  # 2e-5# try lr=8e-8 with reg node and edges penalty of 1 [0, 0, 1, 1] # 2e-03 #2e-07
    params["n_moments"] = 10
    params["m"] = len(processed_sub_graph.G_sub.nodes())
    params["n"] = len(processed_sub_graph.G_sub.edges())
    params['node_features_number'] = sub_graph.node_features_number
    params["k_update_plot"] = 250
    params['spectral_op_type'] = 'Laplacian'  # 'Laplacian'. 'Adjacency'
    params[
        "moment_type"] = moment_type
    params['num_mid_layers'] = 10
    params["reg_params"] = reg_params
    params[
        "reg_terms"] = reg_terms
    # params["quantile_level"] = (params["n"]-params["m"])/params["m"]
    params["graphcnn_hidden_dim"] = graphcnn_hidden_dim  # 16 #1024 #512#2048# 256# 128
    params['device'] = device
    params[
        'is_use_model_compliation'] = False  # Only supported for Pytorch > 2.0, and currently not supported on Windows
    params['scaler'] = 1  # e+05 #1e-03#e+3#8 # scale down loss terms
    params['apply_quantization'] = False
    # params['dynamic_metric_training_experiments_pace'] = 10
    params['max_grad_norm'] = 20
    params['to_line'] = to_line
    params['weight_decay'] = 1e-04
    params['solver_type'] = solver_type

    return params

def set_experiment_folder(experiment_header, experiment_id, experiment_config_map):
    subgraph_instance_name = experiment_config_map['subgraph_instance_name']

    # create experiment folder
    dump_path = f"{experiment_header}{os.sep}{subgraph_instance_name}{os.sep}{experiment_id}{os.sep}"

    experiment_results_path = f"{dump_path}results.p"
    dir_path = os.path.dirname(experiment_results_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # save config file
    experiment_results_path = f"{dump_path}config.p"
    with open(experiment_results_path, 'wb') as f:
        pickle.dump(experiment_config_map, f)

    return experiment_results_path


def setup_experiment():
    #region read example
    sub_graph = load_sub_graph(loader_params)

    graph_processor = GraphProcessor(params={'to_line': to_line})  # , 'to_undirected': 'symmetrize'})
    processed_sub_graph = graph_processor.pre_process(sub_graph)

    processed_G = processed_sub_graph.G
    #endregion

    # configure params
    params = build_general_params_map(sub_graph, processed_sub_graph, device, to_line, graphcnn_hidden_dim, moment_type, maxiter=maxiter,
                                      solver_type=solver_type,
                                      reg_terms = [], reg_params = [])
    params['node_classifier_network_type'] = node_classifier_network_type

    #region embeddings
    # Graph CNN model factory definition
    model_factory_func = lambda device: GraphCNN(num_layers=5, num_mlp_layers=2,
                                                 input_dim=params['node_features_number'],
                                                 hidden_dim=params["graphcnn_hidden_dim"], output_dim=1,
                                                 final_dropout=0.5, learn_eps=False, graph_pooling_type="sum",
                                                 neighbor_pooling_type="sum", device=device).to(dtype=TORCH_DTYPE)
    params["graphcnn_factory_func"] = model_factory_func

    previous_model_output_path = None
    embedding_nns = GraphEmbeddingNetworkFactory.create_embedding_networks(sub_graph, params,
                                                                           [embedding_type])
    for embedding_nn in embedding_nns:
        _ = embedding_nn.eval()

    # prepare attributes for potential dump by dataloader of localization examples ->
    # avoid using lambda functions, not stable on some platforms and torch version
    params["graphcnn_factory_func"] = None
    #endregion

    #region node classifier

    node_classifier_network_factory = NodeClassifierNetworkFactory(
        last_layer_type=last_layer_type,
        node_classifier_network_type=node_classifier_network_type, params=params)
    node_classifier_network = node_classifier_network_factory.create(processed_G)
    #endregion

    #region solver
    params['gradient_average_iterations_amount'] = gradient_average_iterations_amount
    params['greedy_search_node_classifier_best_quantile'] = greedy_search_node_classifier_best_quantile
    params['greedy_search_node_classifier_worst_quantile'] = greedy_search_node_classifier_worst_quantile

    embedding_metric_nn = EmbeddingMetricNetwork(loss_fun=torch.nn.MSELoss(), params=params)

    composite_solver = build_composite_solver(embedding_nns, embedding_metric_nn, node_classifier_network,
                                              graph_processor, params, device)
    #endregion

    return composite_solver, node_classifier_network_factory, sub_graph, params


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Override the show method to do nothing
    plt.show = lambda: None

    mp.set_start_method("spawn")

    # region params
    device = 'cpu'
    is_use_features = True
    to_line = False
    loader_params = {'data_path': DATA_PATH,
                     'g_full_path': f'comp1_2{os.sep}full_graph.p',
                     'g_sub_path': f'comp1_2{os.sep}subgraph0.p',
                     'is_use_features': is_use_features}

    graphcnn_hidden_dim = 128
    moment_type = MomentEmbeddingType.RawWithFeatures  # MomentEmbeddingType.RawWithFeatures #MomentEmbeddingType.Raw
    # embedding_type = EmbeddingNetworkType.Moments  ## Stub, Spectral, GraphCNN
    # node_classifier_network_type = NodeClassifierNetworkType.NN  # NodeClassifierNetworkType.NN/NodeClassifierNetworkType.GoogleSoftmax/NodeClassifierNetworkType.GAN/Identity NodeClassifierNetworkType.Clustering
    # last_layer_type = NodeClassifierLastLayerType.SquaredNormalized #NodeClassifierLastLayerType.Sigmoid, NodeClassifierLastLayerType.Identity
    solver_type = CompositeSolverOptimizerType.ADAM  # CompositeSolverOptimizerType.STUB # CompositeSolverOptimizerType.FW_continuous # CompositeSolverOptimizerType.FW_binary.GD #CompositeSolverOptimizerType.ADAM #CompositeSolverOptimizerType.LBFGS

    greedy_search_node_classifier_best_quantile = 1 #0.875
    greedy_search_node_classifier_worst_quantile = 0 #0.125
    gradient_average_iterations_amount = 0 #5 #for FW
    # binarization_type = IndicatorBinarizationType.greedy_stepwise  # IndicatorBinarizationType.simulated_greedy_stepwise #greedy_stepwise #TopK
    series_binarization_func = IndicatorDistributionBinarizer.from_indicators_series_to_binary_indicator

    full_graph_perturbation_analysis_max_ratio = 2
    # endregion

    experiment_header = "nodes_number_analysis"  # "diameter_analysis"

    maxiter = 1_500

    embedding_types = [EmbeddingNetworkType.Moments, EmbeddingNetworkType.Spectral, EmbeddingNetworkType.GraphCNN]
    node_classifier_network_types = [NodeClassifierNetworkType.NN, NodeClassifierNetworkType.GCN]
    last_layer_types = [NodeClassifierLastLayerType.SquaredNormalized]
    binarization_types = [IndicatorBinarizationType.greedy_stepwise, IndicatorBinarizationType.simulated_greedy_stepwise,
                                          IndicatorBinarizationType.TopK]

    experiment_id = 0
    for embedding_type in embedding_types:
        for node_classifier_network_type in node_classifier_network_types:
            for last_layer_type in last_layer_types:
                for binarization_type in binarization_types:

                    experiment_id += 1

                    composite_solver, node_classifier_network_factory, sub_graph, params = setup_experiment()

                    #region analysis
                    subgraph_instance_name = loader_params['g_sub_path']
                    experiment_config_map = {'moment_type': moment_type, 'embedding_type': embedding_type,
                                             'graphcnn_hidden_dim': graphcnn_hidden_dim,
                                             'node_classifier_network_type': node_classifier_network_type,
                                             'last_layer_type': last_layer_type,
                                             'solver_type': solver_type,
                                             'greedy_search_node_classifier_best_quantile': greedy_search_node_classifier_best_quantile,
                                             'greedy_search_node_classifier_worst_quantile': greedy_search_node_classifier_worst_quantile,
                                             'gradient_average_iterations_amount': gradient_average_iterations_amount,
                                             'binarization_type': binarization_type,
                                             'full_graph_perturbation_analysis_max_ratio': full_graph_perturbation_analysis_max_ratio,
                                             'is_use_features': is_use_features,
                                             'to_line': to_line,
                                             'subgraph_instance_name': subgraph_instance_name,
                                             }
                    experiment_results_path = set_experiment_folder(experiment_header, experiment_id, experiment_config_map)

                    # full graph perturbation analysis experiment
                    seed = 10  # for plotting
                    plot_services = PlotServices(seed)

                    # inference_score_analyzer = DiameterVsSubgraphDetectionAnalysis(composite_solver, node_classifier_network_factory, params, plot_services, binarization_type, series_binarization_func, use_full_graph_edges=True, dump_path=experiment_results_path)
                    # perturbation_analysis_results = inference_score_analyzer.run(g_full=sub_graph.G, g_sub=sub_graph.G_sub, max_diameter=nx.diameter(sub_graph.G) * full_graph_perturbation_analysis_max_ratio)

                    inference_score_analyzer = NodesNumberVsSubgraphDetectionAnalysis(composite_solver,
                                                                                      node_classifier_network_factory, params,
                                                                                      plot_services, binarization_type,
                                                                                      series_binarization_func,
                                                                                      use_full_graph_edges=True,
                                                                                      dump_path=experiment_results_path)
                    perturbation_analysis_results = inference_score_analyzer.run(g_full=sub_graph.G, g_sub=sub_graph.G_sub,
                                                                                 max_n_nodes=len(
                                                                                     sub_graph.G) * full_graph_perturbation_analysis_max_ratio)

                    # for debug: perturbation_analysis_results[?]['binarized_solution']

                    # plot_analysis_scores(perturbation_analysis_results, 'f_beta_score', "graph nodes number")
                    plot_analysis_scores(perturbation_analysis_results, 'is_connected', "graph nodes number")
                    plot_analysis_scores(perturbation_analysis_results, 'correctly_captured_nodes_number', "graph nodes number")
                    plot_analysis_scores(perturbation_analysis_results, 'overlap_cdf_score', "graph nodes number")
                    plot_analysis_scores(perturbation_analysis_results, 'hausdorff_relative_distance', "graph nodes number")
                    plot_analysis_scores(perturbation_analysis_results, 'hausdorff_relative_distance_cdf_score', "graph nodes number")

                    #endregion