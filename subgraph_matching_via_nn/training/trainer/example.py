import os

import torch
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.data.data_loaders import load_graph
from subgraph_matching_via_nn.data.paths import DATA_PATH, COMP1_FULL_path, \
    COMP1_SUB0_path
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    MomentEmbeddingNetwork
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import EmbeddingMetricNetwork
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import \
    MLPGraphMetricNetwork
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.training.trainer.SimilarityMetricTrainer import SimilarityMetricTrainer

if __name__ == "__main__":
    print("Train example")

    solver_params = {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "lr": 0.0002, "weight_decay": 1e-3,
        "max_epochs": None,
        "cycle_patience": 5, "step_size_up": 10, "step_size_down": 10,
        "loss_convergence_threshold": None,
        "train_loss_convergence_threshold": 1e-1,
        "successive_convergence_min_iterations_amount": 50,
        "margin_loss_margin_value": 100,
        "max_grad_norm": 0.1
    }

    input_dim = 1
    problem_params = {"input_dim": input_dim}

    dump_base_path = f".{os.sep}runlogs"

    # from SubgraphMatching.GraphSimilarityModule.NeuroSEDGraphSimilarityModule import \
    #     NeuroSEDGraphSimilarityModule
    # from SubgraphMatching.Data.dataset_loader import DatasetLoaderUtils
    # from SubgraphMatching.Data.PairSampleEditDistanceInfo import \
    #     Pair_Sample_Edit_Distance_Info
    # from SubgraphMatching.Data.PairSampleEditDistanceInfoToPairSampleInfoConvertor import \
    #     PairSampleEditDistanceInfoToPairSampleInfoConvertor
    #
    # neuro_sed_sim_metric = NeuroSEDGraphSimilarityModule(device=solver_params["device"],
    #                                                      input_dim=problem_params[
    #                                                          "input_dim"])
    loss_fun = torch.nn.MSELoss()
    embedding_metric_network = EmbeddingMetricNetwork(loss_fun=loss_fun)
    moment_embedding_nn = MomentEmbeddingNetwork(n_moments=6,
                                                 moments_type='standardized_raw')

    embedding_nns = \
        [
            moment_embedding_nn,
        ]
    graph_metric_nn = MLPGraphMetricNetwork(embedding_networks=embedding_nns,
                                            embdding_metric_network=embedding_metric_network)
    trainer = SimilarityMetricTrainer(graph_metric_nn, dump_base_path,
                                      problem_params, solver_params)
    #
    # # region use NeuroSed dataset
    # NAME = "AIDS"
    # use_line_graphs = True
    # train_samples_list, val_samples_list = DatasetLoaderUtils.load_data_set(NAME,
    #                                                                         use_line_graphs=use_line_graphs,
    #                                                                         train_sample_size=100,
    #                                                                         val_sample_size=100)
    print("loaded dataset")

    # def convert_pair_sample_edit_distance_info_to_sample_api(
    #         sample: Pair_Sample_Edit_Distance_Info):
    #     # for converting NeuroSed pairs into boolean labels pairs
    #     dissimilarity_threshold = solver_params['margin_loss_margin_value']
    #     pair_sample_edit_distance_info_to_pair_sample_info_convertor = PairSampleEditDistanceInfoToPairSampleInfoConvertor(
    #         dissimilarity_threshold)
    #     return pair_sample_edit_distance_info_to_pair_sample_info_convertor.convert(
    #         sample)
    #
    #
    # def convert_pair_sample_edit_distance_info_list_to_sample_api_list(
    #         sample_list: List[Pair_Sample_Edit_Distance_Info]):
    #     converted_samples_list = []
    #     for edit_distance_sample in sample_list:
    #         converted_sample = convert_pair_sample_edit_distance_info_to_sample_api(
    #             edit_distance_sample)
    #         converted_samples_list.append(converted_sample)
    #     return converted_samples_list

    # train_samples_list = convert_pair_sample_edit_distance_info_list_to_sample_api_list(
    #     train_samples_list)
    # val_samples_list = convert_pair_sample_edit_distance_info_list_to_sample_api_list(
    #     val_samples_list)
    # # endregion

    num_training_examples = 10
    train_samples_list = []
    val_samples_list = []
    for i in range(num_training_examples):
        # Set the size of the graph and the subgraph
        n = 10  # Number of nodes in the graph (for random graph)
        m = 7  # Number of nodes in the subgraph (for random graph)
        seed = 10  # for plotting
        loader_params = {'graph_size': n,
                         'subgraph_size': m,
                         'data_path': DATA_PATH,
                         'g_full_path': COMP1_FULL_path,
                         'g_sub_path': COMP1_SUB0_path}

        sub_graph1 = \
            load_graph(type='random',
                       loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'
        sub_graph2 = \
            load_graph(type='random',
                       loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'
        G1_annotated = AnnotatedGraph(sub_graph1.G)
        G2_annotated = AnnotatedGraph(sub_graph2.G)
        train_samples_list.append(Pair_Sample_Info(
            subgraph=G1_annotated,
            masked_graph=G2_annotated,
            is_negative_sample=torch.tensor(True)))
        val_samples_list.append(Pair_Sample_Info(
            subgraph=G1_annotated,
            masked_graph=G2_annotated,
            is_negative_sample=torch.tensor(True)))

    trainer.train(train_samples_list, val_samples_list)
