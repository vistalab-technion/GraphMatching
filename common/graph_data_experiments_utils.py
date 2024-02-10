from typing import List, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score
import cmath
import math
import pathlib
import os
import sys
import numpy as np
import torch
from torch import optim
from common.graph_utils import SubGraphGenerator
from subgraph_matching_via_nn.data.data_loaders import load_graph
from common.logger import TimeLogging
from powerful_gnns.util import S2VGraph, load_data_given_graph_list_and_label_map
from powerful_gnns.models.graphcnn import GraphCNN
from common.EmbeddingCalculationsService import pairwise_l2_distance, show_distance_matrix, \
    calculate_energy_based_hidden_rep
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import MLPGraphMetricNetwork
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import EmbeddingMetricNetwork
from subgraph_matching_via_nn.graph_embedding_networks.gnn_embedding_network import GNNEmbeddingNetwork
from subgraph_matching_via_nn.training.trainer.S2VGraphEmbeddingSimilarityMetricTrainer import \
    S2VGraphEmbeddingSimilarityMetricTrainer
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import SingleEmbeddingGraphMetricNetwork
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import S2VGraphEmbeddingGraphMetricNetwork
from networkx import NetworkXError
import itertools
import networkx as nx
from common.parallel_computation import tqdm_joblib
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from os import cpu_count
import pickle
from common.graph_utils import GED_graph_generator
from subgraph_matching_via_nn.training.MarginLoss import MarginLoss
import networkx as nx
from igraph import Graph
from scipy.stats import norm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
import time
from common.wmds.wmds_r_wrapper import apply_wmds


def create_graph_metric_net(model_factory_func, device):
    model = model_factory_func(device=device)

    loss_fun = torch.nn.MSELoss()
    embedding_metric_network = EmbeddingMetricNetwork(loss_fun=loss_fun)

    gnn_embedding_nn = GNNEmbeddingNetwork(gnn_model=model)

    # embedding_nns = \
    #     [
    #         GNNEmbeddingNetwork(gnn_model=model),
    #     ]

    graph_metric_nn = S2VGraphEmbeddingGraphMetricNetwork(embedding_network=gnn_embedding_nn,
                                           embdding_metric_network=embedding_metric_network,
                                           device=device)

    # graph_metric_nn = SingleEmbeddingGraphMetricNetwork(embedding_network=embedding_nns[0],
    #                                        embdding_metric_network=embedding_metric_network,
    #                                        device=device)

    # graph_metric_nn = MLPGraphMetricNetwork(embedding_networks=embedding_nns,
    #                                         embdding_metric_network=embedding_metric_network,
    #                                         device=device)

    return graph_metric_nn, model

def init_embedding_net_and_trainer(model_factory_func, solver_params, problem_params, dump_base_path = f".{os.sep}runlogs", graph_metric_nn_checkpoint_path=None):

    # must start on CPU to allow moving model to GPU, due to existing pytorch bug
    device = 'cpu'
    graph_metric_nn, model = create_graph_metric_net(model_factory_func, device)

    if graph_metric_nn_checkpoint_path is not None:
        graph_metric_nn.load_state_dict(torch.load(graph_metric_nn_checkpoint_path, map_location=device))
        model = graph_metric_nn.embedding_networks[0].gnn_model

    if solver_params['is_use_model_compliation']:
        graph_metric_nn = torch.compile(graph_metric_nn)
        model = graph_metric_nn.embedding_networks[0].gnn_model

    trainer = S2VGraphEmbeddingSimilarityMetricTrainer(graph_metric_nn, dump_base_path,
                                      problem_params, solver_params)

    return trainer, graph_metric_nn, model

def get_graph_wl_distances(graphs, node_label):
    from wwl import laplacian_kernel
    
    import wwl # import inside to avoid too many import calls to these modules while performing parallel work with many processes, ending up with a RuntimeError: CUDA error: out of memory

    igraphs = [Graph() for i in range(len(graphs))]
    igraphs = [igraph.from_networkx(graph) for igraph, graph in zip(igraphs, graphs)]
    kernel_values = wwl.pairwise_wasserstein_distance(igraphs, [len(igraphs)-1], enforce_continuous=True, num_iterations=10)
    kernel_values = laplacian_kernel(kernel_values) #normalize energey

    # gk_wl = WL_metric.GK_WL()
    # kernel_values = gk_wl.compare_list(graphs, h=5, node_label=node_label)
    return 1- kernel_values
    #return np.round(1 - kernel_values, decimals=4)

def transform_into_s2vgraphs(graphs, labels, device):
    g_list = []
    label_dict = {}
    for subgraph, label in zip(graphs, labels):
        if type(subgraph) is nx.Graph:
            s2v_graph = S2VGraph(subgraph, label)
        else:
            s2v_graph = S2VGraph(subgraph.G, label)
        g_list.append(s2v_graph)
        if not label in label_dict:
            mapped = len(label_dict)
            label_dict[label] = mapped

    # Process graph features
    graphs, num_classes = load_data_given_graph_list_and_label_map(g_list, label_dict, degree_as_tag=True, device=device)

    return graphs, num_classes

def generate_s2v_graphs(networkx_graphs, device, print_stats=True):
    graphs, _ = transform_into_s2vgraphs(networkx_graphs, [None for graph in networkx_graphs], device)

    for s2v_graph in graphs:
        # convert graph features here, not only in trainer! (heatmap is probably wrong)
        annotated_graph = AnnotatedGraph(s2v_graph.g)
        s2v_graph.node_features = annotated_graph.node_indicator.to(device=device)

    return graphs

def generate_k_subgraph(reference_subgraph, k):
    full_graph = reference_subgraph.G
    
    subgraphs_iterator = itertools.combinations(full_graph, k)
    k_subgraphs = [full_graph.subgraph(s) for s in subgraphs_iterator]
    k_subgraph_annotated_graphs = [AnnotatedGraph(g, label=i) for i, g in enumerate(k_subgraphs) if len(g.edges) >0]

    return k_subgraph_annotated_graphs

def plot_map(map, xlabel, ylabel, title):
    keys = list(map.keys())
    values = list(map.values())
      
    fig = plt.figure(figsize = (15, 5))

    # creating the bar plot
    plt.plot(keys, values, color ='maroon')

    extraticks = keys
    xticks = list(extraticks)
    plt.xticks(xticks)

    plt.xlabel(xlabel)
    ax = matplotlib.pyplot.gca()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def chart_bar_plot(data_map, xlabel, ylabel, title):
    keys = list(data_map.keys())
    values = list(data_map.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(keys, values, color ='maroon',
            width = 0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_histogram(sequence, x_title="", min_range=None, max_range=None):
    # Fit a distribution to the data
    if min_range is None:
        min_range = min(sequence)
    if max_range is None:
        max_range = max(sequence)

    print(min_range)
    print(max_range)
    sequence = [item for item in sequence if ((item >= min_range) and (item <= max_range))]

    fig, ax = plt.subplots(1)
    # sns.histplot(sequence, bins='auto', stat='density',
    #              label='Normalized Histogram', ax=ax)

    mu, std = norm.fit(sequence)
    x = np.linspace(min_range, max_range, 100)
    if not np.all(x == x[0]):
        y = norm.pdf(x, mu, std)
    else:
        y = np.ones_like(x)
    sns.lineplot(x=x, y=y, color='red', label='Fitted Normal Distribution', ax=ax)

    # Plot the KDE
    sns.kdeplot(sequence, label='KDE', bw_adjust=0.5, ax=ax)

    ax.set_xlabel(x_title)
    ax.set_ylabel('Density')
    ax.set_title(f'{x_title} Distribution')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.show()

# go over pairs in ged_examples1/2, and for their subgraphs generate wl+dist
# only take into account their wl_dist from the all pairs vs all pairs matrix
def generate_wl_dist_for_ged_pairs(ged_examples_list):
    pairs_index_to_dist_map = {}

    all_ged_example_list_graphs = []
    for ged_example_list in ged_examples_list:
        ged_example_list_graphs = [ged_example_list[0].subgraph.g] + [pair.masked_graph.g for pair in ged_example_list]
        all_ged_example_list_graphs += ged_example_list_graphs

    wl_dist_matrix = get_graph_wl_distances(all_ged_example_list_graphs, node_label=False)

    base_graph_index = 0
    pairs_index = 0
    for ged_example_list_index, ged_example_list in enumerate(ged_examples_list):
        for relative_index, pair in enumerate(ged_example_list):
            wl_dist = wl_dist_matrix[base_graph_index][base_graph_index + relative_index + 1]
            pairs_index_to_dist_map[pairs_index] = wl_dist

            pairs_index += 1
        base_graph_index += 1 + (relative_index + 1)

    return pairs_index_to_dist_map

def show_wl_dist_histograms(positive_examples, negative_examples, max_examples_amount = 1000):
    positive_examples_pairs_index_to_dist_map = {}
    negative_examples_pairs_index_to_dist_map = {}

    ged_examples_pairs_index_to_dist_map = generate_wl_dist_for_ged_pairs(positive_examples[:max_examples_amount] + negative_examples[:max_examples_amount])
    print("finished distances calculation")

    for pair_index, wl_dist in ged_examples_pairs_index_to_dist_map.items():
        if pair_index < len(ged_1_examples):
            positive_examples_pairs_index_to_dist_map[pair_index] = wl_dist
        else:
            negative_examples_pairs_index_to_dist_map[pair_index] = wl_dist
    print("plotting")

    plot_histogram(list(positive_examples_pairs_index_to_dist_map.values()), "positive examples WL distance")
    plot_histogram(list(negative_examples_pairs_index_to_dist_map.values()), "negative examples WL distance")

    print(max(positive_examples_pairs_index_to_dist_map.values()))
    print(min(negative_examples_pairs_index_to_dist_map.values()))

    return positive_examples_pairs_index_to_dist_map, negative_examples_pairs_index_to_dist_map

def get_min_non_diagonal_entry(tensor_, device, single_reference_graph_index=None, show_min_off_diagonal=True):
    # show_min_off_diagonal: True for min, False for max
    shape = tensor_.shape
    assert len(shape) == 2

    is_1_vs_all = False

    if shape[0] != shape[1]:
        assert ((shape[1] == 1) and (single_reference_graph_index is not None))
        is_1_vs_all = True

    diagonal_stub_value = float("inf")
    if not show_min_off_diagonal:
        diagonal_stub_value = -diagonal_stub_value

    if is_1_vs_all:
        tensor_with_inf_diag = tensor_.detach().clone().requires_grad_(False)
        tensor_with_inf_diag[single_reference_graph_index] = diagonal_stub_value
    else:
        tensor_with_inf_diag = tensor_ + torch.diag_embed(torch.ones(tensor_.shape[0], device=device) * diagonal_stub_value)

    if show_min_off_diagonal:
        return torch.min(tensor_with_inf_diag).item()
    else:
        return torch.max(tensor_with_inf_diag).item()

def show_distances_heatmap(graphs, model, device, show_min_off_diagonal: bool = True):
    # show_min_off_diagonal: True for min, False for max

    model.eval()
    all_embeddings = model.get_embedding(graphs)

    l2_dists = pairwise_l2_distance(all_embeddings)

    rounding_constant = 10 ** 3
    cos_dists = torch.round(calculate_energy_based_hidden_rep(all_embeddings, threshold=-1) * rounding_constant) / rounding_constant

    show_distance_matrix(l2_dists, "l2-distances")
    show_distance_matrix(cos_dists, "cosine-distances")

    print(f"Off matrix diagonal margin: {get_min_non_diagonal_entry(l2_dists, device, show_min_off_diagonal=show_min_off_diagonal)}")
    print(f"Off matrix diagonal margin: {get_min_non_diagonal_entry(cos_dists, device, show_min_off_diagonal=show_min_off_diagonal)}")

def get_examples_distances(trainer, graph_metric_nn, samples):

    train_loaders, val_loaders = trainer.get_data_loaders(samples, [], new_samples_amount=0, device_ids=[0])

    positive_distances = []
    negative_distances = []

    # max_pair_batches_to_test = 10

    i = 0

    for pairs_batch in train_loaders[0]:
        # if i >= max_pair_batches_to_test:
        #     break
        i += 1

        with torch.no_grad():
            distances = graph_metric_nn.forward(
                [sample.s2v_graphs for sample in pairs_batch]
            )

        for pair, distance in zip(pairs_batch, distances):
            distance = distance.item()
            is_negative_example = pair.pair_sample_info.is_negative_sample
            # print(f"{i} : {distance}")

            if is_negative_example==False:
                positive_distances.append(distance)
                # if distance > solver_params['margin_loss_margin_value']:
                #     print(f"positive loss term not zero, as distance for this positive example pair is {distance}")
            else:
                negative_distances.append(distance)
                # if distance < solver_params['margin_loss_margin_value']:
                #     print(f"negative loss term not zero, as distance for this negative example pair is {distance}")
    return positive_distances, negative_distances

def calc_margin_loss(distances, pos_labels, margin):
    negative_labels = 1 - pos_labels
    lossCriterion = MarginLoss(margin)
    pos_loss, neg_loss = lossCriterion.get_loss(distances, pos_labels, negative_labels)
    print(pos_loss.sum().item())
    print(neg_loss.sum().item())

def generate_perturbed_graphs(reference_graph: nx.Graph):
    # assumption: there are no self loops, and no duplicate edges of opposite directions

    # graphs generated by removing an existing edge
    for edge in reference_graph.edges:
        copy_graph = reference_graph.copy()
        copy_graph.remove_edge(*edge)
        yield copy_graph

    # graphs generated by adding a missing edge
    sorted_nodes = sorted(reference_graph.nodes, reverse=False)
    for i, node_i in enumerate(sorted_nodes):
        if i == len(sorted_nodes) - 1:
            break

        for node_j in sorted_nodes[i+1:]:
            edge = (node_i, node_j)
            if reference_graph.has_edge(*edge):
                continue
            copy_graph = reference_graph.copy()
            copy_graph.add_edge(*edge)
            yield copy_graph

def generate_pair_example(G1_annotated, G2_annotated, is_negative_example):
    return Pair_Sample_Info(
        subgraph=G1_annotated,
        masked_graph=G2_annotated,
        is_negative_sample=torch.tensor(is_negative_example))

def generate_positive_and_negative_ged_pairs(pos_annotated_subgraphs, pos_ged_dist, pos_pairs_n, neg_annotated_subgraphs, neg_ged_dist, neg_pairs_n):
    with tqdm_joblib(tqdm(desc="My calculation", total=len(neg_annotated_subgraphs))) as progress_bar:
        ged_2_examples = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
            delayed(generate_random_ged_paris)(annotated_subgraph=annotated_subgraph, ged_dist=neg_ged_dist, pairs_n=neg_pairs_n, is_negative_example=True)
            for annotated_subgraph in neg_annotated_subgraphs
        )
    
    with tqdm_joblib(tqdm(desc="My calculation", total=len(pos_annotated_subgraphs))) as progress_bar:
        ged_1_examples = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
            delayed(generate_random_ged_paris)(annotated_subgraph=annotated_subgraph, ged_dist=pos_ged_dist, pairs_n=pos_pairs_n, is_negative_example=False)
            for annotated_subgraph in pos_annotated_subgraphs
        )
    
    ged_examples1 = [elem for lst in ged_1_examples for elem in lst]
    ged_examples2 = [elem for lst in ged_2_examples for elem in lst]

    return ged_examples1, ged_examples2

def compare_graphs_and_generate_pair_example(G1_annotated, G2_annotated, isomorphic_pairs):
    SG1 = G1_annotated.g
    SG2 = G2_annotated.g

    if nx.is_isomorphic(SG1, SG2):
        isomorphic_pairs.append((SG1, SG2))
        return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = False)
    else:
        return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = True)

    # try:
    #     diff_graph = nx.symmetric_difference(SG1, SG2)
    # except NetworkXError:
    #     # node sets are different
    #     if nx.is_empty(SG1) and nx.is_empty(SG2) and (len(SG1) == len(SG2)):
    #         # positive example
    #         print("node sets are different, but isomorphic graphs")
    #         return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = False)
    #     else:
    #         # negative example
    #         print("node sets are different, and not isomorphic graphs")
    #         return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = True)

    # if nx.is_empty(diff_graph):
    #     # print("isomorphic graphs")
    #     isomorphic_pairs.append((SG1, SG2))
    #     # positive example
    #     return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = False)
    # else:
    #     # negative example
    #     return generate_pair_example(G1_annotated, G2_annotated, is_negative_example = True)

def remove_isolated_nodes_from_graph(graph:nx.Graph):
    isolated_nodes_indices = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes_indices)

def create_k_subgraphs_for_circuit(circuit_base_dir, circuit_file_name, is_parallel):
    g_file_rel_path = 'full_graph.p'
    g_sub_file_rel_path = 'subgraph0.p'

    circuit_dir = f"{circuit_base_dir}{circuit_file_name}{os.sep}"
    loader_params = {
     'data_path' : str(circuit_dir),
     'g_full_path': g_file_rel_path,
     'g_sub_path': g_sub_file_rel_path}
    
    sub_graph = \
        load_graph(type='subcircuit',
                   loader_params=loader_params)

    G_sub = sub_graph.G_sub
    k = len(G_sub)
    n = len(sub_graph.G)
    candidate_nodes_to_remove_from_full_graph = list(set(sub_graph.G.nodes).difference(G_sub.nodes))
    G_perturbed = sub_graph.G.copy()
    n_nodes_to_remove_from_full_graph = len(candidate_nodes_to_remove_from_full_graph) // 2
    G_perturbed.remove_nodes_from(candidate_nodes_to_remove_from_full_graph[:n_nodes_to_remove_from_full_graph])

    remove_isolated_nodes_from_graph(G_perturbed)
    
    print(f"full graph has {n} nodes, subgraph has {k} nodes, removing {n - len(G_perturbed)} non subgraph nodes from full graph")

    source_graph = G_perturbed
    print("starting generating subgraphs")
    curr_time = TimeLogging.log_time(None, "start generate_k_subgraphs")

    
    k_subgraphs = SubGraphGenerator.generate_k_subgraphs(source_graph, k=k, is_parallel=is_parallel)
    k_subgraph_annotated_graphs = [AnnotatedGraph(g, label=i) for i, g in enumerate(k_subgraphs)]
    
    curr_time = TimeLogging.log_time(curr_time, "end generate_k_subgraphs")

    return k_subgraph_annotated_graphs, G_perturbed, G_sub

def compare_graphs_and_generate_pair_example_against_base_graph(k_subgraph_annotated_graphs_file_path, base_graph_indices, chunk_index, output_folder_base_path):
    curr_time = TimeLogging.log_time(None, "enter loading k-subgraphs file")
    with open(k_subgraph_annotated_graphs_file_path, 'rb') as file:
        k_subgraph_annotated_graphs = pickle.load(file)
    curr_time = TimeLogging.log_time(curr_time, "finished loading k-subgraphs file")
    
    train_samples_list = []

    for base_graph_index in range(base_graph_indices[0], base_graph_indices[1]):
        isomorphic_pairs = []
        SG_annotated_1 = k_subgraph_annotated_graphs[base_graph_index]
    
        train_samples_list += [compare_graphs_and_generate_pair_example(SG_annotated_1, k_subgraph_annotated_graphs[subgraph_counter2], isomorphic_pairs) 
                              for subgraph_counter2 in range(base_graph_index+1, len(k_subgraph_annotated_graphs))]
        # print(f"#isomorphic_pairs for graph #{base_graph_index} is {len(isomorphic_pairs)}")
    
    with open(f"{output_folder_base_path}/{chunk_index}.p", 'wb') as f:
        pickle.dump(train_samples_list, f)
    print(f"finished chunk #{chunk_index} of {len(train_samples_list)} pairs", flush=True)
    
    # return train_samples_list

def generate_pairs_data_set(circuit_base_dir, circuit_file_name, output_folder_base_path = "brute_force_pairs"):
    train_samples_list = []
    
    curr_time = TimeLogging.log_time(None, "enter generate_pairs_data_set")
    cpu_num = int(cpu_count())

    k_subgraph_annotated_graphs, G_perturbed, G_sub = create_k_subgraphs_for_circuit(circuit_base_dir, circuit_file_name, is_parallel=True)
    k_subgraph_annotated_graphs_file_path = f"k_subgraph_annotated_graphs.p"
    with open(k_subgraph_annotated_graphs_file_path, 'wb') as f:
        pickle.dump(k_subgraph_annotated_graphs, f)
    
    if not os.path.exists(output_folder_base_path):
        os.makedirs(output_folder_base_path)
    
    n = len(k_subgraph_annotated_graphs)
    chunks_amount = cpu_num
    chunk_size = int(math.ceil(n / chunks_amount))

    base_graph_chunks_indices = [(chunk_index * chunk_size, min(chunk_index * chunk_size + chunk_size, n)) for chunk_index in range(0, chunks_amount)]
    
    with tqdm_joblib(tqdm(desc="Pairs dataset construction", total=len(base_graph_chunks_indices))) as progress_bar:
        Parallel(n_jobs=cpu_num, backend='multiprocessing')(
                delayed(compare_graphs_and_generate_pair_example_against_base_graph)
                    (k_subgraph_annotated_graphs_file_path=k_subgraph_annotated_graphs_file_path, base_graph_indices=base_graph_chunk_indices, chunk_index=chunk_index, output_folder_base_path=output_folder_base_path)
                    for chunk_index, base_graph_chunk_indices in enumerate(base_graph_chunks_indices)
            )

    curr_time = TimeLogging.log_time(curr_time, "finished generate_pairs_data_set")
    # train_samples_list = [elem for lst in train_samples_list for elem in lst]
    
    return k_subgraph_annotated_graphs, G_perturbed, G_sub

def generate_random_ged_paris(annotated_subgraph: AnnotatedGraph, ged_dist: int, pairs_n: int, is_negative_example: bool, force_exactly_pairs_n:bool=True):
    train_samples_list = []
    SG1 = annotated_subgraph

    ged_graph_generator = GED_graph_generator(SG1.g, ged_dist)

    # generate pairs

    for i, ged_perturbed_graph in enumerate(ged_graph_generator.generate()):
        if i == pairs_n:
            break

        if len(ged_perturbed_graph.edges) == 0:
            # dont create empty graphs
            continue

        # positive example
        train_samples_list.append(generate_pair_example(SG1, AnnotatedGraph(ged_perturbed_graph), is_negative_example = is_negative_example))

    if force_exactly_pairs_n and (i < pairs_n):
        raise ValueError(f"Not enough possible instances to generate as prescribed ({pairs_n})")

    sys.stdout.flush()
    return train_samples_list

def train_model_with_hyperparams(used_train_samples_list, solver_params, problem_params, processes_device_ids, input_dim=1, num_classes=1, num_layers=5, num_mlp_layers=2, hidden_dim=128, trainer=None):

    last_checkpoint_model_path = None

    # val_samples_list = used_train_samples_list
    model_factory_func = lambda device: GraphCNN(num_layers=num_layers, num_mlp_layers = num_mlp_layers, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes, final_dropout=0.5, learn_eps=False, graph_pooling_type="sum", neighbor_pooling_type="sum", device=device)
    solver_params["batch_size"] = 512 * 8
    solver_params["k_update_plot"] = 10
    solver_params["lr"] = 1e-1
    solver_params["max_epochs"] = 200
    current_trainer, graph_metric_nn, model = init_embedding_net_and_trainer(model_factory_func, solver_params, problem_params, graph_metric_nn_checkpoint_path=last_checkpoint_model_path)

    if trainer is None:
        best_val_loss = current_trainer.train(processes_device_ids=processes_device_ids, use_existing_data_loaders=False, train_samples_list=used_train_samples_list, val_samples_list=[])
    else:
        current_trainer.set_existing_data_loader_paths(trainer.previous_train_loader_paths, trainer.previous_val_loader_paths)
        best_val_loss = current_trainer.train(processes_device_ids=processes_device_ids, use_existing_data_loaders=True)

    return current_trainer, graph_metric_nn, best_val_loss

def visualize_embeddings(graph_metric_nn, reference_subgraph, positive_ged_pairs, negative_ged_pairs, device):
    # compute GIN model distance for all n * (n-1) / 2 pairs, or at least part of them (leave empty if not directly induces GED pairs)

    # graph_labels = [-1] + [0 for pair in positive_ged_pairs] + [1 for pair in negative_ged_pairs]
    # marker_sizes = [10] + [1 for pair in positive_ged_pairs + negative_ged_pairs]

    all_embeddings = get_embeddings(graph_metric_nn, reference_subgraph, positive_ged_pairs, negative_ged_pairs, device)

    # apply pairwise loss
    l2_dists = pairwise_l2_distance(all_embeddings)

    # apply the MDS with 2d
    sim_matrix = l2_dists.detach().cpu().numpy()
    w_matrix = np.ones(l2_dists.shape)
    mds_embeddings = apply_wmds(sim_matrix, w_matrix, dim=2)
    print(f"plotting {len(mds_embeddings)} samples")

    # plot it (with 2 labels)
    plt.figure(figsize=(8, 8), dpi=80)

    n_samples = mds_embeddings.shape[0]
    alpha =  min (1 / n_samples * 100, 0.5)

    plt.scatter(mds_embeddings[1:len(positive_ged_pairs)+1, 0], mds_embeddings[1:len(positive_ged_pairs)+1, 1],
            c='green',
            alpha=alpha,
            s=5, marker='^')
    plt.scatter(mds_embeddings[len(positive_ged_pairs)+1:, 0], mds_embeddings[len(positive_ged_pairs)+1:, 1],
            c='red',
            alpha=alpha,
            s=5, marker='v')
    plt.scatter(mds_embeddings[0, 0], mds_embeddings[0, 1],
            c='black',
            alpha=1,
            s=45, marker='o')

    plt.grid()

    plt.title('Feature space')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.plot()
    plt.show()

def create_ged_visulaization_graphs(reference_graph, label_class_examples_amount=50, positive_GED=1, negative_GED=4):

    assert negative_GED > positive_GED

    negative_examples_k_subgraphs = positive_examples_k_subgraphs = [reference_graph]

    with tqdm_joblib(tqdm(desc="My calculation", total=len(negative_examples_k_subgraphs))) as progress_bar:
        negative_ged_examples = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
            delayed(generate_random_ged_paris)(annotated_subgraph=annotated_subgraph, ged_dist=negative_GED, pairs_n=label_class_examples_amount, is_negative_example=True)
            for annotated_subgraph in [reference_graph]
        )

    with tqdm_joblib(tqdm(desc="My calculation", total=len(positive_examples_k_subgraphs))) as progress_bar:
        positive_ged_examples = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
            delayed(generate_random_ged_paris)(annotated_subgraph=annotated_subgraph, ged_dist=positive_GED, pairs_n=label_class_examples_amount, is_negative_example=False)
            for annotated_subgraph in [reference_graph]
        )

    negative_ged_examples= [elem for lst in negative_ged_examples for elem in lst]
    positive_ged_examples = [elem for lst in positive_ged_examples for elem in lst]

    return negative_ged_examples, positive_ged_examples

def get_embeddings(graph_metric_nn, reference_subgraph, positive_ged_pairs, negative_ged_pairs, device):
    # collect all graphs
    all_graphs = [reference_subgraph] + [pair.masked_graph for pair in positive_ged_pairs] + [pair.masked_graph for pair in negative_ged_pairs]
    all_s2v_graphs = generate_s2v_graphs([graph.g for graph in all_graphs], device)

    # apply batching (work on multiple graphs to generate their embeddings
    _ = graph_metric_nn.eval()
    embedding_network = graph_metric_nn.embedding_networks[0]
    all_embeddings = embedding_network.forward_graphs(all_s2v_graphs)

    return all_embeddings

def get_model_dump_path(hidden_dim, model_checkpoints_base_folder = "./mp/"):
    return f"{model_checkpoints_base_folder}{os.sep}{hidden_dim}/best_model_state_dict.pt"

def visualize_embeddings_for_trained_model(ref_graph, positive_pairs, negative_pairs, solver_params, problem_params, input_dim, num_classes, device):
    for hidden_dim in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        print(hidden_dim)
        model_dump_path = get_model_dump_path(hidden_dim)

        model_factory_func = lambda device: GraphCNN(num_layers=5, num_mlp_layers = 2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes, final_dropout=0.5, learn_eps=False, graph_pooling_type="sum", neighbor_pooling_type="sum", device=device)

        _, graph_metric_nn, _ = init_embedding_net_and_trainer(model_factory_func, solver_params, problem_params, graph_metric_nn_checkpoint_path=model_dump_path)

        # calc embeddings of all pairs, apply MDS and plot
        visualize_embeddings(graph_metric_nn, ref_graph, positive_pairs, negative_pairs, device)

def measure_clustering(data, gt_labels, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    predicted_labels = kmeans.labels_


    # Compute silhouette score
    silhouette = silhouette_score(data, predicted_labels)

    # Compute Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(data, predicted_labels)

    # nmi
    nmi = normalized_mutual_info_score(predicted_labels, gt_labels)

    return {"sil":silhouette, "dav": davies_bouldin, "nmi":nmi}

def show_score_plot(scores_map, score_type, score_name):
    scores = []
    hid_dims = []
    for hid_dim, hid_dim_scores_map in scores_map.items():
        score = hid_dim_scores_map[score_type]

        hid_dims.append(hid_dim)
        scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(hid_dims, scores)

    ax.set(xlabel='hidden dimension value', ylabel='score',
           title=f'{score_name} clustering score')
    ax.grid()
    plt.show()

def generate_radial_3d_points(graph_metric_nn, trainer, pairs, radius):
    # for each radius, take all sampled GED pairs, and divide the circle into that amount of samples
    X = []
    losses = []

    positive_pairs_distances, negative_pairs_distances = get_examples_distances(trainer, graph_metric_nn, pairs)
    # if len(positive_pairs_distances) != 0:
    #     print("positive pairs:")
    #     print(positive_pairs_distances)
    #     print(max(positive_pairs_distances))
    # if len(negative_pairs_distances) != 0:
    #     print("negative pairs:")
    #     print(negative_pairs_distances)
    #     print(min(negative_pairs_distances))
    pairs_distances = positive_pairs_distances + negative_pairs_distances

    n_samples = len(pairs_distances)
    frequency_angle = 360 / n_samples

    for i, distance in enumerate(pairs_distances):
        z = cmath.rect(radius, math.radians(i * frequency_angle))
        X.append(np.array([z.real, z.imag]))

        losses.append(distance)

    return X, losses


def create_metric_and_trainer(hidden_dim, solver_params, problem_params, input_dim = 1, num_classes = 1):
    model_factory_func = lambda device: GraphCNN(num_layers=5, num_mlp_layers = 2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes, final_dropout=0.5, learn_eps=False, graph_pooling_type="sum", neighbor_pooling_type="sum", device=device)

    model_dump_path = get_model_dump_path(hidden_dim)
    trainer, graph_metric_nn, _ = init_embedding_net_and_trainer(model_factory_func, solver_params, problem_params, graph_metric_nn_checkpoint_path=model_dump_path)
    return trainer, graph_metric_nn

def visualize_reference_graph_model_loss_space(solver_params, problem_params, hidden_dim, ref_graph: AnnotatedGraph, GED_dists: Union[List[int], dict], max_samples_per_radius = 100):
    trainer, graph_metric_nn = create_metric_and_trainer(hidden_dim, solver_params, problem_params)
    _ = graph_metric_nn.eval()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ged_dist_to_pairs_map = {}
    if type(GED_dists) is dict:
        ged_dist_to_pairs_map = GED_dists
    else:
        for GED_dist in GED_dists:
            # generate pairs
            with tqdm_joblib(tqdm(desc="My calculation", total=1)) as progress_bar:
                pairs = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
                    delayed(generate_random_ged_paris)(annotated_subgraph=g, ged_dist=GED_dist, pairs_n=max_samples_per_radius, is_negative_example=False, force_exactly_pairs_n=False)
                    for g in [ref_graph]
            )
            pairs = [elem for lst in pairs for elem in lst]
            print(f"Generated {len(pairs)} GED-{GED_dist} pairs")
            ged_dist_to_pairs_map[GED_dist] = pairs

    for GED_dist, pairs in ged_dist_to_pairs_map.items():
        # calculate 3D visualization coordinates
        X, Z = generate_radial_3d_points(graph_metric_nn, trainer, pairs, radius=GED_dist)
        X = np.vstack(X)
        Z = np.vstack(Z)

        # Plot a 3D surface
        X = np.vstack(X)
        Z = np.vstack(Z)
        x = X[:, 0]
        y = X[:, 1]

        ax.scatter3D(x.reshape(-1), y.reshape(-1), Z, cmap='Greens')
    return ged_dist_to_pairs_map

def calc_margin_loss_for_pairs(trainer, graph_metric_nn, solver_params, pairs):
    train_positive_distances, train_negative_distances = get_examples_distances(trainer, graph_metric_nn, pairs)

    calc_margin_loss(torch.tensor(train_positive_distances+train_negative_distances), torch.cat((torch.ones(len(train_positive_distances)), torch.zeros(len(train_negative_distances)))), margin = solver_params['margin_loss_margin_value'])

    plot_histogram(train_positive_distances, "positive pair distances", min_range=0)
    plot_histogram(train_negative_distances, "negative pair distances", min_range=0)

    return train_positive_distances, train_negative_distances

# for ged, ged_pairs in multiple_ged_radiuses_graphs_map.items():
def plot_histogram_for_ged_pairs(trainer, graph_metric_nn, ged, ged_pairs, max_range=None):
    train_positive_distances, train_negative_distances = get_examples_distances(trainer, graph_metric_nn, ged_pairs)
    plot_histogram(train_positive_distances + train_negative_distances, f"{ged} pair distances", min_range=0, max_range=max_range)

def calc_performance_metrics(graph_metric_nn, samples, positive_distances, negative_distances, solver_params, device):
    FP = 0
    FN = 0

    single_pair_forward_positive_distances = []
    single_pair_forward_negative_distances = []
    for i in range(0, len(samples)):
        if i % 5_000 == 0:
            print(f"post iteration {i}")
        is_negative_sample = samples[i].is_negative_sample
        graph1 = samples[i].subgraph
        graph2 = samples[i].masked_graph

        s2v_graphs = generate_s2v_graphs([graph1.g, graph2.g], device, print_stats=False)

        distance = graph_metric_nn.forward([(s2v_graphs[0], s2v_graphs[1])]).item()

        if is_negative_sample:
            single_pair_forward_negative_distances.append(distance)
            if (distance < solver_params['margin_loss_margin_value']):
                FP += 1
                #print(f"negative loss term not zero, as distance for this negative example pair is {distance}")
        if (not is_negative_sample):
            single_pair_forward_positive_distances.append(distance)
            if (distance > solver_params['margin_loss_margin_value']):
                FN += 1
                #print(f"negative loss term not zero, as distance for this positive example pair is {distance}")
    print(f"FP={FP} out of {len(negative_distances)} negative examples")
    print(f"FN={FN} out of {len(positive_distances)} positive examples")

    return single_pair_forward_positive_distances, single_pair_forward_negative_distances