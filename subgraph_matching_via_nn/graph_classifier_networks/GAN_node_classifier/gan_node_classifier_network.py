from typing import List
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from common.graph_utils import SubGraphGenerator
from common.logger import TimeLogging
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.dataset import gan_dataset
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.discriminator_model import Discriminator
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.gan_trainer import GANTrainer
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.generator_model import Generator
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import BaseNodeClassifierNetwork
from subgraph_matching_via_nn.graph_generators.graph_generators import BaseGraphGenerator
from subgraph_matching_via_nn.utils.graph_utils import get_node_indicator


class GANNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, noise_dim, classification_layer, device):
        super().__init__(classification_layer=classification_layer, input_dim=None, device=device)
        self.latent_dim = noise_dim

        weights_tensor = torch.zeros((1, noise_dim), device=self.device)
        self.noise_input = nn.Parameter(weights_tensor)

        self.classification_layer = classification_layer
        self.init_params()

        self.n_reference_subgraph_nodes = None
        self.n_reference_subgraph_edges = None
        self.constrained_subgraphs_generator = None

        # Defining training parameters
        self.batch_size = 512
        self.num_epochs = 500
        self.lr = 0.0002

    def train_node_classifier(self,
                              G: nx.graph = None,
                              G_sub: nx.graph = None,
                              graph_generator: BaseGraphGenerator = None):
        self.n_reference_subgraph_nodes = len(G_sub.nodes())
        self.n_reference_subgraph_edges = len(G_sub.edges())

        node_indicators = self.__generate_samples(G)
        self.constrained_subgraphs_generator = self.__train_gan_model(node_indicators)
        self.constrained_subgraphs_generator.eval() #the only  trainableparams should be the noise (input vector) params

        return node_indicators #for debug/evaluation pruposes

    def __train_gan_model(self, node_indicators: List[np.array]):
        # MODEL INITIALIZATION
        num_features = len(node_indicators[0].reshape(-1))

        generator = Generator(self.latent_dim, num_features, device=self.device)
        discriminator = Discriminator(num_features, device=self.device)

        # Create a dataloader of the dataset

        node_indicators_df = pd.DataFrame(node_indicators)
        dataset = gan_dataset(node_indicators_df)

        dataset_size = len(dataset)
        batch_size = min(dataset_size, self.batch_size)
        print(f"Training {type(self)} with batch size of {batch_size} for dataset of size {dataset_size}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # train
        dtype = self.noise_input.dtype
        GANTrainer.train_gan(discriminator, generator, dataloader, self.num_epochs, self.lr, dtype, self.device)

        return generator

    def forward(self, A, x=None, params: dict = None):
        if x is None:
            x = self.noise_input
        else:
            x = x.to(device=self.device)
            self.noise_input.data = x.data
            x = self.noise_input

        x = self.constrained_subgraphs_generator(x)

        w = self.classification_layer(A, x)

        return w

    def init_params(self, default_weights=None):
        with torch.no_grad():
            if default_weights is None:
                self.noise_input.data = torch.normal(0, 1, (1, self.latent_dim), dtype=self.noise_input.dtype, device=self.device, requires_grad=False)
            else:
                self.noise_input.data = default_weights

            self.classification_layer.init_weights()

    def __generate_samples(self, full_graph):
        # generate k subgraphs
        G_perturbed = full_graph.copy()

        source_graph = G_perturbed
        print("starting generating subgraphs")
        curr_time = TimeLogging.log_time(None, "start generate_k_subgraphs")

        k_subgraphs, k_subgraphs_original_nodes = SubGraphGenerator.generate_k_subgraphs(source_graph, k=self.n_reference_subgraph_nodes, is_parallel=True)

        curr_time = TimeLogging.log_time(curr_time, "end generate_k_subgraphs")

        # filter subgraphs by edges number constraint
        filtered_k_subgraphs_with_original_nodes = [(k_subgraph, original_nodes)
                                                    for k_subgraph, original_nodes in zip(k_subgraphs, k_subgraphs_original_nodes)
                                                    if len(k_subgraph.edges) == self.n_reference_subgraph_edges]

        print(f"#Connected k={self.n_reference_subgraph_nodes} nodes subgraphs = {len(k_subgraphs)},"
              f" out of which m={self.n_reference_subgraph_edges} edges subgraphs={len(filtered_k_subgraphs_with_original_nodes)}")

        # convert subgraph to node indicator
        node_indicators = [get_node_indicator(G=full_graph, G_sub=subgraph_example.subgraph(original_subgraph_nodes))
                           for subgraph_example, original_subgraph_nodes in filtered_k_subgraphs_with_original_nodes]

        return node_indicators