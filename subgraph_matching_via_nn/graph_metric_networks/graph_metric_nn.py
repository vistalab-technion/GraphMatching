from inspect import stack
from typing import List

import torch
from torch import nn, Tensor

from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph, GraphConstants
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    BaseGraphEmbeddingNetwork
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import \
    BaseEmbeddingMetricNetwork
from subgraph_matching_via_nn.training.mlp import MLP


class BaseGraphMetricNetwork(nn.Module):
    def __init__(self,
                 embedding_networks: List[BaseGraphEmbeddingNetwork],
                 embdding_metric_network: BaseEmbeddingMetricNetwork):
        super().__init__()
        self.embedding_networks = embedding_networks
        self.embdding_metric_network = embdding_metric_network

    def get_params_list(self) -> List[Tensor]:
        return list(self.parameters())


class MLPGraphMetricNetwork(BaseGraphMetricNetwork):
    def __init__(self,
                 embedding_networks: List[BaseGraphEmbeddingNetwork],
                 embdding_metric_network: BaseEmbeddingMetricNetwork):
        super().__init__(embedding_networks=embedding_networks,
                         embdding_metric_network=embdding_metric_network)

        input_dim = sum(
            [embedding_network.output_dim for embedding_network in embedding_networks])
        self.mlp = MLP(num_layers=2,
                       input_dim=input_dim,
                       hidden_dim=10,
                       output_dim=10,
                       device='cpu')

    def forward(self, G1: AnnotatedGraph, G2: AnnotatedGraph):
        loss = 0.0
        cat_embedding1 = []
        cat_embedding2 = []
        for embedding_network in self.embedding_networks:
            embedding1 = embedding_network(G1.adjacency_matrix_torch,
                                           G1.node_indicator)
            embedding2 = embedding_network(G2.adjacency_matrix_torch,
                                           G2.node_indicator)
            cat_embedding1.append(embedding1)
            cat_embedding2.append(embedding2)

            # loss += self.embdding_metric_network(embedding1, embedding2)
        embedding1_mlp = self.mlp(torch.cat(cat_embedding1).repeat(2,1))
        embedding2_mlp = self.mlp(torch.cat(cat_embedding1).repeat(2,1))
        loss = self.embdding_metric_network(embedding1_mlp, embedding2_mlp)
        # loss = loss + self.mlp(torch.ones(2, 2)).sum()
        return loss
