import abc
from abc import abstractmethod
from typing import List, TypeVar, Tuple
import torch
from torch import nn, Tensor

from powerful_gnns.util import S2VGraph
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph, GraphConstants
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    BaseGraphEmbeddingNetwork, GraphsBatchEmbeddingNetwork
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import \
    BaseEmbeddingMetricNetwork
from subgraph_matching_via_nn.training.mlp import MLP

T = TypeVar("T", S2VGraph, AnnotatedGraph)

class BaseGraphMetricNetwork(nn.Module, abc.ABC):
    def __init__(self,
                 embedding_networks: List[BaseGraphEmbeddingNetwork],
                 embdding_metric_network: BaseEmbeddingMetricNetwork):
        super().__init__()
        self.embedding_networks = nn.ModuleList(embedding_networks)
        self.embdding_metric_network = embdding_metric_network

    def get_params_list(self) -> List[Tensor]:
        return list(self.parameters())

    # TODO: Alternatively - annotated graph should contain what S2VGraph contains, so we use one unified graph repr accross board
    @abstractmethod
    def forward(self, graph_pairs_list: List[Tuple[T, T]]):
        pass


class S2VGraphEmbeddingGraphMetricNetwork(BaseGraphMetricNetwork):
    def __init__(self,
                 embedding_network: GraphsBatchEmbeddingNetwork,
                 embdding_metric_network: BaseEmbeddingMetricNetwork,
                 device: str = 'cpu'):
        super().__init__(embedding_networks=[embedding_network],
                         embdding_metric_network=embdding_metric_network)

    def forward(self, graph_pairs_list: List[Tuple[S2VGraph, S2VGraph]]):
        embedding_network = self.embedding_networks[0]

        # apply batching (work on multiple graphs to generate their embeddings
        all_graphs = [[tup[0], tup[1]] for tup in graph_pairs_list]
        all_graphs = [item for sublist in all_graphs for item in sublist]

        embeddings = embedding_network.forward_graphs(all_graphs)
        first_pair_element_embeddings = [embeddings[i] for i in range(len(embeddings)) if i % 2 == 0]
        second_pair_element_embeddings = [embeddings[i] for i in range(len(embeddings)) if i % 2 != 0]

        distances = self.embdding_metric_network(first_pair_element_embeddings, second_pair_element_embeddings,
                                                 is_sum_tensor_into_single_item=False)

        return distances


class SingleEmbeddingGraphMetricNetwork(BaseGraphMetricNetwork):
    def __init__(self,
                 embedding_network: BaseGraphEmbeddingNetwork,
                 embdding_metric_network: BaseEmbeddingMetricNetwork,
                 device: str = 'cpu'):
        super().__init__(embedding_networks=[embedding_network],
                         embdding_metric_network=embdding_metric_network)

    def forward(self, graph_pairs_list: List[Tuple[AnnotatedGraph, AnnotatedGraph]]):
        loss = 0.0

        for graph_pair in graph_pairs_list:
            G1, G2 = graph_pair

            embedding_network = self.embedding_networks[0]

            emb1 = embedding_network(G1.adjacency_matrix_torch,
                                               G1.node_indicator)
            emb2 = embedding_network(G2.adjacency_matrix_torch,
                                     G2.node_indicator)

            loss += self.embdding_metric_network(emb1, emb2)

        return loss

class MLPGraphMetricNetwork(BaseGraphMetricNetwork):
    def __init__(self,
                 embedding_networks: List[BaseGraphEmbeddingNetwork],
                 embdding_metric_network: BaseEmbeddingMetricNetwork,
                 device: str = 'cpu'):
        super().__init__(embedding_networks=embedding_networks,
                         embdding_metric_network=embdding_metric_network)

        input_dim = sum(
            [embedding_network.output_dim for embedding_network in embedding_networks])
        self.mlp = MLP(num_layers=2,
                       input_dim=input_dim,
                       hidden_dim=10,
                       output_dim=10,
                       device=device)

    def forward(self, graph_pairs_list: List[Tuple[AnnotatedGraph, AnnotatedGraph]]):
        loss = 0.0

        for graph_pair in graph_pairs_list:

            G1, G2 = graph_pair

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
            loss += self.embdding_metric_network(embedding1_mlp, embedding2_mlp)
            # loss = loss + self.mlp(torch.ones(2, 2)).sum()
        return loss
