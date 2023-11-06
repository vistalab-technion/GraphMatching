from typing import List

import numpy as np
import torch
from torch import nn

from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import \
    BaseNodeClassifierNetwork
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    BaseGraphEmbeddingNetwork
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class CompositeNeuralNetwork(nn.Module):
    def __init__(self,
                 node_classifier_network: BaseNodeClassifierNetwork,
                 embedding_networks: List[BaseGraphEmbeddingNetwork]):
        super().__init__()

        self.node_classifier_network = node_classifier_network
        self.embedding_networks = embedding_networks

    def forward(self, A, x=None, params: dict = None):
        # compute node classifier
        w = self.classify(A=A, x=x, params=params)

        # compute embedding
        embeddings = self.embed(A=A, w=w, params=params)

        return embeddings, w

    def embed(self, A, w, params: dict = None):
        embeddings = []
        for embedding_network in self.embedding_networks:
            embeddings.append(embedding_network(A=A, w=w, params=params))
            # TODO: apply mlp. for example
            #  ||a(emb1-emb1_gt)||^2+||b(emb2-emb2_gt)||^2  s.t. (a^2+b^2)=1
            #  total_emb = mlp(embeddings) - > ||total_emb - total_emb_gt||^2
        return embeddings

    def classify(self, A, x, params: dict = None):
        w = self.node_classifier_network(A=A, x=x, params=params)
        return w

    def init_params(self, **kwargs):
        # TODO: dix passing of parameters to submodules
        self.node_classifier_network.init_params(**kwargs)
        self.embedding_network.init_params(**kwargs)

    @staticmethod
    def get_initial_indicator(gt_node_indicator_processed: np.array, is_based_on_gt_indicator: bool = True,
                       rand_factor: float = 0):
        """
        Returns:
            GT indicator as a distribution vector
            Initial prescribed indicator distribution vector
        """
        gt_indicator_tensor = torch.tensor(gt_node_indicator_processed)[:, None].type(TORCH_DTYPE)
        gt_indicator_tensor = gt_indicator_tensor / gt_indicator_tensor.sum()

        gt_indicator_factor = 1 if is_based_on_gt_indicator else 0
        x0 = (gt_indicator_factor * gt_indicator_tensor.clone() +
              rand_factor * (torch.rand(gt_indicator_tensor.shape, dtype=TORCH_DTYPE) - 0.5))

        x0 = x0 / torch.sum(x0)

        return gt_indicator_tensor, x0

    def init_network_with_indicator(self, processed_sub_graph: SubGraph):
        gt_node_indicator_processed = processed_sub_graph.node_indicator

        gt_indicator_tensor, x0 = CompositeNeuralNetwork.get_initial_indicator(gt_node_indicator_processed)
        # self.node_classifier_network.init_params(default_weights=x0) #TODO

        return gt_indicator_tensor
