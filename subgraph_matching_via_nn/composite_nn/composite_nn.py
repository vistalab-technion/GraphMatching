from typing import List

from torch import nn

from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import \
    BaseNodeClassifierNetwork
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    BaseGraphEmbeddingNetwork


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
