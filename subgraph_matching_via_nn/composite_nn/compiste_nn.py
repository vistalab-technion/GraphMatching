from torch import nn

from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import \
    BaseNodeClassifierNetwork
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    BaseGraphEmbeddingNetwork


class CompositeNeuralNetwork(nn.Module):
    def __init__(self,
                 node_classifier_network: BaseNodeClassifierNetwork,
                 embedding_network: BaseGraphEmbeddingNetwork):
        super().__init__()

        self.node_classifier_network = node_classifier_network
        self.embedding_network = embedding_network

    def forward(self, A, x=None):
        # compute node classifier
        w = self.node_classifier_network(A=A, x=x)

        # compute embedding
        embedding = self.embedding_network(A=A, w=w)
        return embedding, w

    def init_params(self, **kwargs):
        # TODO: dix passing of parameters to submodules
        self.node_classifier_network.init_params(**kwargs)
        self.embedding_network.init_params(**kwargs)
