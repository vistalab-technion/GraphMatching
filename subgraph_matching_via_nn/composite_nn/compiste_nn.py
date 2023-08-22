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

        self.mask_gen_network = node_classifier_network
        self.embedding_network = embedding_network

    def forward(self, A, x=None):
        # compute mask
        w = self.mask_gen_network(A=A, x=x)

        # compute embedding
        embedding = self.embedding_network(A=A, w=w)
        return embedding, w

    def init_params(self):
        self.mask_gen_network.init_params()
        self.embedding_network.init_params()
