import torch
from torch import nn

from rbf_layer.rbf_layer import RBFLayer
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNEmbeddingNetwork(nn.Module):
    def __init__(self, num_node_features_input, num_node_features_output, hidden_features_amount=16):
        super().__init__()

        self.num_node_features_input = num_node_features_input
        self.hidden_features_amount = hidden_features_amount
        self.num_node_features_output = num_node_features_output

        self.conv1 = GCNConv(num_node_features_input, hidden_features_amount)
        self.conv2 = GCNConv(hidden_features_amount, num_node_features_output)

    def forward(self, A, x=None):
        edge_index = A.nonzero().t()

        if x is None:
            #TODO: assuming full graph, so taking all nodes (as x)
            x = torch.ones(A.shape[0], self.num_node_features_input)

        x = self.conv1(x, edge_index)
        #
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index).unsqueeze(1)
        x = x.sum(dim=0)
        return x

        # return x.double()

# Define an RBF layer where the dimensionality of the input feature is 2,
# the number of kernels is 3, and 2 output features

# euclidean norm
def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

from subgraph_matching_via_nn.data.sub_graph import SubGraph


class RBFGraphModel(nn.Module):
    def __init__(self, embedding_model: GCNEmbeddingNetwork, embedding_features_size):
        super().__init__()

        self.embedding_model = embedding_model
        self.rbf = RBFLayer(in_features_dim=embedding_features_size,
               num_kernels=5,
               out_features_dim=1,
               radial_function=rbf_gaussian,
               norm_function=euclidean_norm,
               normalization=True)

    def forward(self, g: SubGraph):

        #return self.rbf(torch.ones(1, 3))

        A = g.A_full

        emb = self.embedding_model(A).unsqueeze(0)
        # for name, param in self.embedding_model.conv1.named_parameters():
        #     print(name, param.grad)
        # print(emb.requires_grad)
        #print(emb.shape)
        # return 1/ torch.exp(emb.sum())
        return emb
        rbf_val = self.rbf(emb)

        # print(f"{emb}, {rbf_val}")

        return rbf_val