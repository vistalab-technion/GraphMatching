from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
from torch_geometric.utils import from_scipy_sparse_matrix
import networkx as nx

from subgraph_matching_via_nn.graph_generation.graph_generation import \
    BaseGraphGenerator
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE

def sigmoid_layer(x, param):
    return torch.sigmoid(param * x)

def softmax_layer(x, param):
    return F.softmax(x, dim=0)

def squared_normalized_layer(x, param):
    x = x**2
    x=x/x.sum()
    return x

class BaseNodeClassifierNetwork(nn.Module):
    def __init__(self, last_layer):
        self._last_layer = last_layer
        super().__init__()

    def train_node_classifier(self,
                              G_sub: nx.graph = None,
                              graph_generator: BaseGraphGenerator = None):
        # Todo: create examples with the graph generator
        # Todo: train node classifier
        return


class NNNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, input_dim, hidden_dim, output_dim, learnable_sigmoid=True,
                 default_sigmoid_param_value=10, last_layer = softmax_layer):
        super().__init__(last_layer = last_layer)

        self.fc1 = nn.Linear(input_dim, hidden_dim,
                             dtype=TORCH_DTYPE)  # First Fully-Connected Layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=TORCH_DTYPE)  # Second Fully-Connected Layer
        self.fc3 = nn.Linear(hidden_dim, output_dim,
                             dtype=TORCH_DTYPE)  # Third Fully-Connected Layer with output size output_dim
        self.skip_connection = nn.Linear(input_dim, hidden_dim,
                                         dtype=TORCH_DTYPE)  # Skip connection
        self.sigmoid_param = nn.Parameter(torch.Tensor([default_sigmoid_param_value]))
        if not learnable_sigmoid:
            self.sigmoid_param.requires_grad = False
        self.register_buffer('uniform_feature', torch.ones(1, input_dim))

    def forward(self, A, x=None):
        if x is None:
            x = self.uniform_feature.type(TORCH_DTYPE)

        skip_x = self.skip_connection(x)  # Compute skip connection

        x = self.fc1(x)  # Apply first fully-connected layer
        x = x + skip_x  # Add skip connection
        x = F.relu(x)  # Apply ReLU activation function
        x = self.fc2(x)  # Apply second fully-connected layer
        x = x + skip_x  # Add skip connection
        x = F.relu(x)  # Apply ReLU activation function
        x = self.fc3(x)  # Apply third fully-connected layer
        # x = torch.matmul(A, x.T)  # Apply adjacency matrix multiplication
        x = x.T
        w = self._last_layer(x,self.sigmoid_param)
        # w = F.softmax(x, dim=0)  # Apply softmax to get the vector w
        #  w = torch.sigmoid(self.sigmoid_param * x)
        # w = x ** 2
        # w = w / w.sum()

        return w


class IdentityNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, input_dim, learnable_sigmoid=True,
                 default_value_sigmoid_param=10, last_layer=softmax_layer):
        super().__init__(last_layer=last_layer)

        self._default_value_sigmoid_param = default_value_sigmoid_param

        self.weights = nn.Parameter(torch.Tensor(input_dim, 1).type(TORCH_DTYPE))
        self.sigmoid_param = nn.Parameter(torch.Tensor([default_value_sigmoid_param]))
        if not learnable_sigmoid:
            self.sigmoid_param.requires_grad = False
        self.register_buffer('uniform_feature', torch.ones(1, input_dim))

        self.init_params()

    def forward(self, A, x=None):
        # x = torch.matmul(A, self.weights)
        x = self.weights
        # w = torch.sigmoid(self.sigmoid_param * x)
        # w = F.softmax(x, dim=0)  # Apply softmax to get the vector w
        w = self._last_layer(x,self.sigmoid_param)

        return w

    def init_params(self, default_weights=None, default_sigmoid_param=None):
        if default_weights is None:
            self.weights.data.fill_(1 / len(self.weights))
        else:
            self.weights.data = default_weights

        if default_sigmoid_param is None:
            self.sigmoid_param.data.fill_(self._default_value_sigmoid_param)
        else:
            self.sigmoid_param.data = default_sigmoid_param


class GCNNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, input_dim, hidden_dim, num_classes, learnable_sigmoid=True,
                 default_value_sigmoid_param=10, last_layer=softmax_layer):
        super(GCNNodeClassifierNetwork, self).__init__(last_layer=last_layer)
        self.conv1 = GCNConv(input_dim, hidden_dim, dtype=TORCH_DTYPE)
        self.conv2 = GCNConv(hidden_dim, num_classes, dtype=TORCH_DTYPE)
        self.skip_connection = nn.Identity(input_dim, num_classes,
                                           dtype=torch.float)  # Skip connection
        self.sigmoid_param = nn.Parameter(torch.Tensor([default_value_sigmoid_param]))
        if not learnable_sigmoid:
            self.sigmoid_param.requires_grad = False
        # self.register_buffer('uniform_feature', torch.ones(1, input_dim))

    def forward(self, A, x=None):
        edge_index = A.nonzero().t()

        if x is None:
            x = torch.ones(A.shape[0], 1)
        skip_x = self.skip_connection(x.float())

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = x + skip_x
        # x = torch.sigmoid(self.sigmoid_param * x)
        # x = x**2
        # x=x/x.sum()
        x = self._last_layer(x, self.sigmoid_param)

        return x.double()


