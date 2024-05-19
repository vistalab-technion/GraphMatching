from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
from torch_geometric.utils import from_scipy_sparse_matrix
import networkx as nx

from subgraph_matching_via_nn.graph_generators.graph_generators import \
    BaseGraphGenerator
from subgraph_matching_via_nn.utils.graph_utils import hamiltonian
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


# import torchsort


class BaseNodeClassifierNetwork(nn.Module):
    def __init__(self, classification_layer, input_dim=None, device='cpu'):
        super().__init__()
        self.classification_layer = classification_layer
        self.input_dim = input_dim
        self.device = device

    def train_node_classifier(self,
                              G: nx.graph = None,
                              G_sub: nx.graph = None,
                              graph_generator: BaseGraphGenerator = None):
        # Todo: create examples with the graph generator
        # Todo: train node classifier
        return

    def init_params(self, default_weights=None):
        pass


class IdentityNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, output_dim, classification_layer):
        super().__init__(classification_layer=classification_layer, input_dim=None)

        self.weights = nn.Parameter(torch.Tensor(output_dim, 1).type(TORCH_DTYPE))
        self.init_params()

    def forward(self, A, x=None, params: dict = None):
        # x = torch.matmul(A, self.weights)
        x = self.weights
        # x = self.diff_binarize(x, params)
        # todo: add function for internal operations. In Identity it will be pass
        w = self.classification_layer(A, x)

        return w

    def init_params(self, default_weights=None):
        if default_weights is None:
            self.weights.data.fill_(1 / len(self.weights))
        else:
            self.weights.data = default_weights

        self.classification_layer.init_weights()


class NNNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, input_dim, hidden_dim, output_dim, num_mid_layers,
                 classification_layer, device='cpu'):
        super().__init__(classification_layer=classification_layer,
                         input_dim=input_dim,
                         device=device)
        if input_dim is not None:
            self.register_buffer('x_stub',
                                 torch.ones(1, input_dim, dtype=TORCH_DTYPE,
                                            device=self.device))

        self.fc_in = nn.Linear(input_dim, hidden_dim,
                               dtype=TORCH_DTYPE)  # First Fully-Connected Layer
        self.mid_layers = nn.Sequential()
        self.hidden_linear_layers = []

        for i in range(num_mid_layers):
            hidden_linear_layer = nn.Linear(hidden_dim, hidden_dim,
                                            dtype=TORCH_DTYPE)
            self.hidden_linear_layers.append(hidden_linear_layer)
            self.mid_layers.add_module(name=f"fc_{i}",
                                       module=hidden_linear_layer)
            self.mid_layers.add_module(name=f"relu_{i}", module=nn.LeakyReLU())

        self.fc_out = nn.Linear(hidden_dim, output_dim,
                                dtype=TORCH_DTYPE)  # Third Fully-Connected Layer with output size output_dim
        #self.init_params()

    def init_params(self, default_weights=None):
        print("ignoring default_weights")
        with torch.no_grad():
            self.fc_in.reset_parameters()
            self.fc_out.reset_parameters()

            mid_layers = list(self.mid_layers.children())
            for layer in mid_layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            for layer in [self.fc_in, self.fc_out] + self.hidden_linear_layers:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in',
                                        nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)

            self.classification_layer.init_weights()

    def forward(self, A, x=None, params: dict = None):
        if x is None:
            x = self.x_stub
        else:
            x = x.to(device=self.device)

        x = self.fc_in(x)  # Apply first fully-connected layer
        skip_x = x
        x = F.relu(x)  # Apply ReLU activation function
        x = self.mid_layers(x)  # Apply second fully-connected layer
        x = x + skip_x  # Add skip connection
        x = F.relu(x)  # Apply ReLU activation function
        x = self.fc_out(x)  # Apply third fully-connected layer
        # x = torch.matmul(A, x.T)  # Apply adjacency matrix multiplication
        x = x.T

        w = self.classification_layer(A, x)
        return w


class GCNNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, num_node_features_input, hidden_dim, num_node_features_output,
                 num_mid_layers,
                 classification_layer, device):
        super(GCNNodeClassifierNetwork, self).__init__(
            classification_layer=classification_layer,
            input_dim=num_node_features_input, device=device)
        self.num_node_features_input = num_node_features_input
        self.fc_in = GCNConv(num_node_features_input, hidden_dim).to(dtype=TORCH_DTYPE)

        self.mid_layers = nn.Sequential()
        for i in range(num_mid_layers):
            self.mid_layers.add_module(name=f"fc_{i}",
                                       module=GCNConv(hidden_dim, hidden_dim).to(
                                           dtype=TORCH_DTYPE))
            self.mid_layers.add_module(name=f"relu_{i}", module=nn.LeakyReLU())

        self.fc_out = GCNConv(hidden_dim, num_node_features_output).to(
            dtype=TORCH_DTYPE)
        # self.skip_connection = nn.Identity(num_node_features_input,
        #                                    num_node_features_output)\
        #     .to(dtype=TORCH_DTYPE)  # Skip connection

    def init_params(self, default_weights=None):
        print("ignoring default_weights")
        with torch.no_grad():
            self.fc_in.reset_parameters()
            self.fc_out.reset_parameters()

            for layer in self.mid_layers.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            self.classification_layer.init_weights()

    def forward(self, A, x=None, params: dict = None):
        edge_index = A.nonzero().t()

        if x is None:
            x = torch.ones(A.shape[0], self.num_node_features_input, dtype=TORCH_DTYPE,
                           device=self.device)  # TODO: avoid recreating
        else:
            x = x.to(device=self.device)

        x = self.fc_in(x, edge_index)  # Apply first fully-connected layer
        skip_x = x
        x = F.leaky_relu(x)  # Apply ReLU activation function

        # apply middle layers
        for mid_layer_i in range(len(self.mid_layers) // 2):
            gcn_layer_i = mid_layer_i * 2
            relu_layer_i = gcn_layer_i + 1

            x = self.mid_layers[gcn_layer_i](x, edge_index)
            x = self.mid_layers[relu_layer_i](x)

        x = x + skip_x  # Add skip connection
        x = F.leaky_relu(x)  # Apply ReLU activation function
        x = self.fc_out(x, edge_index)  # Apply third fully-connected layer

        # skip_x = self.skip_connection(x)
        # x = self.conv1(x, edge_index)
        # x = F.leaky_relu(x)
        # x = self.conv2(x, edge_index)
        # x = x + skip_x

        x = self.classification_layer(A, x)

        return x.double()


class GraphPowerIterationNetwork(BaseNodeClassifierNetwork):
    def __init__(self, num_nodes, embedding_dim, num_iterations, classification_layer):
        super().__init__(classification_layer=classification_layer)

        # Initialization
        self.L_learnable = nn.Parameter(torch.rand((num_nodes, num_nodes)))
        self.num_iterations = num_iterations
        self.fc = nn.Linear(embedding_dim,
                            1)  # Sample output layer for binary classification

    def spectral_op(self, L, w, rho):
        # Define your E(w) function here based on the optimal indicator function.
        # For simplicity, we return an identity matrix

        spectral_op = L + rho
        return torch.eye(w.size(0))

    def forward(self, rho, epsilon, initial_indicator=None):
        if initial_indicator is None:
            w_k = torch.rand((self.L_learnable.size(0),))
        else:
            w_k = initial_indicator

        for _ in range(self.num_iterations):
            L = self.L_learnable + rho * self.spectral_op(w_k)
            y_k1 = torch.linalg.solve(L + epsilon * torch.eye(L.size(0)), w_k)

            # Update the indicator
            w_k = y_k1 / torch.norm(y_k1)

        # Node classification
        out = F.sigmoid(self.fc(w_k))
        return out

#
# for i in range(params["maxiter"]):
#     H = spectral_op(A, w, params)
#     #C = (torch.eye(n) - s * w @ w.T) @ H @ H @ (torch.eye(n) - s * w @ w.T)
#     # Compute eigenvalues and eigenvectors of A
#     e, v = find_smallest_eigenvector(H)
#     #e,v = inverse_power_method(A = H, epsilon=1e-8, max_iterations=500)
#     # w = params["scale"] * v
#     # w = v / torch.norm(v, float('inf'))
#     # w = params["scale"] * w
#     w = v / v.sum()
#     #obj = w.T @ H @ w +params["c"]**2
#     obj = obj_fun(A, w, params)

# # Create the model
# model = GraphPowerIterationNetwork(num_nodes=10, embedding_dim=10, num_iterations=5)
#
# # Sample forward pass
# rho = 0.5
# epsilon = 0.01
# output = model(rho, epsilon)
# print(output)
