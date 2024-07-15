import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import BaseNodeClassifierNetwork
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class ClusteringNodeClassifierNetwork(BaseNodeClassifierNetwork):
    def __init__(self, output_dim, classification_layer, device):
        super().__init__(classification_layer=classification_layer, input_dim=None, device=device)

        weights_tensor = torch.rand((output_dim, 1), dtype=TORCH_DTYPE, device=self.device)

        self.weights = nn.Parameter(weights_tensor)
        self.classification_layer = classification_layer
        self.init_params()

    def __cluster_by_node_features(self, A, node_features, k):
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(node_features)

        # Get cluster labels
        labels = torch.from_numpy(kmeans.labels_).to(device=A.device)

        # decide on cluster identity, based on n and k
        def closer_to_target(a, b, c):
            if abs(a - c) < abs(b - c):
                return a
            else:
                return b

        n = len(labels)
        one_cluster_ratio = labels.sum() / n
        zero_cluster_ratio = 1 - one_cluster_ratio

        target_subgraph_ratio = k / len(A)

        subgraph_cluster_label = 0
        if closer_to_target(zero_cluster_ratio, one_cluster_ratio, target_subgraph_ratio) == one_cluster_ratio:
            subgraph_cluster_label = 1

        # Assign cluster labels back to nodes
        w = (labels == subgraph_cluster_label).to(dtype=A.dtype)

        return w

    def forward(self, A, x=None, node_features=None, params: dict = None):
        x = self.__cluster_by_node_features(A=A, node_features=node_features)
        # x = self.diff_binarize(x, params)
        w = self.classification_layer(A, x)

        return w

    def init_params(self, default_weights=None):
        with torch.no_grad():
            if default_weights is None:
                self.weights.data.fill_(0)
            else:
                self.weights.data = default_weights

            self.classification_layer.init_weights()

if __name__ == "__main__":

    # Create a sample graph with node features
    G = nx.Graph()

    # Add nodes with features as a dictionary
    G.add_node(0, features=np.array([0.1, 0.2]))
    G.add_node(1, features=np.array([0.2, 0.3]))
    G.add_node(2, features=np.array([0.3, 0.4]))
    G.add_node(3, features=np.array([0.4, 0.5]))
    G.add_node(4, features=np.array([0.5, 0.6]))

    # Extract node features
    node_features = np.array([G.nodes[node]['features'] for node in G.nodes])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(node_features)

    # Get cluster labels
    labels = kmeans.labels_

    # Assign cluster labels back to nodes
    for node, label in zip(G.nodes, labels):
        G.nodes[node]['cluster'] = label

    # Print node clusters
    for node in G.nodes(data=True):
        print(f"Node {node[0]}: Cluster {node[1]['cluster']}")
