import networkx as nx
import numpy as np
from torch import tensor
from subgraph_matching_via_nn.utils.utils import get_graph_adj_mat_as_tensor, TORCH_DTYPE


class SubGraph:
    def __init__(self, G: nx.graph, G_sub: nx.graph=None, node_indicator: np.array=None, edge_indicator: np.array=None, is_line_graph: bool=False,
                 device='cpu'):
        """
        G - networkx graph object
        G_sub - subgraphice
        node_indicator - node indicator (i.e., 1 for nodes of G_sub that are in G)
        edge_indicator -
        edge indicator (i.e., dict[(i,j)]= 1 for edges of G_sub that are in G)
        """
        self.G = G
        self.G_sub = G_sub
        self.node_indicator = node_indicator
        self.edge_indicator = edge_indicator
        self.is_line_graph = is_line_graph

        self.device = device

    def set_device(self, device: str):
        self.device = device

    @property
    def A_full(self):
        return get_graph_adj_mat_as_tensor(self.G).to(device=self.device)

    @property
    def A_sub(self):
        return get_graph_adj_mat_as_tensor(self.G_sub).to(device=self.device)

    @property
    def w_gt(self):
        # Assuming you have your graph G and subgraph G_sub defined
        # turn indicator into tensor and normalize to get distribution on nodes of line graph
        return (tensor(self.node_indicator, device=self.device)[:, None].float() / tensor(self.node_indicator, device=self.device).sum()).type(
        TORCH_DTYPE)

    @property
    def distribution_indicator(self):
        if self.is_line_graph:
            return self.node_indicator
        else:
            return self.edge_indicator
