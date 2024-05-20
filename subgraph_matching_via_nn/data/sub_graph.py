import networkx as nx
from torch import tensor
from subgraph_matching_via_nn.utils.graph_utils import get_edge_indicator, get_node_indicator, \
    get_normalized_node_indicator
from subgraph_matching_via_nn.utils.utils import get_graph_adj_mat_as_tensor, TORCH_DTYPE


class SubGraph:
    def __init__(self, G: nx.graph, G_sub: nx.graph=None, is_line_graph: bool=False,
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

        self.node_indicator = None
        self.edge_indicator = None
        if G_sub is not None:
            self.node_indicator = get_node_indicator(G=G, G_sub=G_sub)
            self.edge_indicator = get_edge_indicator(G=G, G_sub=G_sub)
        self.is_line_graph = is_line_graph

        self.device = device

    def inverse(self):
        assert self.is_line_graph

        inversed_G = nx.inverse_line_graph(self.G)
        inversed_G_sub = nx.inverse_line_graph(self.G_sub)
        inversed_sub_graph = SubGraph(G=inversed_G, G_sub=inversed_G_sub,
                                      is_line_graph=(not self.is_line_graph), device=self.device)

        return inversed_sub_graph

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
        return get_normalized_node_indicator(self.node_indicator, dtype=TORCH_DTYPE).to(device=self.device)

    @property
    def gt_indicator(self):
        return None if self.is_line_graph else self.node_indicator

    @property
    def original_edge_distribution_indicator(self):
        if self.is_line_graph:
            return self.node_indicator
        else:
            return self.edge_indicator
