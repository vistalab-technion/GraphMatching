import networkx as nx
import torch

from subgraph_matching_via_nn.data.graph_constants import GraphConstants
from subgraph_matching_via_nn.utils.graph_utils import get_edge_indicator, get_node_indicator, \
    get_normalized_node_indicator
from subgraph_matching_via_nn.utils.utils import get_graph_adj_mat_as_tensor, TORCH_DTYPE, \
    extract_node_features_from_graph


class SubGraph:
    def __init__(self, G: nx.graph, G_sub: nx.graph=None, is_line_graph: bool=False,
                 device='cpu', original_graph: nx.Graph=None, original_subgraph: nx.Graph=None):
        """
        G - networkx graph object
        G_sub - subgraph
        node_indicator - node indicator (i.e., 1 for nodes of G_sub that are in G)
        edge_indicator -
        edge indicator (i.e., dict[(i,j)]= 1 for edges of G_sub that are in G)
        in case of line graph, original_graph must be passed
        """
        self.G = G
        self.G_sub = G_sub

        self.node_indicator = None
        self.edge_indicator = None
        self.is_line_graph = is_line_graph

        self.device = device

        self.a_node_features = self.__extract_node_features(original_graph, processed_g=self.G)

        if G_sub is not None:
            self.node_indicator = get_node_indicator(G=G, G_sub=G_sub) # For the same nx.Graph, the nodes order are maintained
            self.edge_indicator = get_edge_indicator(G=G, G_sub=G_sub) # For the same nx.Graph, the edges mapping is the same

            self.a_sub_node_features = self.__extract_node_features(original_subgraph, processed_g=self.G_sub)

    def __extract_node_features(self, original_graph, processed_g):
        if self.is_line_graph:
            assert original_graph is not None
        else:
            original_graph = processed_g

        original_node_features = extract_node_features_from_graph(original_graph, GraphConstants.NODE_GATE_TYPE_ATTRIBUTE_NAME)
        if original_node_features is None:
            return None

        if self.is_line_graph:
            original_nodes_list = list(original_graph.nodes)
            node_features = []
            for edge in processed_g.nodes:
                src_node, target_node = edge
                src_node_index = original_nodes_list.index(src_node)
                target_node_index = original_nodes_list.index(target_node)

                node_features.append(
                    (original_node_features[src_node_index] + original_node_features[target_node_index]) / 2
                )
        else:
            node_features = original_node_features

        node_features = torch.stack(node_features)

        return node_features

    def inverse(self):
        assert self.is_line_graph

        inversed_G = nx.inverse_line_graph(self.G)
        inversed_G_sub = nx.inverse_line_graph(self.G_sub)
        inversed_sub_graph = SubGraph(G=inversed_G, G_sub=inversed_G_sub,
                                      is_line_graph=(not self.is_line_graph), device=self.device, original_graph=self.G,
                                      original_subgraph=self.G_sub)

        return inversed_sub_graph

    def set_device(self, device: str):
        self.device = device

    @property
    def A_node_features(self):
        return self.a_node_features

    @property
    def A_sub_node_features(self):
        return self.a_sub_node_features

    @property
    def node_features_number(self):
        if self.a_node_features is None:
            return 0

        return len(self.a_node_features[0])

    @property
    def A_full(self):
        """
        The order of the adjacency matrix nodes is guaranteed to be the same as the graph nodes order
        """
        return get_graph_adj_mat_as_tensor(self.G).to(device=self.device)

    @property
    def A_sub(self):
        """
        The order of the adjacency matrix nodes is guaranteed to be the same as the graph nodes order
        """
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
