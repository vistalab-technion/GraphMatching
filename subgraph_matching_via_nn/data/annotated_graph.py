import networkx as nx
import torch
from torch import Tensor

from common.graph_utils import relabel_graph_nodes_by_contiguous_order
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class GraphConstants:
    EDGE_WEIGHT_ATTRIBUTE_NAME = 'weight_attribute'
    NODE_DEGREE_WEIGHT_ATTRIBUTE_NAME = 'weight_attribute'


class AnnotatedGraph(object):
    def __init__(self, g: nx.graph, label=None, node_attributes=None,
                 device: str = "cpu",
                 node_attribute_name: str = GraphConstants.NODE_DEGREE_WEIGHT_ATTRIBUTE_NAME):
        '''
            g: a networkx graph
            label: an integer graph label
        '''
        self.label = label

        g = relabel_graph_nodes_by_contiguous_order(g, copy=True)
        self.g = g

        self.node_attribute_name_to_vector_map = {}
        self.edge_attribute_name_to_vector_map = {}

        if node_attributes is None:
            n = len(g.nodes)
            node_attributes = torch.zeros(n, device=device, dtype=TORCH_DTYPE)
            degrees_map = dict(nx.degree(g))
            n_non_isolated_nodes = len(
                list(filter(lambda val: val > 0, list(degrees_map.values()))))

            for node_index, node_id in enumerate(g.nodes):
                if degrees_map[node_id] > 0:
                    node_attributes[node_index] = 1 / n_non_isolated_nodes

        self.attach_node_attributes(node_attributes, node_attribute_name)

    def attach_edge_attributes(self, device=None, requires_grad=False,
                               edge_weight_attrs: object = None,
                               edge_attribute_name=GraphConstants.EDGE_WEIGHT_ATTRIBUTE_NAME):

        if edge_weight_attrs is None:
            edge_list = self.g.edges
            edge_weight_attrs = torch.ones((len(edge_list)),
                                           requires_grad=requires_grad,
                                           device=device)

        self.edge_attribute_name_to_vector_map[edge_attribute_name] = edge_weight_attrs

    def get_edge_attributes_vector(self, edge_attribute_name) -> Tensor:
        nodes = self.g.nodes
        n_nodes = len(nodes)
        if n_nodes == 0:
            raise ValueError("Empty graph")

        if edge_attribute_name in self.edge_attribute_name_to_vector_map:
            return self.edge_attribute_name_to_vector_map[edge_attribute_name]
        else:
            raise ValueError("No edge attributes exist")

    def attach_node_attributes(self, node_attribute_mask,
                               node_attribute_name):
        self.node_attribute_name_to_vector_map[
            node_attribute_name] = node_attribute_mask

    def get_node_attributes_vector(self, node_attribute_name) -> Tensor:
        nodes = self.g.nodes
        n_nodes = len(nodes)
        if n_nodes == 0:
            raise ValueError("Empty graph")

        if node_attribute_name in self.node_attribute_name_to_vector_map:
            return self.node_attribute_name_to_vector_map[node_attribute_name]
        else:
            raise ValueError("No node attributes exist")

    @property
    def adjacency_matrix(self):
        return nx.to_numpy_array(self.g)

    @property
    def adjacency_matrix_torch(self):
        return torch.from_numpy(self.adjacency_matrix).type(TORCH_DTYPE)

    @property
    def node_indicator(self):
        return self.get_node_attributes_vector(
            GraphConstants.NODE_DEGREE_WEIGHT_ATTRIBUTE_NAME)[..., None]
