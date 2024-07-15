import networkx as nx
from subgraph_matching_via_nn.utils.graph_utils import create_weighted_adjacency_matrix_from_node_mask,\
    create_weighted_adjacency_matrix_from_edge_mask, joint_degree_matrix, adjacency_matrix_to_edges
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import BaseGraphEmbeddingNetwork


class WeightedJointDegreesEmbeddingNetwork(BaseGraphEmbeddingNetwork):

    def __init__(self, n_nodes: int):
        super(WeightedJointDegreesEmbeddingNetwork, self).__init__()
        self.n_nodes = n_nodes

    @property
    def output_dim(self):
        return self.n_nodes * self.n_nodes

    @property
    def embedding_type(self):
        return "WeightedJointDegreesEmbedding"

    def forward(self, A, w, node_features=None, params: dict = None, is_use_last_args: bool = False):
        # input: A, w
        # computes the joint degree matrix
        #     with the following variant: sum the w (when we work with edge mask)
        #     for node mask, convert it to an edge mask and then apply (use the logic from the plot utils)

        n = A.shape[0]
        if n == len(w):
            # node mask
            # convert a node weight mask to edge_mask
            weight_matrix = create_weighted_adjacency_matrix_from_node_mask(w, n)
        else:
            # edge mask
            weight_matrix = create_weighted_adjacency_matrix_from_edge_mask(w, n)
        return joint_degree_matrix(A, edge_weight_matrix=weight_matrix)


if __name__ == "__main__":
    joint_degrees = {
        1: {4: 1},
        2: {2: 2, 3: 2, 4: 2},
        3: {2: 2, 4: 1},
        4: {1: 1, 2: 2, 3: 1},
    }
    G = nx.joint_degree_graph(joint_degrees)
    #nx.draw(G)
    A = nx.adjacency_matrix(G).todense()

    # initialize embedding network
    output_matrix = joint_degree_matrix(A)

    # Test non weighted case
    def test_joint_degree_matrix(joint_degrees_map, joint_degree_matrix):
        for node_degree_i, node_degree_i_map in joint_degrees_map.items():
            for node_degree_j, n_edges in node_degree_i_map.items():
                assert (n_edges == joint_degree_matrix[node_degree_i][node_degree_j])

    test_joint_degree_matrix(joint_degrees, output_matrix)

    # Test weighted case
    n = A.shape[0]
    embedding_network = WeightedJointDegreesEmbeddingNetwork(n)

    edge_tuples = adjacency_matrix_to_edges(A)
    # edge mask
    # weighted_output_matrix = weigthed_joint_degree_matrix(A, w={item: 1 for item in edge_tuples})

    # node mask
    weighted_output_matrix = embedding_network.forward(A=A, w={node: 1 for node in range(n)})

    test_joint_degree_matrix(joint_degrees, weighted_output_matrix)

    # float mask
    # np.random.randint(0, 10, size=(n)) / 10
