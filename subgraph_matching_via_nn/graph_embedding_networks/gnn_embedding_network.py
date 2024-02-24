import networkx as nx
from powerful_gnns.models.graphcnn import GraphCNN
from powerful_gnns.util import load_data_given_graph_list_and_label_map, S2VGraph
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import \
    GraphsBatchEmbeddingNetwork


class GNNEmbeddingNetwork(GraphsBatchEmbeddingNetwork):
    def __init__(self, gnn_model: GraphCNN, params):
        super().__init__()
        self.gnn_model = gnn_model
        self.params = params

    def forward_graphs(self, batch_graph):
        embeddings = self.gnn_model.get_embedding(batch_graph)
        return embeddings

    def forward(self, A, w, params: dict = None):
        # A -> nx.Graph
        G = nx.from_numpy_matrix(A.detach().cpu().numpy())

        # nx.Graph -> S2VGraph
        s2v_graph = S2VGraph(G, label=None)
        batch_graph, _ = load_data_given_graph_list_and_label_map([s2v_graph], label_dict = {}, degree_as_tag=True,
                                                                  device=w.device, print_stats=False)

        # set the graph node mask
        s2v_graph = batch_graph[0]
        s2v_graph.node_mask = w
        s2v_graph.node_features = s2v_graph.node_features.to(device=s2v_graph.node_mask.device)

        embedding = self.gnn_model.get_embedding(batch_graph)
        return embedding

    def init_params(self):
        pass

    @property
    def output_dim(self):
        return self.gnn_model.get_embedding_dim()

    @property
    def embedding_type(self):
        return f"GNN"
