from enum import Enum
from subgraph_matching_via_nn.graph_classifier_networks.classification_layer.classification_layer import \
    TopkSoftmaxClassificationLayer, SigmoidClassificationLayer, SoftmaxClassificationLayer, \
    SquaredNormalizedClassificationLayer, IdentityClassificationLayer
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import NNNodeClassifierNetwork, \
    IdentityNodeClassifierNetwork, GCNNodeClassifierNetwork


class NodeClassifierLastLayerType(Enum):
    TopKSoftmax = 0,
    Sigmoid = 1,
    Softmax = 2,
    SquaredNormalized = 3,
    Identity = 4,


class NodeClassifierNetworkType(Enum):
    NN = 0,
    Identity = 1,
    GCN = 2,


class NodeClassifierNetworkFactory:

    @staticmethod
    def create_node_classifier_network(processed_G, last_layer_type: NodeClassifierLastLayerType,
                                       node_classifier_network_type: NodeClassifierNetworkType, params):
        device = params['device']
        input_dim = len(processed_G.nodes())
        hidden_dim = 20
        output_dim = len(processed_G.nodes())
        if last_layer_type == NodeClassifierLastLayerType.Identity:
            last_layer = IdentityClassificationLayer()
        elif last_layer_type == NodeClassifierLastLayerType.TopKSoftmax:
            last_layer = TopkSoftmaxClassificationLayer(k=params["m"], default_temp=0.1, learnable_temp=False)
        elif last_layer_type == NodeClassifierLastLayerType.Sigmoid:
            last_layer = SigmoidClassificationLayer()
        elif last_layer_type == NodeClassifierLastLayerType.Softmax:
            last_layer = SoftmaxClassificationLayer()
        elif last_layer_type == NodeClassifierLastLayerType.SquaredNormalized:
            last_layer = SquaredNormalizedClassificationLayer()
        else:
            raise ValueError(f"Unsupported layer type: {last_layer_type}")

        if node_classifier_network_type == NodeClassifierNetworkType.NN:
            node_classifier_network = NNNodeClassifierNetwork(input_dim=input_dim,
                                                              hidden_dim=hidden_dim,
                                                              output_dim=output_dim,
                                                              classification_layer=last_layer,
                                                              num_mid_layers=params['num_mid_layers'],
                                                              device=device
                                                              )
        elif node_classifier_network_type == NodeClassifierNetworkType.Identity:
            node_classifier_network = IdentityNodeClassifierNetwork(output_dim=output_dim,
                                                                    classification_layer=last_layer,
                                                                    device=device)
        elif node_classifier_network_type == NodeClassifierNetworkType.GCN:
            node_classifier_network = GCNNodeClassifierNetwork(input_dim=1,
                                                               hidden_dim=hidden_dim,
                                                               num_classes=1,
                                                               classification_layer=last_layer,
                                                               device=device
                                                               )
        else:
            raise ValueError(f"Unsupported layer type: {node_classifier_network_type}")

        return node_classifier_network.to(device=device)
