from enum import Enum
from typing import List

import torch
from subgraph_matching_via_nn.graph_embedding_networks.gnn_embedding_network import GNNEmbeddingNetwork
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import MomentEmbeddingNetwork, \
    SpectralEmbeddingNetwork
from subgraph_matching_via_nn.utils.graph_utils import laplacian
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class EmbeddingNetworkType(Enum):
    Moments = 0,
    Spectral = 1,
    GraphCNN = 2,


class GraphEmbeddingNetworkFactory:

    @staticmethod
    def create_embedding_networks(sub_graph, params, embedding_network_types: List[EmbeddingNetworkType]):
        indicator_size = params["m"]

        embedding_nns = []
        for embedding_network_type in embedding_network_types:
            if embedding_network_type == EmbeddingNetworkType.Moments:
                embedding_nn = MomentEmbeddingNetwork(n_moments=params["n_moments"],
                                                      moments_type=params["moment_type"])

            elif embedding_network_type == EmbeddingNetworkType.Spectral:
                evals, _ = torch.linalg.eigh(laplacian(sub_graph.A_sub))
                params["n_eigs"] = min(indicator_size, 10)
                params['diagonal_scale'] = 2 * torch.max(evals)
                params['zero_eig_scale'] = 10

                embedding_nn = SpectralEmbeddingNetwork(n_eigs=params["n_eigs"],
                                                        spectral_op_type=params['spectral_op_type'],
                                                        diagonal_scale=params['diagonal_scale'],
                                                        indicator_scale=indicator_size,
                                                        zero_eig_scale=params["zero_eig_scale"])
            elif embedding_network_type == EmbeddingNetworkType.GraphCNN:
                model_factory_func = params["graphcnn_factory_func"]
                model = model_factory_func(device='cpu')
                embedding_nn = GNNEmbeddingNetwork(gnn_model=model, params=params)
            else:
                raise ValueError(f"Unsupported network type: {embedding_network_type}")

            if params['is_use_model_compliation']:
                embedding_nn = torch.compile(embedding_nn)
            embedding_nns.append(embedding_nn)

        return [embedding_nn.to(device=params['device'], dtype=TORCH_DTYPE) for embedding_nn in embedding_nns]
