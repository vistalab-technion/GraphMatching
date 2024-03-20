from typing import List
import torch
from torch import nn

from subgraph_matching_via_nn.composite_nn.composite_solver import BaseCompositeSolver, PickleSupportedCompositeSolver
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import BaseGraphEmbeddingNetwork
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class ReplayableLocalizationState(nn.Module):
    def __init__(self, sub_graph: SubGraph, w: torch.Tensor):
        super(ReplayableLocalizationState, self).__init__()
        self.sub_graph = sub_graph
        self.w = w


class LocalizationStateReplayer:
    def __init__(self, composite_solver: PickleSupportedCompositeSolver, state: ReplayableLocalizationState):
        self.composite_solver = composite_solver
        self.state = state

        # prepare attributes for potential dump (by dataloader) ->
        # avoid using lambda functions, not stable on some platforms and torch version
        # thus, use PickleSupportedCompositeSolver


    def forward(self, embedding_networks: List[BaseGraphEmbeddingNetwork]):
        w = self.state.w
        sub_graph = self.state.sub_graph
        return self.composite_solver.solve_using_external_params(w, sub_graph.A_full, sub_graph.A_sub,
                                                                 embedding_networks=embedding_networks, dtype=TORCH_DTYPE)

    def get_params_list(self, embedding_networks: List[BaseGraphEmbeddingNetwork]):
        all_params = [list(embedding_network.parameters()) for embedding_network in embedding_networks]
        all_params = [param for lst in all_params for param in lst]
        return all_params
