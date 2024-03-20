from typing import Dict, List
import torch

from subgraph_matching_via_nn.composite_nn.localization_state_replayer import LocalizationStateReplayer
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import BaseGraphEmbeddingNetwork
from subgraph_matching_via_nn.training.grad_distance import get_grad_distance


class LocalizationGradDistance:

    def __init__(self, problem_params: Dict, solver_params: Dict):
        self.problem_params = problem_params
        self.solver_params = solver_params

    def compute_grad_distance(self, localization_state_replayer: LocalizationStateReplayer,
                              embedding_networks: List[BaseGraphEmbeddingNetwork]):
        if (localization_state_replayer is None) or (len(localization_state_replayer.get_params_list(embedding_networks)) == 0):
            return float("Inf") * torch.ones(size=(1,), device=self.solver_params["device"], requires_grad=False)

        return get_grad_distance([localization_state_replayer], embedding_networks)
