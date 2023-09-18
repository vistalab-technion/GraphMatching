from typing import Dict
import torch

from subgraph_matching_via_nn.training.PairSampleBase import PairSampleBase
from subgraph_matching_via_nn.training.grad_distance import get_grad_distance


class LocalizationGradDistance:

    def __init__(self, problem_params: Dict, solver_params: Dict):
        self.problem_params = problem_params
        self.solver_params = solver_params

    def compute_grad_distance(self, pair_sample_info: PairSampleBase):
        g1 = pair_sample_info.graph
        g2 = pair_sample_info.subgraph

        loss_instance = pair_sample_info.localization_object
        if (loss_instance is None) or (len(loss_instance.get_params_list()) == 0):
            return float("Inf") * torch.ones(size=(1,), device=self.solver_params["device"], requires_grad=False)

        # graph1, graph2 needed for API, but not really used when calculating grad distance
        alignment_loss_fn = lambda graph1, graph2: pair_sample_info.localization_object

        return get_grad_distance([g1], [g2], alignment_loss_fn)
