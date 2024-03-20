from typing import List
import torch
from torch import stack
from torch.autograd import grad

from subgraph_matching_via_nn.composite_nn.localization_state_replayer import LocalizationStateReplayer
from subgraph_matching_via_nn.graph_embedding_networks.graph_embedding_nn import BaseGraphEmbeddingNetwork


def get_grad_distance(loss_instances: List[LocalizationStateReplayer], embedding_networks: List[BaseGraphEmbeddingNetwork]):
    with torch.enable_grad():
        loss_vals = [loss_instance.forward(embedding_networks) for loss_instance in loss_instances]
    params_list = [loss_instance.get_params_list(embedding_networks) for loss_instance in loss_instances]
    return get_grad_distance_via_loss_values(params_list, loss_vals)

def get_grad_distance_via_loss_values(params_lists, loss_vals):
    return stack([
        torch.norm(
            torch.cat(
                [elem.reshape(-1) for elem in
                                  grad(loss_val, params_list, retain_graph=True, create_graph=True,
                                       allow_unused=True) if elem is not None]
            ), p=2, dim=0)
        if params_list is not None else torch.tensor(float('nan'), requires_grad=False, device=loss_val.device)
        for loss_val, params_list in zip(loss_vals, params_lists)
    ]).reshape(-1) # reshape to have the proper 1-d shape
