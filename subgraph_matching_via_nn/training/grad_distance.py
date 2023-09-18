import torch
from torch import stack
from torch.autograd import grad


def get_grad_distance(z1, z2, loss_function):
    loss_instances = [loss_function(za, zb) for za, zb in zip(z1, z2)]
    loss_vals = [loss_instance()[0] for loss_instance in loss_instances]
    params_list = []
    for loss_instance in loss_instances:
        params_list += loss_instance.get_params_list()
    return get_grad_distance_via_loss_values(params_list, loss_vals)

def get_grad_distance_via_loss_values(params_list, loss_vals):
    return stack([torch.norm(stack(grad(loss_val, params, retain_graph=True, create_graph=True)))
                  if params is not None else torch.tensor(float('nan'), requires_grad=False, device=params.device)
                  for loss_val, params in
           zip(loss_vals, params_list)]).reshape(-1) # reshape to have the proper 1-d shape
