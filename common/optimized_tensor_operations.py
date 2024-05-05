import torch
import torch.nn.functional as F


@torch.jit.script
def jit_relu(h):
    # non-linearity
    h = F.relu(h)
    return h
