from torch import nn
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class Discriminator(nn.Module):
    def __init__(self, input_dim, device, dtype=TORCH_DTYPE):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(dtype=dtype, device=device)

    def forward(self, x):
        return self.model(x)