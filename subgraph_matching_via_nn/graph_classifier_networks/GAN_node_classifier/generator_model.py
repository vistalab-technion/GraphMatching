from torch import nn
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, device, dtype=TORCH_DTYPE):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        ).to(dtype=dtype, device=device)

    def forward(self, x):
        return self.model(x)

    @property
    def latent_dimension_size(self):
        return self.latent_dim
