from abc import abstractmethod, ABC

import torch
from torch import nn, cat

from subgraph_matching_via_nn.utils.graph_utils import hamiltonian, graph_edit_matrix


class BaseGraphEmbeddingNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, A, w, params: dict = None):
        pass

    @property
    @abstractmethod
    def output_dim(self):
        pass

    @property
    @abstractmethod
    def embedding_type(self):
        pass


class MomentEmbeddingNetwork(BaseGraphEmbeddingNetwork):
    def __init__(self, n_moments, moments_type='standardized'):
        super().__init__()
        self._moments_type = moments_type
        self._n_moments = n_moments

    def forward(self, A, w, params: dict = None):
        if self._moments_type == 'standardized_central':
            embedding = self.compute_standardized_central_moments(w, A, self._n_moments)
        elif self._moments_type == 'standardized_raw':
            embedding = self.compute_standardized_raw_moments(w, A, self._n_moments)
        elif self._moments_type == 'raw':
            embedding = self.compute_raw_moments(w, A, self._n_moments)
        elif self._moments_type == 'central':
            embedding = self.compute_central_moments(w, A, self._n_moments)
        else:
            raise Exception("unknown moments type")
        return embedding

    @staticmethod
    def compute_standardized_central_moments(w, A, n_moments):
        mean = w.T @ A @ w
        var = w.T @ ((A @ w - mean) ** 2)
        moments = []
        # for standardized moments, mom1 == 0, mom2 == 1 so no need to append them
        if n_moments > 2:
            for k in range(3, n_moments + 1):
                # mom = w.T @ ((A @ w - mean) ** k)
                # moments.append(mom / (var ** (k / 2)))
                mom = w.T @ (((A @ w - mean) / (var ** 0.5)) ** k)
                moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_standardized_raw_moments(w, A, n_moments):
        var = w.T @ ((A @ w) ** 2)
        moments = []
        # for raw standardized moments, mom2 == 1 so no need to append it
        if n_moments > 2:
            for k in range(1, n_moments + 1):
                if k != 2:
                    mom = (w.T @ (((A @ w) / (var ** 0.5)) ** k))
                    moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_central_moments(w, A, n_moments):
        mean = w.T @ A @ w
        moments = []
        if n_moments >= 2:
            for k in range(2, n_moments + 1):
                mom = w.T @ ((A @ w - mean) ** k)
                moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_raw_moments(w, A, n_moments):
        moments = []
        for k in range(1, n_moments + 1):
            mom = w.T @ ((A @ w) ** k)
            moments.append(mom)
        return cat(moments).squeeze()

    def init_params(self):
        pass

    @property
    def output_dim(self):
        if self._moments_type == 'standardized_central':
            return self._n_moments - 2
        elif self._moments_type == 'standardized_raw':
            return self._n_moments - 1
        elif self._moments_type == 'raw':
            return self._n_moments
        elif self._moments_type == 'central':
            return self._n_moments - 1

    @property
    def embedding_type(self):
        return f"{self._moments_type} moments"


class SpectralEmbeddingNetwork(BaseGraphEmbeddingNetwork):
    def __init__(self, n_eigs=5,
                 spectral_op_type='Laplacian',
                 diagonal_scale: float = 1,
                 indicator_scale: float = 1,
                 zero_eig_scale: float = 1):
        super().__init__()
        self._spectral_op_type = spectral_op_type
        self._n_eigs = n_eigs
        self._diagonal_scale = diagonal_scale
        self._indicator_scale = indicator_scale
        self._zero_eig_scale = zero_eig_scale

    def forward(self, A, w, params: dict = None):
        H = self.spectral_operator(A, w)
        evals, evecs = torch.linalg.eigh(H)
        embedding = evals[:self._n_eigs]

        embedding_clone = embedding.clone()
        embedding_clone[0] = self._zero_eig_scale * embedding[0]
        return embedding_clone

    def spectral_operator(self, A, w):
        v = 1 - self._indicator_scale * w

        if self._spectral_op_type == 'Laplacian':
            H = hamiltonian(A, v, self._diagonal_scale)
        if self._spectral_op_type == 'Adjacency':
            E = graph_edit_matrix(A, v)
            H = A - E + self._diagonal_scale * torch.diag(v.squeeze())
        if self._spectral_op_type == 'SquaredProjectedHamiltonian':
            H_unprojected = hamiltonian(A, v, self._diagonal_scale)
            x = w / w.norm()
            H_projected = H_unprojected @ (torch.eye(H_unprojected.shape[0]) - x @ x.T)
            H = H_projected.T @ H_projected
        return H

    def init_params(self):
        pass

    @property
    def output_dim(self):
        return self._n_eigs

    @property
    def embedding_type(self):
        return f"{self._spectral_op_type} eigs"


class NeuralSEDEmbeddingNetwork(BaseGraphEmbeddingNetwork):
    pass
