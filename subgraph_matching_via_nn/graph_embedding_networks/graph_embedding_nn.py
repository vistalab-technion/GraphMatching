import abc
from abc import abstractmethod, ABC

import torch
from torch import nn, cat

from subgraph_matching_via_nn.utils.graph_utils import hamiltonian, graph_edit_matrix


class BaseGraphEmbeddingNetwork(nn.Module, ABC):
    def __init__(self, indicator_scale=1):
        super().__init__()
        self._indicator_scale = indicator_scale

    def forward(self, A, w, params: dict = None, is_use_last_args: bool = False):
        pass

    @property
    @abstractmethod
    def output_dim(self):
        pass

    @property
    @abstractmethod
    def embedding_type(self):
        pass


class GraphsBatchEmbeddingNetwork(BaseGraphEmbeddingNetwork, abc.ABC):
    @abstractmethod
    def forward_graphs(self, batch_graph):
        pass


class MomentEmbeddingNetwork(BaseGraphEmbeddingNetwork):
    def __init__(self, n_moments, moments_type='standardized', indicator_scale=1):
        super().__init__(indicator_scale=indicator_scale)
        self._moments_type = moments_type
        self._n_moments = n_moments

    def forward(self, A, w, params: dict = None, is_use_last_args: bool = False):
        if self._moments_type == 'standardized_central':
            embedding = self.compute_standardized_central_moments(w, A, self._n_moments,
                                                                  self._indicator_scale)
        elif self._moments_type == 'standardized_raw':
            embedding = self.compute_standardized_raw_moments(w, A, self._n_moments,
                                                              self._indicator_scale)
        elif self._moments_type == 'raw':
            embedding = self.compute_raw_moments(w, A, self._n_moments,
                                                 self._indicator_scale)
        elif self._moments_type == 'central':
            embedding = self.compute_central_moments(w, A, self._n_moments,
                                                     self._indicator_scale)
        else:
            raise Exception("unknown moments type")
        return embedding

    @staticmethod
    def compute_standardized_central_moments(w, A, n_moments, indicator_scale=1):
        scaled_w = indicator_scale * w
        mean = w.T @ A @ scaled_w
        var = w.T @ ((A @ scaled_w - mean) ** 2)
        moments = []
        # for standardized moments, mom1 == 0, mom2 == 1 so no need to append them
        if n_moments > 2:
            for k in range(3, n_moments + 1):
                mom = w.T @ (((A @ scaled_w - mean) / (var ** 0.5)) ** k)
                moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_standardized_raw_moments(w, A, n_moments, indicator_scale=1):
        scaled_w = indicator_scale * w
        var = w.T @ ((A @ scaled_w) ** 2)
        moments = []
        # for raw standardized moments, mom2 == 1 so no need to append it
        if n_moments > 2:
            for k in range(1, n_moments + 1):
                if k != 2:
                    mom = (w.T @ (((A @ scaled_w) / (var ** 0.5)) ** k))
                    moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_central_moments(w, A, n_moments, indicator_scale=1):
        scaled_w = indicator_scale * w
        mean = w.T @ A @ scaled_w
        moments = []
        if n_moments >= 2:
            for k in range(2, n_moments + 1):
                mom = w.T @ ((A @ scaled_w - mean) ** k)
                moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_raw_moments(w, A, n_moments, indicator_scale=1):

        moments = []
        sclaed_w = indicator_scale * w
        for k in range(1, n_moments + 1):
            mom = w.T @ ((A @ sclaed_w) ** k)
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
        super().__init__(indicator_scale)
        self._spectral_op_type = spectral_op_type
        self._n_eigs = n_eigs
        self._diagonal_scale = diagonal_scale
        self._zero_eig_scale = zero_eig_scale

    def forward(self, A, w, params: dict = None, is_use_last_args: bool = False):
        H = self.spectral_operator(A, w)
        evals, evecs = torch.linalg.eigh(H)
        embedding = evals[:self._n_eigs]

        embedding_clone = embedding.clone()
        embedding_clone[0] = self._zero_eig_scale * embedding[0]
        return embedding_clone

    def spectral_operator(self, A, w):
        v = 1 - self._indicator_scale * w
        x = w / w.norm()
        v = v - torch.dot(v.squeeze(), x.squeeze()) * x
        if self._spectral_op_type == 'Laplacian':
            H = hamiltonian(A, v, self._diagonal_scale)
        if self._spectral_op_type == 'Adjacency':
            E = graph_edit_matrix(A, v)
            H = A - E + self._diagonal_scale * torch.diag(v.squeeze())
        if self._spectral_op_type == 'SquaredProjectedHamiltonian':
            H_unprojected = hamiltonian(A, v, self._diagonal_scale)
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
