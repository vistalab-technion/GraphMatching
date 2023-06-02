from torch import nn, cat


class MomentEmbeddingNetwork(nn.Module):
    def __init__(self, n_moments, moments_type='standardized'):
        super().__init__()
        self._moments_type = moments_type
        self._n_moments = n_moments

    def forward(self, A, w):
        if self._moments_type == 'standardized':
            embedding = self.compute_standardized_moments(w, A, self._n_moments)
        elif self._moments_type == 'raw':
            embedding = self.compute_raw_moments(w, A, self._n_moments)
        elif self._moments_type == 'central':
            embedding = self.compute_central_moments(w, A, self._n_moments)
        else:
            raise Exception("unknown moments type")
        return embedding

    @staticmethod
    def compute_standardized_moments(w, A, n_moments):
        mean = w.T @ A @ w
        var = w.T @ ((A @ w - mean) ** 2)
        moments = []
        # for standardized moments, mom1 = 0, mom2 = 1 so no need to append them
        if n_moments > 2:
            for k in range(3, n_moments + 1):
                mom = w.T @ ((A @ w - mean) ** k)
                moments.append(mom / (var ** (k / 2)))
        return cat(moments).squeeze()

    @staticmethod
    def compute_raw_moments(w, A, n_moments):
        mean = w.T @ A @ w
        moments = []
        if n_moments >= 2:
            for k in range(2, n_moments + 1):
                mom = w.T @ ((A @ w - mean) ** k)
                moments.append(mom)
        return cat(moments).squeeze()

    @staticmethod
    def compute_central_moments(w, A, n_moments):
        moments = []
        for k in range(1, n_moments + 1):
            mom = w.T @ ((A @ w) ** k)
            moments.append(mom)
        return cat(moments).squeeze()

    def init_params(self):
        pass
