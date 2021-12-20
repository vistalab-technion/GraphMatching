from abc import ABC, abstractmethod
from typing import Optional
from logging import exception
import numpy as np
from torch import stack

import cvxpy as cp
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def l21(z: Tensor):
    l21norm = sum([torch.norm(z[:, j], p=2) for j in range(z.shape[1])])
    return l21norm


def l21_prox_loss(x, z, lamb):
    return (1 / (2 * lamb)) * torch.norm(x - z, p='fro') ** 2 + l21(x)


@abstractmethod
class ProxParams:
    def __init__(self, lamb):
        self.lamb = lamb


class ProxBase(ABC):
    """
    base class for proximal operators
    """

    def __init__(self, solver=""):
        self.solver = solver

    @abstractmethod
    def __call__(self, z: Tensor, lamb: float):
        pass


class ProxL2(ProxBase):
    """
     class for l2 proximal operator
    """

    def __init__(self, solver=""):
        super().__init__(solver=solver)

    def __call__(self, z: Tensor, lamb: float):
        norm_z = torch.norm(z, p="fro")
        if norm_z <= lamb:
            return z
        else:
            return (1 - lamb / norm_z) * z


class ProxL21(ProxBase):
    """
     class for l2 proximal operator
    """

    def __init__(self, solver=""):
        super().__init__(solver=solver)
        self.prox_l2 = ProxL2()

    def __call__(self, z: Tensor, lamb: float):
        if self.solver == "cvx":
            return self._cvx_prox(z=z, lamb=lamb)
        else:
            return self._prox(z=z, lamb=lamb)

    def _prox(self, z: Tensor, lamb: float):
        z_list = []
        for j in range(z.shape[1]):
            z_list.append(self.prox_l2(z=z[:, j], lamb=lamb))
        return stack(z_list, dim=1)


class ProxSymmetricCenteredMatrix(ProxBase):
    """
     class for projection on set of symmetric and centered
    """

    def __init__(self, solver=""):
        super().__init__(solver=solver)

    @staticmethod
    def __double_centering(x):
        n = x.shape[0]
        row_sum = torch.sum(x, dim=0).unsqueeze(0)
        col_sum = torch.sum(x, dim=1).unsqueeze(1)
        all_sum = sum(col_sum)
        dc_x = x - (1 / n) * (
                Tensor.repeat(row_sum, n, 1) + Tensor.repeat(col_sum, 1, n)) + (
                       1 / n ** 2) * all_sum
        return dc_x

    def __call__(self, z: Tensor, lamb: float = 0):
        if self.solver == "cvx":
            return self._cvx_prox(z=z, lamb=lamb)
        else:
            return self._prox(z=z, lamb=lamb)

    def _prox(self, z: Tensor, lamb: float = 0):
        if z.shape[0] != z.shape[1]:
            raise exception('matrix has to be square')
        z_symm = 0.5 * (z + z.T)
        return self.__double_centering(z_symm)


class ProxL21ForSymmetricCenteredMatrix(ProxBase):
    """
     class for l21_for_symmetric_centered_matrix proximal operator
    """

    def __init__(self, rho: float = 1,
                 tol: float = 1e-6,
                 maxiter: int = 1000,
                 x0: Optional = None,
                 y0: Optional = None,
                 solver=""):

        super().__init__(solver=solver)

        self.rho = rho
        self.tol = tol
        self.maxiter = maxiter
        self.x0 = x0
        self.y0 = y0
        self.l21_prox = ProxL21()
        self.symmetric_centered_prox = ProxSymmetricCenteredMatrix()

    def __stopping_condition(self, x, y):
        r = torch.norm(x - y, p='fro')
        return r, r < self.tol

    def __call__(self, z: Tensor, lamb: float = 0):
        if self.solver == "cvx":
            return self._cvx_prox(z=z, lamb=lamb)
        else:
            return self._prox(z=z, lamb=lamb)

    def _prox(self, lamb: float, z: Tensor):

        converged = False
        counter = 0
        mu = 1 / (1 / lamb + self.rho)
        t = 1 / (1 + self.rho * lamb)
        nu = torch.zeros(z.shape)  # initial lagrange multiplier
        loss = []
        if self.x0 is not None:
            x = self.x0
        else:
            x = z
        if self.y0 is not None:
            y = self.y0
        else:
            y = z
        while not converged and (counter <= self.maxiter):
            w = t * z + (1 - t) * (y + nu)
            x = self.l21_prox(z=w, lamb=mu)
            y = self.symmetric_centered_prox(z=x - nu)
            nu = nu + (y - x)
            r, converged = self.__stopping_condition(x, y)
            counter += 1
            # TODO: this loss is wrong,
            #  need to change the computed loss to be
            #  the augmented Lagrangian
            loss.append((1 / (2 * lamb)) * torch.norm(x - z, p='fro') ** 2 + l21(z))
            if np.mod(counter, 10) == 0:
                # plt.loglog(loss)
                # plt.show()
                print(f'counter: {counter}, loss: {loss[-1]}, r: {r}')
        return x

    def _cvx_prox(self, lamb: float, z: Tensor):
        z = z.numpy()
        x = cp.Variable(z.shape)
        ones = np.ones([z.shape[0], 1])
        zeros = 0 * ones
        prob = cp.Problem(cp.Minimize(
            cp.mixed_norm(x, 2, 1) + (1 / (2 * lamb)) * cp.sum_squares(x - z)),
            [x @ ones == zeros,
             x == x.T])
        prob.solve()
        print(f"cvx min value {prob.value}")
        return x.value


if __name__ == '__main__':
    # test l2 prox
    n = 5
    z = torch.rand(n, 1)
    lamb = 1.0
    my_l2_prox = ProxL2()
    z_prox = my_l2_prox(z=z, lamb=lamb)
    z = z / torch.norm(z)
    z_prox = my_l2_prox(z=z, lamb=lamb)
    print(f"l2 prox {z_prox}")
    assert torch.all(z_prox == z)

    # test l21 prox
    Z = torch.rand(n, n)
    my_l21prox = ProxL21()
    Z_prox = my_l21prox(z=Z, lamb=lamb)
    print(f"l21 prox {Z_prox}")

    # test symmetric centered prox
    my_symm_centered_prox = ProxSymmetricCenteredMatrix()
    Z_prox = my_symm_centered_prox(z=Z)
    print(f"symmetric centered {Z_prox}")

    # test symmetric centered prox
    rho = 0.001
    my_l21_for_symmetric_centered_matrix_prox = ProxL21ForSymmetricCenteredMatrix(
        rho=rho, maxiter=50000, tol=1e-6)
    Z_prox = my_l21_for_symmetric_centered_matrix_prox(z=Z.type(torch.DoubleTensor), lamb=lamb)
    cvx_l21_for_symmetric_centered_matrix_prox = ProxL21ForSymmetricCenteredMatrix(
        rho=rho, maxiter=50000, solver="cvx")
    Z_prox_cvx = cvx_l21_for_symmetric_centered_matrix_prox(z=Z, lamb=lamb)
    print(f"l21 on symmetric centered {Z_prox}")
    print(f"l21 on symmetric centered cvx{Z_prox_cvx}")

    loss_my_solver = l21_prox_loss(Z_prox, Z, lamb)
    loss_cvx_solver = l21_prox_loss(torch.tensor(Z_prox_cvx), Z, lamb)
    print(f"l21 on symmetric centered loss "
          f"{loss_my_solver} vs cvx loss {loss_cvx_solver}")
