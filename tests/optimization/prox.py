import pytest
from optimization.prox.prox import ProxL2, ProxL21, ProxSymmetricCenteredMatrix, \
    ProxL21ForSymmetricCenteredMatrix, l21_prox_loss
from tests.optimization.conftest import BaseTestOptimization
import torch


class TestProx(BaseTestOptimization):

    def test_l2prox(self, n=5, lamb=0.1):
        z = torch.rand(n, 1)
        my_l2_prox = ProxL2()
        my_l2_prox_cvx = ProxL2(solver="cvx")
        z_prox = my_l2_prox(z=z, lamb=lamb)
        z_prox_cvx = my_l2_prox_cvx(z=z, lamb=lamb)
        print(f"l2 prox {z_prox}")
        print(f"l2 prox  cvx {z_prox_cvx}")
        assert torch.isclose(torch.norm(z_prox - z_prox_cvx).type(torch.FloatTensor),
                             torch.zeros(1),
                             self.atol, self.rtol)

    def test_l21_prox(self, n=5, lamb=0.1):
        # test l21 prox
        Z = torch.rand(n, n)
        my_l21prox = ProxL21()
        Z_prox = my_l21prox(z=Z, lamb=lamb)
        my_l21prox_cvx = ProxL21(solver="cvx")
        Z_prox_cvx = my_l21prox_cvx(z=Z, lamb=lamb)
        print(f"l21 prox {Z_prox}")
        print(f"l21 prox cvx{Z_prox_cvx}")
        assert torch.isclose(torch.norm(Z_prox - Z_prox_cvx).type(torch.FloatTensor),
                             torch.zeros(1),
                             1e-3, 1e-2)

    def test_symm_centered_prox(self, n=5):
        # test symmetric centered prox
        Z = torch.rand(n, n)
        my_symm_centered_prox = ProxSymmetricCenteredMatrix()
        Z_prox = my_symm_centered_prox(z=Z)
        # test symmetric centered prox with cvx
        my_symm_centered_prox_cvx = ProxSymmetricCenteredMatrix(solver="cvx")
        Z_prox_cvx = my_symm_centered_prox_cvx(z=Z)
        print(f"symmetric centered {Z_prox}")
        print(f"symmetric centered {Z_prox_cvx}")
        assert torch.isclose(torch.norm(Z_prox - Z_prox_cvx).type(torch.FloatTensor),
                             torch.zeros(1),
                             1e-3, 1e-2)

    @pytest.mark.skip
    def test_l21_symm_centered_prox(self, n=5, lamb=0.1):
        Z = torch.rand(n, n)
        rho = 0.001
        my_l21_for_symmetric_centered_matrix_prox = ProxL21ForSymmetricCenteredMatrix(
            rho=rho, maxiter=50000, tol=1e-6)
        Z_prox = my_l21_for_symmetric_centered_matrix_prox(z=Z.type(torch.DoubleTensor),
                                                           lamb=lamb)
        my_l21_for_symmetric_centered_matrix_prox_cvx = ProxL21ForSymmetricCenteredMatrix(
            rho=rho, maxiter=50000, solver="cvx")
        Z_prox_cvx = my_l21_for_symmetric_centered_matrix_prox_cvx(z=Z, lamb=lamb)
        print(f"l21 on symmetric centered {Z_prox}")
        print(f"l21 on symmetric centered cvx{Z_prox_cvx}")
        loss_my_solver = l21_prox_loss(Z_prox, Z, lamb)
        loss_cvx_solver = l21_prox_loss(Z_prox_cvx, Z, lamb)
        print(f"l21 on symmetric centered loss "
              f"{loss_my_solver} vs cvx loss {loss_cvx_solver}")
        assert torch.isclose(torch.norm(Z_prox - Z_prox_cvx).type(torch.FloatTensor),
                             torch.zeros(1),
                             1e-3, 1e-2)
        assert torch.isclose(loss_my_solver,
                             loss_cvx_solver,
                             1e-3, 1e-2)