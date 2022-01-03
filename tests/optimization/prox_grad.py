import torch
import cvxpy as cp
import numpy as np
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL2
from tests.optimization.conftest import BaseTestOptimization
import pytest


class TestProx(BaseTestOptimization):
    @pytest.mark.parametrize(
        "nesterov",
        (False, True)
    )
    def test_prox_grad(self, nesterov):
        # define some simple problem, e.g. min ||Ax-b||^2+mu*||x||_2
        A = torch.randn([5, 10])
        x_gt = torch.randn(10)
        y = A @ x_gt + 0.01 * torch.randn(A.shape[0])
        l2prox = ProxL2()
        x = torch.zeros(10, requires_grad=True)
        maxiter = 5000
        mu = 0.2
        lamb = 0.1
        lr = 0.0001
        momentum = 0.1
        dampening = 0
        pgm = PGM(params=[x],
                  proxs=[l2prox],
                  lr=lr,
                  momentum=momentum,
                  dampening=dampening,
                  nesterov=nesterov)
        loss_fuction = lambda x: torch.norm(A @ x - y) ** 2 + mu * torch.norm(x)
        loss_vals = []
        for i in range(maxiter):
            loss = loss_fuction(x)
            loss.backward()
            pgm.step(lamb=lamb)
            loss_vals.append(loss.item())
        print("done")

        # compare to cvx
        x_cvx = cp.Variable(x.shape)
        prob = cp.Problem(cp.Minimize(
            cp.sum_squares(A @ x_cvx - y) + mu * cp.norm2(x_cvx)))
        prob.solve()
        x_cvx_opt = torch.tensor(x_cvx.value).type(torch.FloatTensor)
        error_x = torch.norm(x.detach() - x_cvx_opt).type(torch.FloatTensor)
        error_loss = torch.norm(
            loss_fuction(x.detach()) - loss_fuction(x_cvx_opt)).type(torch.FloatTensor)

        assert torch.isclose(error_x, torch.zeros(1), 1e-3, 1e-2)
        assert torch.isclose(error_loss, torch.zeros(1), 1e-3, 1e-2)
