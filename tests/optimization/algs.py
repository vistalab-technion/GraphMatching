import torch
import cvxpy as cp
import numpy as np
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL2, ProxL1, ProxL21ForSymmetricCenteredMatrix, \
    l21
from tests.optimization.conftest import BaseTestOptimization
import pytest
import matplotlib.pyplot as plt


class TestProx(BaseTestOptimization):
    @pytest.mark.parametrize(
        "nesterov",
        (False,)
    )
    def test_prox_grad_l2(self, nesterov):
        # define some simple problem, e.g. min ||Ax-b||_2^2+mu*||x||_2
        A = torch.randn([5, 10], dtype=torch.float64)
        x_gt = torch.randn(10, dtype=torch.float64)
        y = A @ x_gt + 0.1 * torch.randn(A.shape[0])
        mu = 1

        # solve with cvx
        x_cvx = cp.Variable(x_gt.shape)
        prob = cp.Problem(cp.Minimize(
            cp.sum_squares(A @ x_cvx - y) + mu * cp.norm2(x_cvx)))
        prob.solve(abstol=1e-6)
        x_cvx_opt = torch.tensor(x_cvx.value)

        # now use our solver to compare
        l2prox = ProxL2()
        x = torch.zeros(10, requires_grad=True, dtype=torch.float64)
        maxiter = 1000
        s = torch.linalg.svdvals(A)
        lr = 1 / (2 * s[0] ** 2)
        lamb = mu * lr  # This setting is important!
        pgm = PGM(params=[x],
                  proxs=[l2prox],
                  lr=lr)
        smooth_loss_fuction = lambda x: torch.norm(A @ x - y) ** 2
        non_smooth_loss_function = lambda x: mu * torch.norm(x)
        full_loss_function = lambda x: \
            smooth_loss_fuction(x) + non_smooth_loss_function(x)
        loss_vals = []
        distance_to_opt = []
        for i in range(maxiter):
            pgm.zero_grad()
            loss = smooth_loss_fuction(x)
            loss.backward()
            pgm.step(lamb=lamb)
            loss_vals.append(full_loss_function(x.detach()))
            distance_to_opt.append(torch.norm(x.detach() - x_cvx_opt))
        print("done")
        self.plots(distance_to_opt, loss_vals)
        error_x = torch.norm(x.detach() - x_cvx_opt)
        error_loss = full_loss_function(x.detach()) - full_loss_function(x_cvx_opt)
        print(x.detach())
        print(x_cvx.value)
        print(f"||x-x_cvx|| = {error_x}")
        print(f"l-l_cvx = {error_loss}")
        assert torch.isclose(error_x, torch.zeros(1, dtype=torch.float64), 1e-3, 1e-2)
        assert torch.isclose(error_loss, torch.zeros(1, dtype=torch.float64), 1e-3,
                             1e-2)

    @pytest.mark.parametrize(
        "nesterov",
        (False,)
    )
    def test_prox_grad_l1(self, nesterov):
        # define some simple problem, e.g. min ||Ax-b||_2^2+mu*||x||_1
        A = torch.randn([5, 10], dtype=torch.float64)
        x_gt = torch.randn(10, dtype=torch.float64)
        y = A @ x_gt + 0.1 * torch.randn(A.shape[0])
        mu = 1

        # solve with cvx
        x_cvx = cp.Variable(x_gt.shape)
        prob = cp.Problem(cp.Minimize(
            cp.sum_squares(A @ x_cvx - y) + mu * cp.norm1(x_cvx)))
        prob.solve(solver='ECOS', abstol=1e-3)
        x_cvx_opt = torch.tensor(x_cvx.value)

        # now use our solver to compare
        l1prox = ProxL1(solver="cvx")
        x = torch.zeros(10, requires_grad=True, dtype=torch.float64)
        w = torch.zeros(10, requires_grad=False, dtype=torch.float64)
        maxiter = 1000
        s = torch.linalg.svdvals(A)
        lr = 1 / (2 * s[0] ** 2)
        lamb = mu * lr  # This setting is important!
        pgm = PGM(params=[x],
                  proxs=[l1prox],
                  lr=lr)
        smooth_loss_fuction = lambda x: torch.norm(A @ x - y) ** 2
        non_smooth_loss_function = lambda x: mu * torch.norm(x, p=1)
        full_loss_function = lambda x: \
            smooth_loss_fuction(x) + non_smooth_loss_function(x)
        loss_vals = []
        distance_to_opt = []
        for i in range(maxiter):
            pgm.zero_grad()
            print(f'iter {i}')
            loss = smooth_loss_fuction(x)
            loss.backward()
            pgm.step(lamb=lamb)
            loss_vals.append(full_loss_function(x.detach()))
            distance_to_opt.append(torch.norm(x.detach() - x_cvx_opt))
        print("done")
        self.plots(distance_to_opt, loss_vals)
        error_x = torch.norm(x.detach() - x_cvx_opt)
        error_w = torch.norm(w - x_cvx_opt)
        error_loss = full_loss_function(x.detach()) - full_loss_function(x_cvx_opt)
        print(x.detach())
        print(x_cvx.value)
        print(f"||x-x_cvx|| = {error_x}")
        print(f"||w-x_cvx|| = {error_w}")
        print(f"l-l_cvx = {error_loss}")
        assert torch.isclose(error_x, torch.zeros(1, dtype=torch.float64), 1e-3, 1e-2)
        assert torch.isclose(error_loss, torch.zeros(1, dtype=torch.float64), 1e-3,
                             1e-2)

    def test_prox_grad_l21_symmetric_centered(self):
        # define some simple problem, e.g.
        # min ||X-Y||^2 +mu||X||_21 s.t. X1=1, X=X'
        n = 10
        y = 0.1 * torch.randn([n, n])
        mu = 100

        # solve with cvx
        x_cvx = cp.Variable(y.shape, symmetric=True)
        objective = cp.sum_squares(y - x_cvx) + mu * cp.mixed_norm(x_cvx, p=2, q=1)
        prob = cp.Problem(objective=cp.Minimize(objective),
                          constraints=[x_cvx @ np.ones(n) == np.zeros(n)])

        prob.solve(solver='ECOS', abstol=1e-3)
        x_cvx_opt = torch.tensor(x_cvx.value)

        # now use our solver to compare
        l21prox_symmetric_centered = ProxL21ForSymmetricCenteredMatrix(solver="cvx")
        x = torch.ones([10, 10], requires_grad=True, dtype=torch.float64)
        maxiter = 1000
        s = [1]
        lr = 1 / (2 * s[0] ** 2)
        lamb = mu * lr  # This setting is important!
        pgm = PGM(params=[x],
                  proxs=[l21prox_symmetric_centered],
                  lr=lr)
        smooth_loss_fuction = lambda x: torch.norm(x - y) ** 2
        non_smooth_loss_function = lambda x: mu * l21(x)
        full_loss_function = lambda x: \
            smooth_loss_fuction(x) + non_smooth_loss_function(x)
        loss_vals = []
        distance_to_opt = []
        for i in range(maxiter):
            pgm.zero_grad()
            #print(f'iter {i}')
            loss = smooth_loss_fuction(x)
            loss.backward()
            pgm.step(lamb=lamb)
            loss_vals.append(full_loss_function(x.detach()))
            distance_to_opt.append(torch.norm(x.detach() - x_cvx_opt))
        print("done")
        self.plots(distance_to_opt, loss_vals)
        error_x = torch.norm(x.detach() - x_cvx_opt)
        error_loss = full_loss_function(x.detach()) - full_loss_function(x_cvx_opt)
        non_smooth_loss_val = non_smooth_loss_function(x.detach())
        print(x.detach())
        print(x_cvx.value)
        print(f"non-smooth loss val = {non_smooth_loss_val}")
        print(f"||x-x_cvx|| = {error_x}")
        print(f"l-l_cvx = {error_loss}")
        assert torch.isclose(error_x, torch.zeros(1, dtype=torch.float64), 1e-3, 1e-2)
        assert torch.isclose(error_loss, torch.zeros(1, dtype=torch.float64), 1e-3,
                             1e-2)

    @staticmethod
    def plots(distance_to_opt, loss_vals):
        plt.loglog(distance_to_opt, 'r')
        plt.title('distance to opt (opt obtained by cvx)')
        plt.xlabel('iter')
        plt.ylabel('distance')
        plt.show()
        plt.loglog(loss_vals, 'b')
        plt.title('full loss')
        plt.xlabel('iter')
        plt.show()
