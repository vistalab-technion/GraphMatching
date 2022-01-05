import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxSymmetricCenteredMatrix, ProxId
from tests.optimization.util import double_centering


def spectrum_alignment_term(ref_spectrum, L, E, v):
    k = ref_spectrum.shape[0]
    L_edited = L + E + torch.diag(v)
    spectrum = torch.linalg.eigvalsh(L_edited)
    loss = torch.norm(spectrum[0:k] - ref_spectrum) ** 2
    return loss


def MSreg(L, E, v):
    return v.T @ (L + E) @ v


def dc(x):
    ProxSymmetricCenteredMatrix(x)


class SubgrapIsomorphismSolver():

    def __init__(self, L, ref_spectrum):
        self.L = L
        self.ref_spectrum = ref_spectrum

    def solve(self):
        L = self.L
        ref_spectrum = self.ref_spectrum
        n = L.shape[0]
        l21_symm_centered_prox = ProxL21ForSymmetricCenteredMatrix(solver="cvx")
        id_prox = ProxId()

        # init
        v = torch.zeros(n, requires_grad=True, dtype=torch.float64)
        E = torch.zeros([n, n], dtype=torch.float64)
        E = double_centering(0.5 * (E + E.T)).requires_grad_()

        maxiter = 1000
        mu_l21 = 1
        mu_MS = 0# 0.1
        # s = torch.linalg.svdvals(A)
        # lr = 1 / (1.1 * s[0] ** 2)
        lr = 0.2
        lamb = mu_l21 * lr  # This setting is important!
        momentum = 0.1
        dampening = 0
        pgm = PGM(params=[{'params': v}, {'params': E}],
                  proxs=[id_prox, l21_symm_centered_prox],
                  lr=lr,
                  momentum=momentum,
                  dampening=dampening,
                  nesterov=False)
        smooth_loss_fuction = lambda ref, L, E, v: \
            spectrum_alignment_term(ref, L, E, v) + mu_MS * MSreg(L, E, v)

        non_smooth_loss_function = lambda E: mu_l21 * l21(E)
        full_loss_function = lambda ref, L, E, v: \
            smooth_loss_fuction(ref, L, E, v) + non_smooth_loss_function(E)

        loss_vals = []
        for i in tqdm(range(maxiter)):
            pgm.zero_grad()
            loss = smooth_loss_fuction(ref_spectrum, L, E, v)
            loss.backward()
            pgm.step(lamb=lamb)
            loss_vals.append(
                full_loss_function(ref_spectrum, L, E.detach(), v.detach()))
        print("done")
        L_edited = L + E.detach() + torch.diag(v.detach())
        spectrum = torch.linalg.eigvalsh(L_edited)
        k = ref_spectrum.shape[0]

        self.loss_vals = loss_vals
        self.E = E.detach().numpy()
        self.v = v.detach().numpy()
        self.spectrum = spectrum.detach().numpy()
        # self.plots()

        print(f"v= {v}")
        print(f"E= {E}")
        print(f"lambda= {spectrum[0:k]}")
        print(f"lambda*= {ref_spectrum}")
        print(f"||lambda-lambda*|| = {torch.norm(spectrum[0:k] - ref_spectrum)}")
        return v, E

    def plots(self):
        plt.loglog(self.loss_vals, 'b')
        plt.title('full loss')
        plt.xlabel('iter')
        plt.show()

        plt.imshow(self.E)
        plt.title('E')
        plt.show()

        plt.plot(np.sort(self.v), 'r')
        plt.title('v')
        plt.show()

        plt.plot(self.ref_spectrum.numpy(), 'og')
        plt.plot(self.spectrum, 'xr')
        plt.title('ref spect vs spect')
        plt.show()

        plt.plot(self.ref_spectrum.numpy(), 'og')
        plt.plot(self.spectrum, 'xr')
        plt.title('ref spect vs spect')
        plt.show()


if __name__ == '__main__':
    torch.manual_seed(0)

    n = 20
    A = torch.randint(2, [n, n])
    A = 0.5 * (A + A.T)
    D = torch.diag(A.sum(dim=1))
    L = D - A

    A_sub = A[1:5, 1:5]
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)
    subgraph_isomorphism_solver = SubgrapIsomorphismSolver(L, ref_spectrum)
    v, E = subgraph_isomorphism_solver.solve()
    subgraph_isomorphism_solver.plots()
