import torch
import numpy as np
from numpy import histogram
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxSymmetricCenteredMatrix, ProxId
from tests.optimization.util import double_centering, set_diag_zero
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from skimage.filters.thresholding import threshold_otsu

import kmeans1d

x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
k = 4

clusters, centroids = kmeans1d.cluster(x, k)

print(clusters)   # [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
print(centroids)  # [-50.0, 4.1, 94.0, 200.5]

def block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.1):
    p11 = set_diag_zero(p_parts * torch.ones(n1, n1))

    p22 = set_diag_zero(p_parts * torch.ones(n2, n2))

    p12 = p_off * torch.ones(n1, n2)

    p = torch.zeros([n, n])
    p[0:n1, 0:n1] = p11
    p[0:n1, n1:n] = p12
    p[n1:n, n1:n] = p22
    p[n1:n, 0:n1] = p12.T

    return p


class SubgraphIsomorphismSolver:

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

        maxiter = 100
        mu_l21 = 1
        mu_MS = 0.3
        mu_trace = 0

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
            self.spectrum_alignment_term(ref, L, E, v) \
            + mu_MS * self.MSreg(L, E, v) + mu_trace * self.trace_reg(E, n)

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

    @staticmethod
    def spectrum_alignment_term(ref_spectrum, L, E, v):
        k = ref_spectrum.shape[0]
        Hamiltonian = L + E + torch.diag(v)
        spectrum = torch.linalg.eigvalsh(Hamiltonian)
        loss = torch.norm(spectrum[0:k] - ref_spectrum) ** 2
        return loss

    @staticmethod
    def MSreg(L, E, v):
        return v.T @ (L + E) @ v

    @staticmethod
    def trace_reg(E, n):
        return (torch.trace(E) - n) ** 2

    def plots(self):
        plt.loglog(self.loss_vals, 'b')
        plt.title('full loss')
        plt.xlabel('iter')
        plt.show()

        ax = plt.subplot()
        im = ax.imshow(self.E)
        divider = make_axes_locatable(ax)
        ax.set_title('E')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        ax = plt.subplot()
        L_edited = self.E + self.L.numpy()
        im = ax.imshow(L_edited)
        divider = make_axes_locatable(ax)
        ax.set_title('L+E')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        ax = plt.subplot()
        A_edited = -set_diag_zero(self.E + self.L.numpy())
        im = ax.imshow(A_edited)
        divider = make_axes_locatable(ax)
        ax.set_title('A edited')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        plt.plot(np.sort(self.v), 'xr')
        plt.title('v')
        plt.show()

        ax = plt.subplot()
        im = ax.imshow(np.diag(self.v))
        divider = make_axes_locatable(ax)
        ax.set_title('diag(v)')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        ax = plt.subplot()
        # _, v_clustered = kmeans2(self.v, 2, minit='points')
        v = self.v - np.min(self.v)
        v = v / np.max(v)
        threshold = threshold_otsu(v, nbins=10)
        v_otsu = (v > threshold).astype(float)
        im = ax.imshow(np.diag(v_otsu))
        divider = make_axes_locatable(ax)
        ax.set_title('diag(v_otsu)')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        ax = plt.subplot()
        # _, v_clustered = kmeans2(self.v, 2, minit='points')
        v = self.v - np.min(self.v)
        v = v / np.max(v)
        v_clustered, centroids = kmeans1d.cluster(v, k=2)
        im = ax.imshow(np.diag(v_clustered))
        divider = make_axes_locatable(ax)
        ax.set_title('diag(v_kmeans)')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        plt.plot(self.ref_spectrum.numpy(), 'og')
        plt.plot(self.spectrum, 'xr')
        plt.title('ref spect vs spect')
        plt.show()


if __name__ == '__main__':
    # torch.manual_seed(12)

    n1 = 5
    n2 = 15
    n = n1 + n2
    p = block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.05)

    A = torch.tril(torch.bernoulli(p))
    A = (A + A.T)
    D = torch.diag(A.sum(dim=1))
    L = D - A

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n1, 0:n1]
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)
    subgraph_isomorphism_solver = SubgraphIsomorphismSolver(L, ref_spectrum)
    v, E = subgraph_isomorphism_solver.solve()
    subgraph_isomorphism_solver.plots()
