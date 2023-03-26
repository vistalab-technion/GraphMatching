import torch
import numpy as np
from numpy import histogram
from torch import tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from optimization.algs.power_method import QRPowerMethod
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxL21ForSymmCentdMatrixAndInequality, ProxSymmetricCenteredMatrix, ProxId, \
    ProxNonNeg
from problem.base import BaseSubgraphIsomorphismSolver, lap_from_adj, \
    block_stochastic_graph, E_from_v
from tests.optimization.util import double_centering, set_diag_zero
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from skimage.filters.thresholding import threshold_otsu
import networkx as nx
import kmeans1d
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from time import sleep
from scipy.linalg import qr, pinv
from sklearn.cluster import SpectralClustering


class ProximalSubgraphIsomorphismSolver(BaseSubgraphIsomorphismSolver):

    def __init__(self, A, ref_spectrum, problem_params, solver_params,
                 save_loss_terms=True):

        """
        Proximal algorithm solver for subgraph spectral matching.

        :param A: Adjacency matrix of graph
        :param ref_spectrum: spectrum of reference graph (i.e., the subgraph)
        :param problem_params: parameters for the algorithm and solver. For example:
            params =
            {'maxiter': 100,
              'show_iter': 10,
              'mu_spectral': 1,
              'mu_l21': 1,
              'mu_MS': 1,
              'mu_split': 1,
              'mu_trace': 0.0,
              'lr': 0.02,
              'v_prox': ProxNonNeg(),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                                trace_upper_bound=
                                                                1.1*torch.trace(L)),
              'trace_val': 0
              }
        :solver_params: parameters for solver
        :param save_loss_terms: flag for saving the individual loss terms
        """

        super().__init__(A=A,
                         ref_spectrum=ref_spectrum,
                         problem_params=problem_params,
                         solver_params=solver_params)

        self.spectrum_alignment_terms = []
        self.MS_reg_terms = []
        self.trace_reg_terms = []
        self.graph_split_terms = []
        self.l21_terms = []
        self.loss_vals = []
        self.save_individual_losses = save_loss_terms
        # init
        self.set_init()

    def set_optim(self, v, E):
        lamb = self.mu_l21 * self.lr  # This setting is important!

        if self.train_E and self.train_v:
            optim_params = [{'params': v}, {'params': E}]
            proxs = [self.v_prox, self.E_prox]
        elif not self.train_E and self.train_v:
            optim_params = [{'params': v}]
            proxs = [self.v_prox]
        elif self.train_E and not self.train_v:
            optim_params = [{'params': E}]
            proxs = [self.E_prox]

        v.requires_grad = self.train_v
        E.requires_grad = self.train_E
        pgm = PGM(params=optim_params,
                  proxs=proxs,
                  lr=self.lr)
        return pgm, lamb

    def solve(self,
              max_outer_iters=10,
              max_inner_iters=10,
              show_iter=10,
              verbose=False):
        outer_iter_counter = 0
        converged_outer = False
        while not converged_outer:
            v, _ = self._solve(maxiter_inner=max_inner_iters,
                               show_iter=show_iter,
                               verbose=verbose)
            E, _ = E_from_v(self.v.detach(), self.A)
            self.set_init(E0=E, v0=v)
            self.v = v
            self.E = E
            outer_iter_counter += 1
            converged_outer = self._check_convergence(self.v.detach(), self.a_tol)
            converged_outer = converged_outer or (outer_iter_counter >= max_outer_iters)
        return v, E

    def _solve(self, maxiter_inner=100, show_iter=10, verbose=False):
        L = self.L

        # Q, R = qr(L.numpy())
        # self.P = torch.eye(L.shape[0]) - torch.tensor(Q @ Q.T)

        ref_spectrum = self.ref_spectrum
        n = L.shape[0]

        # init
        v = self.v
        E = self.E

        # for linear inverse problems this is the optimal setting for the step size
        # s = torch.linalg.svdvals(A)
        # lr = 1 / (1.1 * s[0] ** 2)

        pgm, lamb = self.set_optim(v, E)

        full_loss_function = lambda ref, L, E, v: \
            self.smooth_loss_function(ref, L, E, v, False) \
            + self.non_smooth_loss_function(E)

        loss_vals = []

        groups = {'loss': ['loss']}
        plotlosses = PlotLosses(groups=groups, outputs=[MatplotlibPlot()])
        # converged_inner = False
        iter_count = 0
        for i in tqdm(range(maxiter_inner)):
            v_prev = self.v.detach()
            pgm.zero_grad()
            loss = \
                self.smooth_loss_function(ref_spectrum, L, E, v,
                                          self.save_individual_losses)
            loss_vals.append(
                full_loss_function(ref_spectrum, L, E.detach(), v.detach()))
            loss.backward()
            pgm.step(lamb=lamb)
            if (iter_count + 1) % show_iter == 0:
                self.plot_loss(plotlosses, loss_vals[-1])
            iter_count += 1
            converged_inner = self._check_convergence(v_prev=v_prev,
                                                      v=self.v.detach(),
                                                      r_tol=self.r_tol,
                                                      a_tol=self.a_tol)
            converged_inner = converged_inner or (iter_count >= maxiter_inner)
            if converged_inner:
                # Yes it's bad practice, but otherwise the progress bar won't update
                break
        print("done")
        L_edited = L + E.detach() + torch.diag(v.detach())
        spectrum = torch.linalg.eigvalsh(L_edited)
        k = ref_spectrum.shape[0]

        self.loss_vals = [*self.loss_vals, *loss_vals]
        self.E = E
        self.v = v
        self.spectrum = spectrum.detach().numpy()
        # self.plots()

        if verbose:
            print(f"v= {v}")
            print(f"E= {E}")
            print(f"lambda= {spectrum[0:k]}")
            print(f"lambda*= {ref_spectrum}")
            print(f"||lambda-lambda*|| = {torch.norm(spectrum[0:k] - ref_spectrum)}")
        return v, E

    def _check_convergence(self, v, a_tol, v_prev=None, r_tol=None):
        # Todo: change conditions to follow kkt
        v_binary, E_binary = self.threshold(v_np=v.detach().numpy(),
                                            threshold_algo=self.threshold_algo)
        eig_max = torch.max(self.ref_spectrum).numpy()
        c = np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
        loss = self.smooth_loss_function(self.ref_spectrum, self.L,
                                         E_binary.detach(),
                                         c * v_binary.detach(), False)
        condition1 = loss < a_tol
        if r_tol is not None:
            condition2 = (torch.norm(v - v_prev) / torch.norm(v)) < r_tol
        else:
            condition2 = False
        return condition1 or condition2

    def plot(self, plots):
        """
        produces various plots

        :param plots: flags for which plots to produce.
        The following plots are supported:
                plots = {'full_loss': True,
                        'E': True,
                        'v': True,
                        'diag(v)': True,
                        'v_otsu': True,
                        'v_kmeans': True,
                        'A edited': True,
                        'L+E': True,
                        'ref spect vs spect': True,
                        'individual loss terms': True}
        """
        self._plot(plots)

        if plots['individual loss terms']:
            font_color = "r"
            # Create two subplots and unpack the output array immediately
            fig, axes = plt.subplots(nrows=3, ncols=2)
            ax = axes.flat
            ax[0].loglog(self.spectrum_alignment_terms)
            ax[0].set_ylabel('loss', c=font_color)
            ax[0].set_xlabel('iteration', c=font_color)
            ax[0].set_title(f'spectral alignment, mu = {self.mu_spectral}',
                            c=font_color)

            ax[1].loglog(self.MS_reg_terms)
            ax[1].set_ylabel('loss', c=font_color)
            ax[1].set_xlabel('iteration', c=font_color)
            ax[1].set_title(f'MS reg, mu={self.mu_MS}', c=font_color)

            ax[2].loglog(self.graph_split_terms)
            ax[2].set_ylabel('loss', c=font_color)
            ax[2].set_xlabel('iteration', c=font_color)
            ax[2].set_title(f'graph split, mu={self.mu_split}', c=font_color)

            ax[3].loglog(self.loss_vals)
            ax[3].set_ylabel('loss', c=font_color)
            ax[3].set_xlabel('iteration', c=font_color)
            ax[3].set_title('full loss', c=font_color)

            ax[4].loglog(self.l21_terms)
            ax[4].set_ylabel('loss', c=font_color)
            ax[4].set_xlabel('iteration', c=font_color)
            ax[4].set_title(f'l21 reg, mu = {self.mu_l21}', c=font_color)

            ax[5].loglog(self.trace_reg_terms)
            ax[5].set_ylabel('loss', c=font_color)
            ax[5].set_xlabel('iteration', c=font_color)
            ax[5].set_title(f'trace reg, mu = {self.mu_trace}', c=font_color)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5,
                                hspace=2)
            plt.show()


class QRPMSubgraphIsomorphismSolver(BaseSubgraphIsomorphismSolver):

    def __init__(self, A, ref_spectrum, problem_params, solver_params,
                 save_loss_terms=True):

        """
        QR Power Method algorithm for subgraph spectral matching.

        :param A: Adjacency matrix of graph
        :param ref_spectrum: spectrum of reference graph (i.e., the subgraph)
        :param problem_params: parameters for the algorithm and solver. For example:
            params =
            {'maxiter': 100,
              'show_iter': 10,
              'mu_spectral': 1,
              'mu_l21': 1,
              'mu_MS': 1,
              'mu_split': 1,
              'mu_trace': 0.0,
              'lr': 0.02,
              'v_prox': ProxNonNeg(),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                                trace_upper_bound=
                                                                1.1*torch.trace(L)),
              'trace_val': 0
              }
        :solver_params: parameters for solver
        :param save_loss_terms: flag for saving the individual loss terms
        """

        super().__init__(A=A,
                         ref_spectrum=ref_spectrum,
                         problem_params=problem_params,
                         solver_params=solver_params)

        self.loss_vals = []
        self.save_individual_losses = save_loss_terms
        # init
        self.set_init()

    def set_init(self, v0=None, E0=None):
        n = self.L.shape[0]
        if v0 is None:
            eig_max = torch.max(self.ref_spectrum)
            c = np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
            v0 = (c / np.sqrt(n)) * np.ones(n)
            self.V = torch.eye(n)
            self.V[:,0] = torch.tensor(v0, requires_grad=self.train_v, dtype=torch.float64)
        else:
            self.V = torch.eye(n)
            self.V[:,0] = v0

        if E0 is None:
            # init
            E = torch.zeros([n, n], dtype=torch.float64)
            self.E = \
                double_centering(0.5 * (E + E.T)).requires_grad_(
                    requires_grad=self.train_E)
        else:
            self.E = E0

        self.E.requires_grad = self.train_E
        self.v.requires_grad = self.train_v

    def set_optim(self, v, E):
        lamb = self.mu_l21 * self.lr  # This setting is important!

        if self.train_E and self.train_v:
            optim_params = [{'params': v}, {'params': E}]
            proxs = [self.v_prox, self.E_prox]
        elif not self.train_E and self.train_v:
            optim_params = [{'params': v}]
            proxs = [self.v_prox]
        elif self.train_E and not self.train_v:
            optim_params = [{'params': E}]
            proxs = [self.E_prox]

        v.requires_grad = self.train_v
        E.requires_grad = self.train_E
        pgm = PGM(params=optim_params,
                  proxs=proxs,
                  lr=self.lr)
        return pgm, lamb

    def _solve(self, maxiter_inner=100, show_iter=10, verbose=False):
        L = self.L

        # Q, R = qr(L.numpy())
        # self.P = torch.eye(L.shape[0]) - torch.tensor(Q @ Q.T)

        ref_spectrum = self.ref_spectrum
        n = L.shape[0]

        # init
        v = self.v
        E = self.E

        full_loss_function = lambda ref, L, E, v: \
            self.smooth_loss_function(ref, L, E, v, False) \
            + self.non_smooth_loss_function(E)

        loss_vals = []


        groups = {'loss': ['loss']}
        plotlosses = PlotLosses(groups=groups, outputs=[MatplotlibPlot()])
        # converged_inner = False
        iter_count = 0
        L_shifted = L - torch.diag(self.ref_spectrum, diagonal=0)

        shift_matrix  = self.shift_operator(self.V)
        L, U = torch.lu(L_shifted + shift_matrix)
        for i in tqdm(range(maxiter_inner)):
            v_prev = self.v.detach()
            Y = torch.linalg.solve(L, V)
            Z = torch.linalg.solve(U, Y)
            V, _ = torch.linalg.qr(Z)
            loss = \
                self.smooth_loss_function(ref_spectrum, L, E, v,
                                          self.save_individual_losses)
            loss_vals.append(
                full_loss_function(ref_spectrum, L, E.detach(), v.detach()))
            if (iter_count + 1) % show_iter == 0:
                self.plot_loss(plotlosses, loss_vals[-1])
            iter_count += 1
            converged_inner = self._check_convergence(v_prev=v_prev,
                                                      v=self.v.detach(),
                                                      r_tol=self.r_tol,
                                                      a_tol=self.a_tol)
            converged_inner = converged_inner or (iter_count >= maxiter_inner)
            if converged_inner:
                # Yes it's bad practice, but otherwise the progress bar won't update
                break
        print("done")
        L_edited = L + E.detach() + torch.diag(v.detach())
        spectrum = torch.linalg.eigvalsh(L_edited)
        k = ref_spectrum.shape[0]

        self.loss_vals = [*self.loss_vals, *loss_vals]
        self.E = E
        self.v = v
        self.spectrum = spectrum.detach().numpy()
        # self.plots()

        if verbose:
            print(f"v= {v}")
            print(f"E= {E}")
            print(f"lambda= {spectrum[0:k]}")
            print(f"lambda*= {ref_spectrum}")
            print(f"||lambda-lambda*|| = {torch.norm(spectrum[0:k] - ref_spectrum)}")
        return v, E

    @property
    def v(self):
        return self.V[:,0]

    def plot(self, plots):
        """
        produces various plots

        :param plots: flags for which plots to produce.
        The following plots are supported:
                plots = {'full_loss': True,
                        'E': True,
                        'v': True,
                        'diag(v)': True,
                        'v_otsu': True,
                        'v_kmeans': True,
                        'A edited': True,
                        'L+E': True,
                        'ref spect vs spect': True,
                        'individual loss terms': True}
        """
        self._plot(plots)
