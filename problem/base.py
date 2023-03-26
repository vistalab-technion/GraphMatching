import abc
from abc import ABC
from abc import abstractmethod
from time import sleep
from typing import Optional

import kmeans1d
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import SpectralClustering
from torch import tensor, Tensor
import networkx as nx
import matplotlib.pyplot as plt

from tests.optimization.util import double_centering, set_diag_zero
from skimage.filters.thresholding import threshold_otsu

def l21(z: Tensor):
    l21norm = sum([torch.norm(z[:, j], p=2) for j in range(z.shape[1])])
    return l21norm


def lap_from_adj(A):
    return torch.diag(A.sum(axis=1)) - A
    # D_sqrt_reciprocal = torch.diag(A.sum(axis=1) **-0.5)
    # return D_sqrt_reciprocal@A@D_sqrt_reciprocal


def block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.1):
    n = n1 + n2
    p11 = set_diag_zero(p_parts * torch.ones(n1, n1))

    p22 = set_diag_zero(p_parts * torch.ones(n2, n2))

    p12 = p_off * torch.ones(n1, n2)

    p = torch.zeros([n, n])
    p[0:n1, 0:n1] = p11
    p[0:n1, n1:n] = p12
    p[n1:n, n1:n] = p22
    p[n1:n, 0:n1] = p12.T

    return p


def E_from_v(v, A):
    v_ = indicator_from_v(v)
    S = -torch.abs(v_[:, None] - v_[:, None].T) * A
    # E = torch.diag(S.sum(axis=1)) - S
    E = lap_from_adj(S)
    return E, S


def indicator_from_v(v):
    v_ = v - torch.min(v)
    if torch.max(v_) != 0:
        v_ = v_ / torch.max(v_)
    # v_ = torch.ones_like(v_,)-v_
    return v_


def indicator_from_v_np(v_np):
    v_ = v_np - np.min(v_np)
    if np.max(v_) != 0:
        v_ = v_ / np.max(v_)
    # v_ = torch.ones_like(v_,)-v_
    return v_


def edgelist_to_adjmatrix(edgeList_file):
    edge_list = np.loadtxt(edgeList_file, usecols=range(2))

    n = int(np.amax(edge_list) + 1)
    # n = int(np.amax(edge_list))
    # print(n)

    e = np.shape(edge_list)[0]

    a = np.zeros((n, n))

    # make adjacency matrix A1

    for i in range(0, e):
        n1 = int(edge_list[i, 0])  # - 1

        n2 = int(edge_list[i, 1])  # - 1

        a[n1, n2] = 1.0
        a[n2, n1] = 1.0

    return a


def plot_on_graph(A, v, E, subset_nodes, pos=None):
    """
    plots the potentials E and v on the full graph

    :param A: adjacency of full graph
    :param n_subgraph: size of subgraph
    """
    if pos is None:
        pos = nx.spring_layout(nx.from_numpy_matrix(A))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 10])
    ax = axes.flat

    vmin = np.min(v)
    vmax = np.max(v)

    G = nx.from_numpy_matrix(A)
    # pos = nx.spring_layout(G)
    # pos = nx.spring_layout(G)
    # plt.rcParams["figure.figsize"] = (20,20)

    # for edge in G.edges():

    for u, w, d in G.edges(data=True):
        d['weight'] = E[u, w]

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    cmap = plt.cm.gnuplot
    # ax = plt.subplot()
    nx.draw(G, node_color=v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
            node_size=30,
            pos=pos, ax=ax[0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    ax[0].set_title('Nodes colored by potential v')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, ax=ax[0], cax=cax)
    ax[0].set_aspect('equal', 'box')

    #  plt.savefig(file+'.png')

    vmin = np.min(weights)
    vmax = np.max(weights)
    # subset_nodes = range(n_subgraph)
    # subset_nodes = np.loadtxt(data_path + graph_name + '_nodes.txt').astype(int)

    color_map = []
    for node in G:
        if node in subset_nodes:
            color_map.append('red')
        else:
            color_map.append('green')
    cmap = plt.cm.gnuplot
    # ax = plt.subplot()
    nx.draw(G, node_color=color_map, edgelist=edges, edge_color=weights, width=2.0,
            edge_cmap=cmap, vmin=vmin,
            vmax=vmax, cmap=cmap, node_size=30, pos=pos, ax=ax[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    ax[1].set_title('Edges colored by E')
    #  plt.savefig(file+'.png')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, ax=ax[1], cax=cax)
    ax[1].set_aspect('equal', 'box')

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    #                     wspace=2,
    #                     hspace=None)

    plt.show()


class BaseSubgraphIsomorphismSolver(ABC):
    def __init__(self, A, ref_spectrum, problem_params, solver_params,
                 subgraph_size=Optional[None]):
        """
        Base class for spectral graph matching algorithms

        :param A: Adjacency matrix of full graph
        :param ref_spectrum: spectrum of reference graph (i.e., the subgraph)
        :param problem_params: parameters for the problem.
        :param solver_params: parameters for the algorithm and solver.
        :param save_loss_terms: flag for saving the individual loss terms
        """

        self.A = A
        self.D = torch.diag(A.sum(dim=1))
        self.L = lap_from_adj(A)
        self.ref_spectrum = ref_spectrum
        if subgraph_size is not None:
            self.subgraph_size = subgraph_size
        else:
            self.subgraph_size = ref_spectrum.shape[0]
        self.set_problem_params(problem_params)
        self.set_solver_params(solver_params)

    def threshold(self, v_np, threshold_algo='1dkmeans'):
        if threshold_algo == '1dkmeans':
            v_ = v_np - np.min(v_np)
            v_ = v_ / np.max(v_)
            v_clustered, centroids = kmeans1d.cluster(v_, k=2)

        elif threshold_algo == 'smallest':
            # just take the smallest subgraph_size entries in v_np
            subgraph_size = self.ref_spectrum.shape[0]
            v_clustered = np.zeros_like(v_np)
            ind = np.argsort(v_np)
            v_clustered[ind[subgraph_size:]] = 1

        elif threshold_algo == 'spectral':
            E, S = E_from_v(tensor(v_np), self.A)
            affinity_matrix = (self.A + S.detach())
            # affinity_matrix = lap_from_adj(affinity_matrix)
            clustering = \
                SpectralClustering(n_clusters=2, assign_labels='discretize',
                                   random_state=0, affinity='precomputed').fit(
                    affinity_matrix)
            v_clustered = np.ones_like(v_np) - clustering.labels_

        v_clustered = torch.tensor(v_clustered, dtype=torch.float64)
        E_clustered, _ = E_from_v(v_clustered, self.A)
        return v_clustered, E_clustered

    def set_problem_params(self, problem_params):
        for key in problem_params:
            setattr(self, key, problem_params[key])

    def set_solver_params(self, solver_params):
        for key in solver_params:
            setattr(self, key, solver_params[key])

    def set_init(self, v0=None, E0=None):
        n = self.L.shape[0]
        if v0 is None:
            eig_max = torch.max(self.ref_spectrum)
            c = np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
            v0 = (c / np.sqrt(n)) * np.ones(n)
            self.v = torch.tensor(v0, requires_grad=self.train_v, dtype=torch.float64)
        else:
            self.v = v0

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

    def _check_convergence(self, v, a_tol, v_prev=None, r_tol=None):
        return False

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

    @abstractmethod
    def _solve(self,
               maxiter_inner,
               show_iter,
               verbose):
        pass

    def non_smooth_loss_function(self, E):
        l21_loss = l21(E)

        if self.save_individual_losses:
            self.l21_terms.append(l21_loss)

        return self.mu_l21 * l21(E)

    def smooth_loss_function(self, ref_spectrum, L, E, v, save_individual_losses=False):

        spectrum_alignment_term = self.spectrum_alignment_loss(ref_spectrum, L, E, v,
                                                               self.weighted_flag)
        MS_reg_term = self.MSreg(L, E, v)
        trace_reg_term = self.trace_reg(E, self.trace_val)
        graph_split_term = self.graph_split_loss(L, E)

        smooth_loss_term = \
            self.mu_spectral * spectrum_alignment_term \
            + self.mu_MS * MS_reg_term \
            + self.mu_trace * trace_reg_term \
            + self.mu_split * graph_split_term

        if save_individual_losses:
            self.spectrum_alignment_terms.append(
                spectrum_alignment_term.detach().numpy())
            self.MS_reg_terms.append(MS_reg_term.detach().numpy())
            self.trace_reg_terms.append(trace_reg_term.detach().numpy())
            self.graph_split_terms.append(graph_split_term.detach().numpy())

        return smooth_loss_term

    def spectrum_alignment_loss(self, ref_spectrum, L, E, v, weighted_flag=True):
        k = ref_spectrum.shape[0]
        Hamiltonian = L + E + torch.diag(v)
        spectrum = torch.linalg.eigvalsh(Hamiltonian)
        if weighted_flag:
            weights = torch.tensor([1 / w if w > 1e-8 else 1 for w in
                                    ref_spectrum])
        else:
            weights = torch.ones_like(ref_spectrum)
        loss = torch.norm((spectrum[0:k] - ref_spectrum) * weights) ** 2
        return loss

    @staticmethod
    def graph_split_loss(L, E):
        L_edited = L + E
        spectrum = torch.linalg.eigvalsh(L_edited)
        # loss = torch.norm(spectrum[0:2]) ** 2
        loss = spectrum[1]
        # print(loss)
        return loss

    @staticmethod
    def MSreg(L, E, v):
        return v.T @ (L + E) @ v

    @staticmethod
    def trace_reg(E, trace_val=0):
        return (torch.trace(E) - trace_val) ** 2

    def plot_loss(self, plotlosses, loss_val, sleep_time=.00001):
        plotlosses.update({
            'loss': loss_val,
        })
        plotlosses.send()
        sleep(sleep_time)

    def _plot(self, plots):
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
        E = self.E.detach().numpy()
        v = self.v.detach().numpy()

        if plots['full_loss']:
            plt.plot(self.loss_vals, 'b')
            plt.title('full loss')
            plt.xlabel('iter')
            plt.show()

        if plots['E']:
            ax = plt.subplot()
            im = ax.imshow(E - np.diag(np.diag(E)))
            divider = make_axes_locatable(ax)
            ax.set_title('E -diag(E)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['L+E']:
            ax = plt.subplot()
            L_edited = E + self.L.numpy()
            im = ax.imshow(L_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('L+E')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['A edited']:
            ax = plt.subplot()
            A_edited = -set_diag_zero(E + self.L.numpy())
            im = ax.imshow(A_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('A edited')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v']:
            plt.plot(np.sort(v), 'xr')
            plt.title('v')
            plt.show()

        if plots['diag(v)']:
            ax = plt.subplot()
            im = ax.imshow(np.diag(v))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v_otsu']:
            ax = plt.subplot()
            # _, v_clustered = kmeans2(self.v, 2, minit='points')
            v = indicator_from_v_np(v)
            threshold = threshold_otsu(v, nbins=10)
            v_otsu = (v > threshold).astype(float)
            im = ax.imshow(np.diag(v_otsu))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v_otsu)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v_kmeans']:
            ax = plt.subplot()
            # _, v_clustered = kmeans2(self.v, 2, minit='points')
            v = indicator_from_v_np(v)
            v_clustered, centroids = kmeans1d.cluster(v, k=2)
            im = ax.imshow(np.diag(v_clustered))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v_kmeans)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['ref spect vs spect']:
            k = self.ref_spectrum.shape[0]
            plt.plot(self.ref_spectrum.numpy(), 'og')
            plt.plot(self.spectrum[:k], 'xr')
            plt.title('ref spect vs first k eigs of subgraph')
            plt.show()
