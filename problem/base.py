import abc
from abc import ABC
from abc import abstractmethod
from typing import Optional

import kmeans1d
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from torch import tensor

from tests.optimization.util import double_centering, set_diag_zero


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

    @abstractmethod
    def solve(self,
              max_outer_iters=1,
              max_inner_iters=10,
              show_iter=10,
              verbose=False):
        """"
        solver for spectral subgraph matching problem
        :param max_outer_iters: number of outer iterations
        :param max_inner_iters: number of inner iterations
        :param show_iters: number of iterations between plots
        :param verbose: flag for printing things throuhgout the iterations
        """
        pass
