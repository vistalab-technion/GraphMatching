import torch
import numpy as np
from numpy import histogram
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxL21ForSymmCentdMatrixAndInequality, ProxSymmetricCenteredMatrix, ProxId, \
    ProxNonNeg
from tests.optimization.util import double_centering, set_diag_zero
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from skimage.filters.thresholding import threshold_otsu
import networkx as nx
import kmeans1d
import optuna
import copy
import os
import datetime


def block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.1):
    p11 = set_diag_zero(p_parts * torch.ones(n1, n1))

    p22 = set_diag_zero(p_parts * torch.ones(n2, n2))

    p12 = p_off * torch.ones(n1, n2)

    n=n1+n2

    p = torch.zeros([n, n])
    p[0:n1, 0:n1] = p11
    p[0:n1, n1:n] = p12
    p[n1:n, n1:n] = p22
    p[n1:n, 0:n1] = p12.T

    return p


class SubgraphIsomorphismSolver:

    def __init__(self, L, ref_spectrum, params, plots):
        self.L = L
        self.ref_spectrum = ref_spectrum
        self.params = params
        self.plots = plots
        self.spectrum_alignment_terms = []
        self.MS_reg_terms = []
        self.trace_reg_terms = []
        self.graph_split_terms = []
        self.l21_terms = []
        self.save_individual_losses = plots['individual loss terms']
        self.mu_spectral = self.params['mu_spectral']
        self.mu_l21 = self.params['mu_l21']
        self.mu_MS = self.params['mu_MS']
        self.mu_trace = self.params['mu_trace']
        self.mu_split = self.params['mu_split']
        self.trace_val = self.params['trace_val']

    def solve(self):
        L = self.L
        ref_spectrum = self.ref_spectrum
        n = L.shape[0]

        # init
        v = torch.ones(n, requires_grad=True, dtype=torch.float64)
        E = torch.zeros([n, n], dtype=torch.float64)
        E = double_centering(0.5 * (E + E.T)).requires_grad_()
        v_prox = self.params['v_prox']
        E_prox = self.params['E_prox']

        # s = torch.linalg.svdvals(A)
        # lr = 1 / (1.1 * s[0] ** 2)
        maxiter = self.params['maxiter']
        lr = self.params['lr']
        lamb = self.mu_l21 * lr  # This setting is important!
        pgm = PGM(params=[{'params': v}, {'params': E}],
                  proxs=[v_prox, E_prox],
                  lr=lr)

        full_loss_function = lambda ref, L, E, v: \
            self.smooth_loss_function(ref, L, E, v) \
            + self.non_smooth_loss_function(E)

        loss_vals = []
        for i in tqdm(range(maxiter)):
            pgm.zero_grad()
            loss = self.smooth_loss_function(ref_spectrum, L, E, v)
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
        self.loss=loss
        # self.plots()
        #print(loss)
        print(f"v= {v}")
        print(f"E= {E}")
        print(f"lambda= {spectrum[0:k]}")
        print(f"lambda*= {ref_spectrum}")
        print(f"||lambda-lambda*|| = {torch.norm(spectrum[0:k] - ref_spectrum)}")
        return v, E

    def non_smooth_loss_function(self, E):
        l21_loss = l21(E)

        if self.save_individual_losses:
            self.l21_terms.append(l21_loss)

        return self.mu_l21 * l21(E)

    def smooth_loss_function(self, ref_spectrum, L, E, v):

        spectrum_alignment_term = self.spectrum_alignment_loss(ref_spectrum, L, E, v)
        MS_reg_term = self.MSreg(L, E, v)
        trace_reg_term = self.trace_reg(E, self.trace_val)
        graph_split_term = self.graph_split_loss(L, E)

        smooth_loss_term = \
            self.mu_spectral * spectrum_alignment_term \
            + self.mu_MS * MS_reg_term \
            + self.mu_trace * trace_reg_term \
            + self.mu_split * graph_split_term

        if self.save_individual_losses:
            self.spectrum_alignment_terms.append(
                spectrum_alignment_term.detach().numpy())
            self.MS_reg_terms.append(MS_reg_term.detach().numpy())
            self.trace_reg_terms.append(trace_reg_term.detach().numpy())
            self.graph_split_terms.append(graph_split_term.detach().numpy())

        return smooth_loss_term

    @staticmethod
    def spectrum_alignment_loss(ref_spectrum, L, E, v):
        k = ref_spectrum.shape[0]
        Hamiltonian = L + E + torch.diag(v)
        spectrum = torch.linalg.eigvalsh(Hamiltonian)
        loss = torch.norm(spectrum[0:k] - ref_spectrum) ** 2
        return loss

    @staticmethod
    def graph_split_loss(L, E):
        L_edited = L + E
        spectrum = torch.linalg.eigvalsh(L_edited)
        #loss = torch.norm(spectrum[0:2]) ** 2
        loss = spectrum[1]
        # print(loss)
        return loss

    @staticmethod
    def MSreg(L, E, v):
        return v.T @ (L + E) @ v

    @staticmethod
    def trace_reg(E, trace_val=0):
        return (torch.trace(E) - trace_val) ** 2

    def plot(self):
        if self.plots['full_loss']:
            plt.loglog(self.loss_vals, 'b')
            plt.title('full loss')
            plt.xlabel('iter')
            plt.show()

        if self.plots['E']:
            ax = plt.subplot()
            im = ax.imshow(self.E - np.diag(np.diag(self.E)))
            divider = make_axes_locatable(ax)
            ax.set_title('E -diag(E)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if self.plots['L+E']:
            ax = plt.subplot()
            L_edited = self.E + self.L.numpy()
            im = ax.imshow(L_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('L+E')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if self.plots['A edited']:
            ax = plt.subplot()
            A_edited = -set_diag_zero(self.E + self.L.numpy())
            im = ax.imshow(A_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('A edited')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if self.plots['v']:
            plt.plot(np.sort(self.v), 'xr')
            plt.title('v')
            plt.show()

        if self.plots['diag(v)']:
            ax = plt.subplot()
            im = ax.imshow(np.diag(self.v))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if self.plots['v_otsu']:
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

        if self.plots['v_kmeans']:
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

        if self.plots['ref spect vs spect']:
            plt.plot(self.ref_spectrum.numpy(), 'og')
            plt.plot(self.spectrum, 'xr')
            plt.title('ref spect vs spect')
            plt.show()

        if self.plots['individual loss terms']:
            font_color = "r"
            # Create two subplots and unpack the output array immediately
            fig, axes = plt.subplots(nrows=3, ncols=2)
            ax = axes.flat
            ax[0].loglog(self.spectrum_alignment_terms)
            ax[0].set_ylabel('loss', c=font_color)
            ax[0].set_xlabel('iteration', c=font_color)
            ax[0].set_title(f'spectral alignment, mu = {self.mu_spectral}', c=font_color)

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
                                hspace=1)
            plt.show()

    def plots_on_graph(self,A,subset_nodes,pos,res_folder,graph_name):
        vmin = np.min(self.v)
        vmax = np.max(self.v)

        G = nx.from_numpy_matrix(A)

        for u, w, d in G.edges(data=True):
            d['weight'] = self.E[u, w]

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        color_map = []
        for node in G:
            if node in subset_nodes:
                color_map.append('blue')
            else:
                color_map.append('green')
        cmap = plt.cm.rainbow
        map2 = plt.cm.gnuplot

        plt.figure(figsize=(10, 10))  # , dpi=1000)
        ax = plt.subplot()
        nx.draw_networkx_nodes(G, node_color=self.v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
                               node_size=40,
                               pos=pos, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []

        plt.colorbar(sm)
        vmin = np.min(weights)
        vmax = np.max(weights)

        nx.draw_networkx_edges(G, edgelist=edges, edge_color=weights, width=2.0,
                               edge_cmap=map2, vmin=vmin, vmax=vmax, cmap=map2, node_size=40, pos=pos)

        sm = plt.cm.ScalarMappable(cmap=map2, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax.set_title('Edges colored by E')
        plt.colorbar(sm, orientation="horizontal")

        #plt.savefig(res_folder + '/images/' + graph_name + '_E.pdf')  # , dpi=100)
        plt.show()

    def plot_on_graph(self, A):
        vmin = np.min(self.v)
        vmax = np.max(self.v)

        G = nx.from_numpy_matrix(A)
        pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G)
        # plt.rcParams["figure.figsize"] = (20,20)

        # for edge in G.edges():

        for u, w, d in G.edges(data=True):
            d['weight'] = self.E[u, w]

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        cmap = plt.cm.gnuplot
        ax = plt.subplot()
        nx.draw(G, node_color=self.v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
                node_size=30,
                pos=pos, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax.set_title('Nodes colored by potential v')
        plt.colorbar(sm)
        #  plt.savefig(file+'.png')
        plt.show()

        vmin = np.min(weights)
        vmax = np.max(weights)
        subset_nodes = range(n1)
        # subset_nodes = np.loadtxt(data_path + graph_name + '_nodes.txt').astype(int)

        color_map = []
        for node in G:
            if node in subset_nodes:
                color_map.append('red')
            else:
                color_map.append('green')
        cmap = plt.cm.gnuplot
        ax = plt.subplot()
        nx.draw(G, node_color=color_map, edgelist=edges, edge_color=weights, width=2.0,
                edge_cmap=cmap, vmin=vmin,
                vmax=vmax, cmap=cmap, node_size=30, pos=pos)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax.set_title('Edges colored by E')
        plt.colorbar(sm)
        #  plt.savefig(file+'.png')
        plt.show()


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


def opt_hyps(trial,full_graph,part_graph,nodes_part):




    A = torch.from_numpy(edgelist_to_adjmatrix(full_graph))
    D = torch.diag(A.sum(dim=1))
    L = D - A

    subset_nodes = np.loadtxt(nodes_part).astype(int)

    #plt.imshow(A)
    #plt.title('A')
    #plt.show()


    A_sub_np=edgelist_to_adjmatrix(part_graph)
    A_sub = torch.from_numpy(A_sub_np)
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)


    params = {'maxiter':100,
              'mu_spectral': trial.suggest_float('mu_spectral', 0.0, 15.0),
              'mu_l21': trial.suggest_float('mu_l21', 0.0, 15.0),
              'mu_MS': trial.suggest_float('mu_MS', 0.0, 15.0),
              'mu_split': trial.suggest_float('mu_split', 0.0, 15.0),
              'mu_trace': trial.suggest_float('mu_trace', 0.0, 15.0),
              'lr': trial.suggest_float('lr', 0.001, 0.3),
              'v_prox': ProxNonNeg(),
              # 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx"),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                               trace_upper_bound=
                                                               1.1 * torch.trace(L)),
              'trace_val': 0
              }
    plots = {
        'full_loss': True,
        'E': True,
        'v': True,
        'diag(v)': True,
        'v_otsu': False,
        'v_kmeans': True,
        'A edited': True,
        'L+E': False,
        'ref spect vs spect': True,
        'individual loss terms': True}

    loss_opt=0.0
    subgraph_isomorphism_solver = \
        SubgraphIsomorphismSolver(L, ref_spectrum, params, plots)

    try:
        v, E = subgraph_isomorphism_solver.solve()

        v_new=v.detach().numpy()
        _, v_clustered = kmeans2(v_new, 2, minit='points')
        v_gt=np.zeros(A.size(dim=1))
        for i in subset_nodes:
            v_gt[i]=1
        v_gt_inv=np.abs(v_gt-1)
        loss_opt=min(np.linalg.norm(v_clustered-v_gt),np.linalg.norm(v_clustered-v_gt_inv))
        print(loss_opt)
    except:
        print("An exception occurred")
        loss_opt=np.sqrt(A.size(dim=1))
        print(loss_opt)


    return loss_opt
    #subgraph_isomorphism_solver.plot()
    #subgraph_isomorphism_solver.plot_on_graph(A.numpy().astype(int))
    # subgraph_isomorphism_solver.plots_on_graph(A.numpy().astype(int),subset_nodes, pos, '../results/test/','b')


if __name__ == '__main__':

    torch.manual_seed(12)

    n1 = 50
    n2 = 60
    n = n1 + n2
    p = block_stochastic_graph(n1, n2, p_parts=0.4, p_off=0.3)

    data = '../data/benchmark/random_graphs/part_5_rest_15/'

    full_graph = data + 'ba_5_2_er_15_323_nc0006_full.txt'
    nodes_part = data + 'ba_5_2_er_15_323_nodes.txt'
    part_graph = data + 'ba_5_2_er_15_323_part.txt'

    A = torch.tril(torch.bernoulli(p)).double()
    A = (A + A.T)
    A = torch.from_numpy(edgelist_to_adjmatrix(full_graph))
    D = torch.diag(A.sum(dim=1))
    L = D - A

    A_np = A.detach().numpy()
    G = nx.from_numpy_matrix(A_np)
    G_disc = copy.deepcopy(G)
    subset_nodes = range(n1)
    subset_nodes = np.loadtxt(nodes_part).astype(int)
    for edge in G_disc.edges():
        u = edge[0]
        v = edge[1]
        if u in subset_nodes and v not in subset_nodes:
            G_disc.remove_edge(*edge)

    pos = nx.spring_layout(G_disc)

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n1, 0:n1]
    A_np_sub = A_sub.detach().numpy()
    A_sub_np=edgelist_to_adjmatrix(part_graph)
    A_sub = torch.from_numpy(A_sub_np)
    G_part = nx.from_numpy_matrix(A_np_sub)

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n1, 0:n1]
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub

    ref_spectrum = torch.linalg.eigvalsh(L_sub)

    params = {'maxiter': 100,
              'mu_spectral':0.2731064475513856,
              'mu_l21': 4.564506354623763,
              'mu_MS': 2.2417505930940984,
              'mu_split': 6.876046123288896,
              'mu_trace': 7.427203409520426,
              'lr': 0.011845777350756984,
              'v_prox': ProxNonNeg(),
              #'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx"),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                                trace_upper_bound=
                                                                1.1*torch.trace(L)),
              'trace_val': 0
              }
    plots = {
        'full_loss': True,
        'E': True,
        'v': True,
        'diag(v)': True,
        'v_otsu': False,
        'v_kmeans': True,
        'A edited': True,
        'L+E': False,
        'ref spect vs spect': True,
        'individual loss terms': True}
    subgraph_isomorphism_solver = \
        SubgraphIsomorphismSolver(L, ref_spectrum, params, plots)
    v, E = subgraph_isomorphism_solver.solve()
    _, v_clustered = kmeans2(v.detach().numpy(), 2, minit='points')
    print(v_clustered)
    _, v_clustered = kmeans2(v.detach().numpy(), 2, minit='points')

    subgraph_isomorphism_solver.plot()
    subgraph_isomorphism_solver.plot_on_graph(A.numpy().astype(int))
    subgraph_isomorphism_solver.plots_on_graph(A.numpy().astype(int),subset_nodes, pos, '../results/test/','b')