import copy

import torch
import numpy as np
from numpy import histogram
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxL21ForSymmCentdMatrixAndInequality, ProxSymmetricCenteredMatrix, ProxId
from tests.optimization.util import double_centering, set_diag_zero
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from skimage.filters.thresholding import threshold_otsu
import networkx as nx
from datetime import datetime
import os
import utils

import kmeans1d


x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
k = 4

clusters, centroids = kmeans1d.cluster(x, k)

print(clusters)  # [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
print(centroids)  # [-50.0, 4.1, 94.0, 200.5]


def block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.1):
    p11 = set_diag_zero(p_parts * torch.ones(n1, n1))
    n=n1+n2

    p22 = set_diag_zero(p_parts * torch.ones(n2, n2))

    p12 = p_off * torch.ones(n1, n2)

    p = torch.zeros([n, n])
    p[0:n1, 0:n1] = p11
    p[0:n1, n1:n] = p12
    p[n1:n, n1:n] = p22
    p[n1:n, 0:n1] = p12.T

    return p


class SubgraphIsomorphismSolver:

    def __init__(self, L, ref_spectrum, params, plots=None):
        self.L = L
        self.ref_spectrum = ref_spectrum
        self.params = params
        self.plots = plots

    def solve(self,full_graph,part_graph,part_nodes):
        L = self.L
        ref_spectrum = self.ref_spectrum
        n = L.shape[0]
        l21_symm_centered_prox = ProxL21ForSymmetricCenteredMatrix(solver="cvx")

        # init
       # v = torch.zeros(n, requires_grad=True, dtype=torch.float64)
       # E = torch.zeros([n, n], dtype=torch.float64)
       # E = double_centering(0.5 * (E + E.T)).requires_grad_()

        v,E=utils.init_groundtruth(full_graph,part_graph,part_nodes,5.0)

        #E = E.requires_grad_()

        #print(type(v))
        maxiter = self.params['maxiter']
        mu_l21 = self.params['mu_l21']
        mu_MS = self.params['mu_MS']
        mu_trace = self.params['mu_trace']
        v_prox = self.params['v_prox']
        E_prox = self.params['E_prox']

        # s = torch.linalg.svdvals(A)
        # lr = 1 / (1.1 * s[0] ** 2)
        lr = self.params['lr']
        lamb = mu_l21 * lr  # This setting is important!
        momentum = self.params['momentum']
        dampening = self.params['dampening']
        pgm = PGM(params=[{'params': v}, {'params': E}],
                  proxs=[v_prox, E_prox],
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


    def plot(self):
        if self.plots['full_loss']:
            plt.loglog(self.loss_vals, 'b')
            plt.title('full loss')
            plt.xlabel('iter')
            plt.show()

        if self.plots['E']:
            ax = plt.subplot()
            im = ax.imshow(self.E)
            divider = make_axes_locatable(ax)
            ax.set_title('E')
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


    def plots_on_graph(self,A,subset_nodes,pos,res_folder,graph_name):
        vmin = np.min(self.v)
        vmax = np.max(self.v)

        G = nx.from_numpy_matrix(A)
        #pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G)
        # plt.rcParams["figure.figsize"] = (20,20)

        # for edge in G.edges():


        for u, w, d in G.edges(data=True):
            d['weight'] = self.E[u, w]

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

       # cmap = plt.cm.gnuplot


       # plt.figure()

        #ax = plt.subplot()
        #nx.draw_networkx_nodes(G, node_color=self.v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
        #        node_size=30,
        #        pos=pos, ax = ax)

        #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        #sm._A = []
        #ax.set_title('Nodes colored by potential v')
        #plt.colorbar(sm)
       # plt.savefig(res_folder + '/images/' + graph_name + '_v.png', dpi=100)
       # plt.show()

        #vmin = np.min(weights)
        #vmax = np.max(weights)

       # subset_nodes=range(n1)
       # subset_nodes = np.loadtxt(data_path + graph_name + '_nodes.txt').astype(int)

        color_map = []
        for node in G:
            if node in subset_nodes:
                color_map.append('blue')
            else:
                color_map.append('green')
        cmap = plt.cm.rainbow
        map2=plt.cm.gnuplot


        plt.figure(figsize=(10, 10))#, dpi=1000)
        ax = plt.subplot()
        nx.draw_networkx_nodes(G, node_color=self.v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
                               node_size=30,
                               pos=pos, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []

        plt.colorbar(sm)
        vmin = np.min(weights)
        vmax = np.max(weights)

        nx.draw_networkx_edges(G, edgelist=edges, edge_color=weights, width=2.0,
                edge_cmap=map2, vmin=vmin,vmax=vmax, cmap=map2, node_size=30, pos=pos)

        sm = plt.cm.ScalarMappable(cmap=map2, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax.set_title('Edges colored by E')
        plt.colorbar(sm,orientation="horizontal")

        plt.savefig(res_folder + '/images/' + graph_name + '_E.pdf')#, dpi=100)
        plt.show()

def run_on_graphset():

        source_folder = '../data/benchmark/random_graphs/part_5_rest_15/'

        #print(source_folder)

        graph_dir =  os.listdir(source_folder)

        now = datetime.now()
        res_folder = '../results/' + now.strftime("%d-%m-%Y_%H-%M-%S")

        os.mkdir(res_folder)
        os.mkdir(res_folder + '/images')

        #sh.copy(config_file, res_folder + '/' + config_file)

        list_full = []
        list_part = []
        list_nodes = []

        for file in graph_dir:
            #print(file)
            if 'full' in file:
                list_full.append(file)
            if 'part' in file:
                list_part.append(file)
            if 'nodes' in file:
                list_nodes.append(file)
        cg = 1
        #print(list_nodes)
        for graph in list_nodes:
            print(cg)
            cg = cg + 1
            cur_graph = graph[0:len(graph) - 10]
            nodes_part = source_folder + '/' + cur_graph + '_nodes.txt'
            part_graph = source_folder + '/' + cur_graph + '_part.txt'

            full_graphs = [g for g in list_full if cur_graph in g]

            for i in range(len(full_graphs)):
                graph_name = full_graphs[i][0:len(full_graphs[i]) - 9]
                full_graph = source_folder + '/' + full_graphs[i]
                disc_graph = source_folder + '/' + graph_name[0:-4] + '0000_full.txt'

               # print(full_graph)
                print(graph_name)

                if (os.stat(part_graph).st_size != 0):
                    #try:
                        run_opt_on_graph(source_folder, disc_graph, graph_name, full_graph, part_graph, nodes_part,res_folder)
                    #except:
                        print('optimization failed')
            # [v_with gt,obj_w_gt,v_in_the_beginning, v_in_the_end, obj values,eigenvalues]=run_optimization(full_graph, part_graph,part_nodes, [big_list_of_params])

def run_opt_on_graph(source_folder, disc_graph, graph_name, full_graph, part_graph, nodes_part,res_folder):

        data_path = source_folder


        #print(data_path + graph_name + '_nc' + str(n_con).zfill(4) + '.txt')
        A = torch.from_numpy(edgelist_to_adjmatrix(full_graph))
        # A = torch.tril(torch.bernoulli(p))

        #A = (A + A.T)'
        print(A)
        D = torch.diag(A.sum(dim=1))
        L = D - A
        subset_nodes = np.loadtxt(nodes_part).astype(int)
        A_d = A.detach().numpy().astype(int)

       # print(data_path + graph_name + '_nc0000.txt')
        A_det = (edgelist_to_adjmatrix(disc_graph))
        G_det = nx.from_numpy_matrix(A_det)
        pos = nx.spring_layout(G_det)

        G = nx.from_numpy_matrix(A_d)

        color_map = []
        for node in G:
            if node in subset_nodes:
                color_map.append('blue')
            else:
                color_map.append('green')
        cmap = plt.cm.gnuplot
        nx.draw(G, node_color=color_map, node_size=30, pos=pos)

        #  plt.savefig(file+'.png')
        plt.show()

        plt.imshow(A)
        plt.title('A')
        plt.show()

        # A_sub = A[0:n1, 0:n1]
        A_sub_np=edgelist_to_adjmatrix(part_graph)
        A_sub = torch.from_numpy(A_sub_np)
        D_sub = torch.diag(A_sub.sum(dim=1))
        L_sub = D_sub - A_sub

        G_part=nx.from_numpy_matrix(A_sub_np)
        ref_spectrum = torch.linalg.eigvalsh(L_sub)
        params = {'maxiter': 100,
                  'mu_l21': 1,
                  'mu_MS': 8,
                  'mu_trace': 0.0,
                  'lr': 0.01,
                  'momentum': 0.1,
                  'dampening': 0.0,
                  'v_prox': ProxId(),
                  # 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx")
                  'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L)
                  }
        plots = {
            'full_loss': True,
            'E': True,
            'v': True,
            'diag(v)': True,
            'v_otsu': False,
            'v_kmeans': True,
            'A edited': False,
            'L+E': False,
            'ref spect vs spect': True}
        subgraph_isomorphism_solver = \
            SubgraphIsomorphismSolver(L, ref_spectrum, params, plots)
        v, E = subgraph_isomorphism_solver.solve(G,G_part,subset_nodes)

        # subgraph_isomorphism_solver.plot()
        subgraph_isomorphism_solver.plots_on_graph(A.detach().numpy().astype(int), subset_nodes, pos,res_folder,graph_name)


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


def run_on_sbm(n_nodes, p_parts,p_off):
    #make graph

        #run opt


    n1 = 5
    n2 = 15
    n = n1 + n2
    n_comps=len(n_nodes)
   # p = block_stochastic_graph(n1, n2)
    p_mat = np.ones((n_comps, n_comps)) * p_off
    np.fill_diagonal(p_mat, p_parts)
    subset_nodes=range(n_nodes[0])
    probs = p_mat.tolist()
    G = nx.stochastic_block_model(n_nodes, probs)
    G_disc= copy.deepcopy(G)

    for edge in G_disc.edges():
        u = edge[0]
        v = edge[1]
        if u in subset_nodes and v not in subset_nodes:
            G_disc.remove_edge(*edge)


    pos = nx.spring_layout(G_disc)
    A_nnp=nx.adjacency_matrix(G).todense().astype(float)
    A = torch.from_numpy(A_nnp)
    #A = torch.tril(torch.bernoulli(p))
    print(A)
    #A = (A + A.T)
    D = torch.diag(A.sum(dim=1))
    L = D - A

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n_nodes[0], 0:n_nodes[0]]
    G_part=nx.from_numpy_matrix(A[0:n_nodes[0], 0:n_nodes[0]].detach().numpy())
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)
    params = {'maxiter': 200,
             'mu_l21': 1,
             'mu_MS': 8,
             'mu_trace': 0.0,
             'lr': 0.002,
             'momentum': 0.1,
             'dampening': 0,
             'v_prox': ProxId(),
             # 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx")
             'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L)
             }
    plots = {
       'full_loss': True,
       'E': True,
       'v': True,
       'diag(v)': True,
       'v_otsu': False,
       'v_kmeans': True,
       'A edited': False,
       'L+E': False,
       'ref spect vs spect': True}
    subgraph_isomorphism_solver = \
       SubgraphIsomorphismSolver(L, ref_spectrum, params, plots)

    v, E = subgraph_isomorphism_solver.solve(G,G_part,subset_nodes)
    #subgraph_isomorphism_solver.run_on_graphset()
    #subgraph_isomorphism_solver.plot()
    subgraph_isomorphism_solver.plots_on_graph(A.detach().numpy().astype(int),subset_nodes,pos,'../results/test/','b')


if __name__ == '__main__':

    #run_on_graphset()
    run_on_sbm([5,15],0.7,0.2)

    source_folder= '../data/11-01-2022_13-53-35'
    graph_name='er_10_944_er_10_964'
    disc_graph=source_folder+'/er_10_944_er_10_964_nc0000_full.txt'

    full_graph=source_folder+'/'+graph_name+'_nc0009_full.txt'
    part_graph=source_folder+'/'+graph_name+'_part.txt'
    nodes_part=source_folder+'/'+graph_name+'_nodes.txt'
    res_folder='../results/test'
   # run_opt_on_graph(source_folder, disc_graph, graph_name, full_graph, part_graph, nodes_part, res_folder)
    #torch.manual_seed(12)

    #data_path='../data/11-01-2022_13-53-08/'
    #graph_name='ba_24_3_nw_23_2_290'

    #n_con=2

    #n1 = 5
    #n2 = 15
    #n = n1 + n2

    #p = block_stochastic_graph(n1, n2)
    #print(data_path+graph_name+'_nc'+str(n_con).zfill(4)+'.txt')
    #A = torch.from_numpy(edgelist_to_adjmatrix(data_path+graph_name+'_nc'+str(n_con).zfill(4)+'_full.txt'))
    #A = torch.tril(torch.bernoulli(p))

    #A = (A + A.T)
    #D = torch.diag(A.sum(dim=1))
    #L = D - A
    #subset_nodes = np.loadtxt(data_path+graph_name+'_nodes.txt').astype(int)
    #A_d=A.detach().numpy().astype(int)

    #print(data_path + graph_name + '_nc0000.txt')
    #A_det = (edgelist_to_adjmatrix(data_path + graph_name + '_nc0000_full.txt'))
    #G_det = nx.from_numpy_matrix(A_det)
    #pos = nx.spring_layout(G_det)

    #G = nx.from_numpy_matrix(A_d)

    #color_map = []
    #for node in G:
    #    if node in subset_nodes:
    #        color_map.append('blue')
    #    else:
    #        color_map.append('green')
    #cmap = plt.cm.gnuplot
    #nx.draw(G, node_color=color_map, node_size=30,pos=pos)

    #  plt.savefig(file+'.png')
    #plt.show()


    #plt.imshow(A)
    #plt.title('A')
    #plt.show()

    #A_sub = A[0:n1, 0:n1]
    #A_sub=torch.from_numpy(edgelist_to_adjmatrix(data_path+graph_name+'_part.txt'))
    #D_sub = torch.diag(A_sub.sum(dim=1))
    #L_sub = D_sub - A_sub
    #ref_spectrum = torch.linalg.eigvalsh(L_sub)
    #params = {'maxiter': 200,
    #          'mu_l21': 1,
    #          'mu_MS': 8,
    #          'mu_trace': 0.0,
    #          'lr': 0.002,
    #          'momentum': 0.1,
    #          'dampening': 0,
    #          'v_prox': ProxId(),
    #          # 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx")
    #          'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L)
    #          }
    #plots = {
    #    'full_loss': True,
    #    'E': True,
    #    'v': True,
    #    'diag(v)': True,
    #    'v_otsu': False,
    #    'v_kmeans': True,
    #    'A edited': False,
    #    'L+E': False,
    #    'ref spect vs spect': True}
    #subgraph_isomorphism_solver = \
    #    SubgraphIsomorphismSolver(L, ref_spectrum, params, plots)
    #v, E = subgraph_isomorphism_solver.solve()
    #subgraph_isomorphism_solver.run_on_graphset()
    #subgraph_isomorphism_solver.plot()
    #subgraph_isomorphism_solver.plots_on_graph(A.detach().numpy().astype(int),subset_nodes,pos)
