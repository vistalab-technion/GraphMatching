import networkx as nx

import numpy as np
import random
import scipy as sci


def init_orig_subgraph(subset_nodes,full_size,noise,rest):
    x0 = np.zeros((full_size, full_size))

    r = np.random.rand(full_size, 1) * noise - noise/2.0
    # print(r)
    np.fill_diagonal(x0, rest)
    for i in subset_nodes:
        x0[i, i] = 0

    for i in range(0,full_size):
       x0[i,i]=x0[i,i]+r[i]

    #print(np.diag(x0))
    return x0


def make_degree_matrix(A):
    vvector = np.sum(A, axis=1).flatten()

    n = np.shape(vvector)[1]


    Deg = np.zeros((n, n))

    for i in range(0, n):

        Deg[i, i] = vvector[0,i]
    return Deg


def make_delimited_adj_matrix(A_part, D_full, D_part, part_size):
    D_part_delimited = D_full[0:part_size, 0:part_size]
    D_diff = D_part_delimited - D_part

    return A_part + D_diff


def calc_Laplacian(A, Deg):
    Deg = sci.linalg.fractional_matrix_power(Deg, -0.5)

    L = np.identity(n) - Deg @ A @ Deg

    return L


def make_potential(nodes, n, tau):
    nodes = nodes.astype(int)
    pot = np.identity(n)

    part_n = np.shape(nodes)[0]

    for i in range(0, part_n):
        pot[nodes[i], nodes[i]] = 0

    for i in range(n):
        if pot[i, i] != 0:
            pot[i, i] = tau

    return pot


def decompose_laplacian(A, Deg, n):
    # print(np.shape(Deg))

    Deg = sci.linalg.fractional_matrix_power(Deg, -0.5)

    L = np.identity(n) - Deg @ A @ Deg
    # plt.imshow(L)
    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eigh(L)

    return L, D, V


def decompose_laplacian_unnorm(A, Deg, n):
    # print(np.shape(Deg))

    L = Deg - A
    # plt.imshow(L)
    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eigh(L)

    return L, D, V


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


def generate_sbm(size_part, size_rest, connecting_edges, density):
    full_size = size_part + size_rest
    p = [[density[0], 0.0], [0.0, density[1]]]
    random.seed(10)
    edges = [(random.randint(0, size_part - 1), random.randint(size_part, full_size - 1)) for i in
             range(connecting_edges)]
    sizes = [size_part, size_rest]
    # print(edges)
    G = nx.stochastic_block_model(sizes, p, seed=0)

    # color_map = []

    # for node in G:
    #    if node < size_part:
    #        color_map.append('blue')
    #    else:
    #        color_map.append('purple')

    for edge in edges:
        G.add_edge(edge[0], edge[1])
        G.add_edge(edge[1], edge[0])

    # nx.draw(G,node_color=color_map)

    # plt.show()

    return G