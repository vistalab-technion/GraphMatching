#import jax.numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
import numpy as np
#plt.rcParams["figure.figsize"] = (25,25)
from jax.config import config
config.update("jax_enable_x64", True)
import networkx as nx
from scipy.optimize import minimize
import jax.numpy as nnp
from jax import grad
import random

import laplacians as lp
import opt_helpers as oh


def main():
    # numbers of edges connecting part and graph
    n_con = 7

    # full graph including the part
    full_graph = 'data/can_processed_' + str(n_con) + '.txt'
    # part of the graph
    part_graph = 'data/can_sub_' + str(n_con) + '.txt'
    # nodes of the part of the graph
    nodes_part = 'data/can_subnodes_0.txt'

    diag_val = 0.0
    noise = 5.0
    rest = 1.0

    # damping factor
    mu = 0.0

    A = lp.edgelist_to_adjmatrix(full_graph)
    full_size = np.shape(A)[0]

    subset_nodes = np.loadtxt(nodes_part).astype(int) - 1
    x0 = lp.init_orig_subgraph(subset_nodes, full_size, noise, rest)

    diag_init = np.diag(x0)
    x0 = np.ndarray.flatten(x0)

    A = lp.edgelist_to_adjmatrix(full_graph)
    A_part = lp.edgelist_to_adjmatrix(part_graph)

    G = nx.from_numpy_matrix(A)

    G_part = nx.from_numpy_matrix(A_part)

    well = nx.to_numpy_matrix(G)

    # colormap

    subset_nodes = np.loadtxt(nodes_part).astype(int) - 1
    G_part = nx.from_numpy_matrix(A_part)
    color_map = []
    for node in G:
        if node in subset_nodes:
            color_map.append('blue')
        else:
            color_map.append('green')

    size_part = G_part.number_of_nodes()
    size_rest = G.number_of_nodes() - size_part

    full_size = size_part + size_rest

    A_full = nx.adjacency_matrix(G)
    A_part = nx.adjacency_matrix(G_part)

    Deg_part = lp.make_degree_matrix(A_part)
    Deg_full = lp.make_degree_matrix(A_full)

    # calculate Laplacians
    L_full, D_full, V_full = lp.decompose_laplacian_unnorm(A_full, Deg_full, size_part + size_rest)
    L_part, D_part, V_part = lp.decompose_laplacian_unnorm(A_part, Deg_part, size_part)


    evnr = 5
    v = np.zeros(379)

    for i in range(379):
        if i not in subset_nodes:
            v[i]=10;
    E = np.zeros((379, 379))

    E[169, 248] = 1.0
    E[169, 122] = 1.0
    E[169, 67] = 1.0
    E[169, 49] = 1.0
    E[100, 67] = 1.0
    E[209, 185] = 1.0
    E[65, 214] = 1.0

    E[248, 169] = 1.0
    E[122, 169] = 1.0
    E[67, 169] = 1.0
    E[49, 169] = 1.0
    E[67, 100] = 1.0
    E[185, 209] = 1.0
    E[214, 65] = 1.0

    E[169, 169] = -4.0
    E[100, 100] = -1.0
    E[248, 248] = -1.0
    E[122, 122] = -1.0
    E[67, 67] = -2.0
    E[49, 49] = -1.0
    E[209, 209] = -1.0
    E[185, 185] = -1.0
    E[214, 214] = -1.0
    E[65, 65] = -1.0

    objective_v(v, E, D_part, L_full, evnr, mu)
    print(objective_E(E, v, D_part, L_full, evnr, mu))

    u = grad_e(E, v, D_part, L_full, evnr, mu)
    a = grad_v(v, E, D_part, L_full, evnr, mu)

    #print(np.shape(u))
    #print(np.shape(a))

    rho=0.1
    lamb=0.1
    nu=np.random.rand(379, 379)
    E,v=solve_E_v(rho,v,E,D_part, L_full, evnr, mu,lamb,nu)
    plt.imshow(E, cmap='hot')
    plt.show()

    print(objective_E(E, v, D_part, L_full, evnr, mu))
def objective_v(v, E, D_part, L_full, evnr, mu):
    # gr_size=int(np.sqrt(np.shape(v)))
    # x=nnp.reshape(v,(gr_size,gr_size))

    gr_part=np.shape(D_part)[0]
    x = nnp.diag(v)

    ham = L_full + x + E
    D_ham, U_ham = nnp.linalg.eigh(ham)

    D_hamm = (D_ham[0:gr_part])
    diag_x = nnp.diag(x)

    loss = nnp.power((D_hamm[0:evnr] - D_part[0:evnr]), 2).sum() + mu * (diag_x.transpose() @ L_full @ diag_x)

    return loss


def objective_E(E, v, D_part, L_full, evnr, mu):
    gr_size=np.shape(v)[0]
    # x=nnp.reshape(v,(gr_size,gr_size))

    gr_part=np.shape(D_part)[0]
    x = nnp.diag(v)
    ham = L_full + x + E
    D_ham, U_ham = nnp.linalg.eigh(ham)

    D_hamm = (D_ham[0:gr_part])
    diag_x = nnp.diag(x)



    cur_l2=nnp.sum(nnp.power(E, 2),0)
    l2=nnp.sqrt(cur_l2)
    l21=nnp.sum(l2)


    print(l21)
    print("l21: %f" %(l21))

    loss = nnp.power((D_hamm[0:evnr] - D_part[0:evnr]), 2).sum() + mu * (diag_x.transpose() @ L_full @ diag_x)+l21

    return loss


def grad_e(E, v, D_part, L_full, evnr, mu):
    grad_obj = grad(objective_E)
    a = grad_obj(E, v, D_part, L_full, evnr, mu)
    return np.array(a)


def grad_v(v, E, D_part, L_full, evnr, mu):
    grad_obj = grad(objective_v)
    a = grad_obj(v,E, D_part, L_full, evnr, mu)
    return np.array(a)


def prox_l21(X, lamb):

    res = prox_l2(X[0, :], lamb)
    for i in range(1, np.shape(X)[0]):
        res = np.c_[(res, prox_l2(X[i, :], lamb))]
    return res


def prox_l2(x, lamb):
    return np.maximum(1.0 - (lamb / np.linalg.norm(x, ord=2.0)), 0.0) * x


def solve_X(Y, Z, nu, lamb, rho):
    W = (1.0 / 2.0 * lamb) * Z + (rho / 2) * (Y + nu)
    mu = 1.0 / ((1.0 / lamb) + rho)

    return prox_l21(W, mu)


def solve_Y(X, nu):
    n = nnp.shape(X)[0]
    ones=nnp.ones((n,1)).astype(float)
    U = X + nu
    W = (1.0 / 2.0) * (U + U.transpose())

    t1=ones@ones.transpose()@W
    t2=W@ones@ones.transpose()
    t3=t1+t2
    t4=(ones@ones.transpose()@W@ones@ones.transpose())
    Y=W-1.0/n*t3+(1.0/(n**2.0))*t4

    return Y


def solve_nu(nu, X, Y):
    return nu + (X - Y)


def ADMM(X, Y, nu, lamb, rho,Z):
    for i in range(10):
        X = solve_X(Y, Z, nu, lamb, rho)
        Y = solve_Y(X, nu)
        nu = solve_nu(nu, X, Y)

    return Y,nu

def prox_grad(rho,v,E,D_part, L_full, evnr, mu,lamb,nu):
    Z_cur=E-rho*grad_e(E, v, D_part, L_full, evnr, mu)
    E,nu=ADMM(E,E,nu,lamb,rho,Z_cur)
    return E,nu;

def grad_descent(rho,v,E,D_part, L_full, evnr, mu):
    v=v-rho*grad_v(v, E, D_part, L_full, evnr, mu)
    return v;


def solve_E_v(rho,v,E,D_part, L_full, evnr, mu,lamb,nu):

    for i in range(0,10):
        E_cur=E
        print(i)
        E_cur,nu=prox_grad(rho,v,E_cur,D_part, L_full, evnr, mu,lamb,nu)

       # v=grad_descent(rho,v,E,D_part, L_full, evnr, mu,lamb,nu)

        print(objective_E(E_cur, v, D_part, L_full, evnr, mu))
        #plt.imshow(E_cur, cmap='hot')
        #plt.show()
    return E_cur,v


if __name__ == '__main__':
    main()
