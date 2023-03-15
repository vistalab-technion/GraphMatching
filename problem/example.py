import torch

from optimization.prox.prox import ProxNonNeg, ProxL21ForSymmCentdMatrixAndInequality
from problem.base import block_stochastic_graph, lap_from_adj
from problem.spectral_subgraph_localization import SubgraphIsomorphismSolver
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(12)

    n1 = 5
    n2 = 15
    n = n1 + n2
    p = block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.2)

    A = torch.tril(torch.bernoulli(p)).double()
    A = (A + A.T)
    D = torch.diag(A.sum(dim=1))
    L = lap_from_adj(A)

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n1, 0:n1]
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)
    params = {'maxiter': 100,
              'show_iter': 10,
              'mu_spectral': 1,
              'mu_l21': 1,
              'mu_MS': 1,
              'mu_split': 1,
              'mu_trace': 0.0,
              'lr': 0.02,
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
    subgraph_isomorphism_solver = \
        SubgraphIsomorphismSolver(L, ref_spectrum, params)
    v, E = subgraph_isomorphism_solver.solve()
    subgraph_isomorphism_solver.plot(plots)
    subgraph_isomorphism_solver.plot_on_graph(A.numpy().astype(int),
                                              n1,
                                              subgraph_isomorphism_solver.v,
                                              subgraph_isomorphism_solver.E)
