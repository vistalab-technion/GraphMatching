import torch
from torch import Tensor


def find_smallest_eigenvector(A):
    eigenvalues, eigenvectors = torch.symeig(A, eigenvectors=True)

    # Find the index of the smallest eigenvalue in abs val
    smallest_eigenvalue_index = torch.argmin(eigenvalues.abs())

    # Get the smallest eigenvalue and eigenvector
    smallest_eigenvalue = eigenvalues[smallest_eigenvalue_index]
    smallest_eigenvector = eigenvectors[:, smallest_eigenvalue_index]
    return smallest_eigenvalue, smallest_eigenvector


def inverse_power_method(A, epsilon=1e-5, max_iterations=100):
    # Initialize a random vector as the starting eigenvector estimate
    n = A.shape[0]
    v = torch.randn(n, 1)
    v = v / torch.norm(v)  # Normalize the vector

    # Perform the inverse power method
    for _ in range(max_iterations):
        v_prev = v

        # Compute the next estimate of the eigenvector
        v = torch.linalg.solve(A, v)
        v = v / torch.norm(v)  # Normalize the vector

        # Compute the corresponding eigenvalue
        eigenvalue = torch.mm(torch.mm(v_prev.t(), A), v) / torch.mm(v_prev.t(), v)

        # Check for convergence
        if torch.norm(v - v_prev) < epsilon:
            break

    return eigenvalue.item(), v


def spectral_subgraph_localization(A: Tensor,
                                   spectral_op,
                                   params,
                                   obj_fun,
                                   w: Tensor = None, ):
    if w is None:
        w = torch.ones(A.shape[0])
    n = A.shape[0]
    s = 1 / (params["scale"] ** 2)
    for i in range(params["maxiter"]):
        H = spectral_op(A, w, params)
        #C = (torch.eye(n) - s * w @ w.T) @ H @ H @ (torch.eye(n) - s * w @ w.T)
        # Compute eigenvalues and eigenvectors of A
        e, v = find_smallest_eigenvector(H)
        #e,v = inverse_power_method(A = H, epsilon=1e-8, max_iterations=500)
        # w = params["scale"] * v
        # w = v / torch.norm(v, float('inf'))
        # w = params["scale"] * w
        w = v / v.sum()
        #obj = w.T @ H @ w +params["c"]**2
        obj = obj_fun(A, w, params)
    return w, obj


#
# % md
#
# # op1 = lambda A, w, params: ((A.T @ w) @ (w.T @ A) - 2 * params["c"] * A+(params["c"]/params["scale"])**2*torch.eye(n))
# op1 = lambda A, w, params: (A - params["c"] * torch.eye(A.shape[0]))
# obj_fun = lambda A, w, params: (w.T @ A @ w - params["c"]) ** 2
# obj_quad_fun = lambda A, w, params: (w.T @ op1(A, w, params) @ w)
#
# w_gt = w_indicator[:, None].float() / w_indicator.sum()
# A_tensor = torch.tensor(A).float()
# c = w_gt.T @ A_tensor @ w_gt
# params = {}
# params["c"] = c
# params["maxiter"] = 1
# params["w_gt"] = w_gt
# params["scale"] = 1 / np.sqrt(len(G_sub.nodes))
#
#
# w_opt, obj_val = spectral_subgraph_localization(A=A_tensor,
#                                                 spectral_op=op1,
#                                                 params=params,
#                                                 w=w_gt,
#                                                 obj_fun=obj_fun)
# w_star = (w_opt.numpy() / w_opt.sum()) * m
# #w_star = w_opt
# plot_graph_with_colors(G, G_sub, w_star, 'w_star')
# plot_graph_with_colors(G, G_sub, w_gt * m, 'w_gt')