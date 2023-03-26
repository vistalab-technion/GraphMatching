import torch
from tqdm import tqdm


class QRPowerMethod:

    def __init__(self, a_tol, r_tol, max_iter):
        self.max_iter = max_iter
        self.a_tol = a_tol
        self.r_tol = r_tol

    def solve(self, A, X, shift_matrix):
        L, U = torch.lu(A + shift_matrix)
        for iter in tqdm(range(self.max_iter)):
            Y = torch.linalg.solve(L, X)
            Z = torch.linalg.solve(U, Y)
            X,_ = torch.linalg.qr(Z)
        return X
