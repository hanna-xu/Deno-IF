import torch
import torch.nn as nn

class L(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, S, Ak_adj_MN, gamma):
        return (X - S + gamma * Ak_adj_MN)/(gamma + 1) # * k1 * k2)

class S(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, L, beta):
        return (X - L)/(beta + 1)

class M(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, N, Ak_L, alpha, gamma):
        I = torch.eye(N.shape[2], device=N.device, dtype=N.dtype)
        I = torch.unsqueeze(torch.unsqueeze(I, 0), 0)
        I = I.repeat(N.shape[0], N.shape[1], 1, 1)
        C = alpha * I + 2 * gamma * torch.matmul(N, N.transpose(2, 3))
        M = 2 * gamma * torch.matmul(torch.matmul(Ak_L, N.transpose(2, 3)), torch.linalg.pinv(C))
        return M

class N(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, Ak_L, alpha, gamma):
        I= torch.eye(M.shape[3], device=M.device, dtype=M.dtype)
        I = torch.unsqueeze(torch.unsqueeze(I, 0), 0)
        I = I.repeat(M.shape[0], M.shape[1], 1, 1)
        C = alpha * I + 2 * gamma * torch.matmul(M.transpose(2, 3), M)
        N = 2 * gamma * torch.matmul(torch.linalg.pinv(C), torch.matmul(M.transpose(2, 3), Ak_L))
        return N