import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Math_Module import L, S, M, N
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time


def pinv(x):
    B, C, H, W = x.shape
    x_pinvs = torch.empty((B, C, W, H), dtype=x.dtype, device=x.device)

    for i in range(B):
        x_i = x[i, 0:1, :, :]
        x_pinv_i = torch.linalg.pinv(x_i)
        x_pinvs[i, 0:1, :, :] = x_pinv_i
    return x_pinvs

def Ak(X, k1, k2):
    B, C, m1, m2 = X.shape
    A = torch.zeros((B, C, m1 * m2, k1 * k2), dtype=X.dtype, device=X.device)
    for b in range(B):
        for c in range(C):
            Xj = X[b, c, :, :].clone()
            for j in range(k2):
                Xi = Xj.clone()
                for i in range(k1):
                    # Assign the vectorized version of the shifted matrix to the appropriate column
                    A[b, c, :, j * k1 + i] = Xi.flatten()
                    # Circular shift along the first dimension (rows)
                    if i < k1 - 1:
                        Xi = torch.roll(Xi, shifts=1, dims=0)

                # Circular shift along the second dimension (columns)
                if j < k2 - 1:
                    Xj = torch.roll(Xj, shifts=1, dims=1)
    return A


def Ak_adj(X, H, W, k1, k2):
    B, C, _, _ = X.shape
    # Initialize output tensor
    A = torch.zeros((B, C, H, W), dtype=X.dtype, device=X.device)

    # Iterate over kernel dimensions
    for j in range(k2):
        for i in range(k1):
            # Extract the corresponding slice from M
            Xi = X[:, :, :, j * k1 + i]
            Xi = Xi.view(B, C, H, W)
            Xi = torch.roll(Xi, shifts=(-i, - j), dims=(2, 3))
            A += Xi
    return A/(k1*k2)


def Ak_3D(X, k1, k2, k3):
    B, m3, m1, m2 = X.shape
    # Initialize the result matrix
    A = torch.empty((B, 1, m1 * m2 * m3, k1 * k2 * k3), dtype=X.dtype, device=X.device)

    for b in range(B):
        # Copy of X to apply circular shifts
        Xk = X[b, :, :, :].clone()
        for k in range(k3):
            Xj = Xk.clone()
            for j in range(k2):
                Xi = Xj.clone()
                for i in range(k1):
                    # Assign the vectorized version of the shifted matrix to the appropriate column
                    A[b, 0:1, :, (k * k2 + j) * k1 + i] = Xi.flatten()
                    if i < k1 - 1:
                        Xi = torch.roll(Xi, shifts=1, dims=1)
                if j < k2 - 1:
                    Xj = torch.roll(Xj, shifts=1, dims=2)
            if k < k3 - 1:
                Xk = torch.roll(Xk, shifts=1, dims=0)
    return A


def Ak_adj_3D(X, H, W, C, k1, k2, k3):
    B = X.shape[0]
    A = torch.zeros((B, C, H, W), dtype=X.dtype, device=X.device)
    for k in range(k3):
        for j in range(k2):
            for i in range(k1):
                Xi = X[:, 0, :, (k * k2 + j) * k1 + i].clone()
                Xi_reshape = Xi.reshape(B, C, H, W)
                Xi_reshape = torch.roll(Xi_reshape, shifts=(-k, -i, -j), dims=(1, 2, 3))
                A += Xi_reshape
    return A/(k1 * k2 * k3)


class Inference_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = L()
        self.S = S()
        self.M = M()
        self.N = N()

    def unfolding(self, input_img, w_alpha=10, w_beta=2, w_gamma=10, iter=40):
        B, C, H, W = input_img.shape
        mk = (H * W)//64
        k1 = 36
        k2 = 36


        for t in range(iter):
            if t == 0:
                L = input_img.detach().clone().to(input_img.device)
                S = torch.clamp(torch.randn_like(input_img), 0, 1).to(input_img.device)
                m1 = H
                m2 = W
                M = torch.eye(m1 * m2, mk, device=input_img.device, dtype=input_img.dtype)
                M = torch.unsqueeze(torch.unsqueeze(M, 0), 0).repeat(B, 1, 1, 1)
                N = torch.eye(mk, k1 * k2, device=input_img.device, dtype=input_img.dtype)
                N = torch.unsqueeze(torch.unsqueeze(N, 0), 0).repeat(B, 1, 1, 1)
            else:
                MN = torch.matmul(M, N)
                Ak_adj_MN = Ak_adj(MN, H, W, k1, k2)

                L = self.L(X=input_img, S=S, Ak_adj_MN=Ak_adj_MN, gamma=w_gamma)
                S = self.S(X=input_img, L=L, beta=w_beta)
                Ak_L = Ak(L, k1=k1, k2=k2)
                M = self.M(N=N, Ak_L=Ak_L, alpha=w_alpha, gamma=w_gamma)
                N = self.N(M=M, Ak_L=Ak_L, alpha=w_alpha, gamma=w_gamma)
        return L


class Inference_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = L()
        self.S = S()
        self.M = M()
        self.N = N()

    def unfolding(self, input_img, w_alpha=10, w_beta=2, w_gamma=10, iter=40):
        B, C, H, W = input_img.shape
        mk = (H * W)//64
        k1 = 36
        k2 = 36
        k3 = 2

        for t in range(iter):
            if t == 0:
                L = input_img.detach().clone().to(input_img.device)
                S = torch.clamp(torch.randn_like(input_img), 0, 1).to(input_img.device)
                m1 = H
                m2 = W
                m3 = C
                M = torch.eye(m1 * m2 * m3, mk, device=input_img.device, dtype=input_img.dtype)
                M = torch.unsqueeze(torch.unsqueeze(M, 0), 0).repeat(B, 1, 1, 1)
                N = torch.eye(mk, k1 * k2 * k3, device=input_img.device, dtype=input_img.dtype)
                N = torch.unsqueeze(torch.unsqueeze(N, 0), 0).repeat(B, 1, 1, 1)
            else:
                MN = torch.matmul(M, N)
                Ak_adj_MN = Ak_adj_3D(MN, H=H, W=W, C=C, k1=k1, k2=k2, k3=k3)

                L = self.L(X=input_img, S=S, Ak_adj_MN=Ak_adj_MN, gamma=w_gamma)
                S = self.S(X=input_img, L=L, beta=w_beta)
                Ak_L = Ak_3D(X=L, k1=k1, k2=k2, k3=k3)
                M = self.M(N=N, Ak_L=Ak_L, alpha=w_alpha, gamma=w_gamma)
                N = self.N(M=M, Ak_L=Ak_L, alpha=w_alpha, gamma=w_gamma)
        return L


class L(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, S, Ak_adj_MN, gamma):
        gamma = gamma.view(X.shape[0], 1, 1, 1).repeat(1, 1, Ak_adj_MN.shape[2], Ak_adj_MN.shape[3])
        return torch.clamp((X - S + gamma * Ak_adj_MN)/(gamma + 1), 0, 1)

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
        I = I.repeat(N.shape[0], 1, 1, 1)
        alpha = alpha.view(I.shape[0], 1, 1, 1).repeat(1, 1, I.shape[2], I.shape[3])
        gamma_1 = gamma.view(I.shape[0], 1, 1, 1).repeat(1, 1, I.shape[2], I.shape[3])
        C = alpha * I + 2 * gamma_1 * torch.matmul(N, N.transpose(2, 3))
        X = torch.matmul(torch.matmul(Ak_L, N.transpose(2, 3)), pinv(C))
        gamma_2 = gamma.view(X.shape[0], 1, 1, 1).repeat(1, 1, X.shape[2], X.shape[3])
        M = 2 * gamma_2 * X
        return M

class N(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, M, Ak_L, alpha, gamma):
        I= torch.eye(M.shape[3], device=M.device, dtype=M.dtype)
        I = torch.unsqueeze(torch.unsqueeze(I, 0), 0)
        I = I.repeat(M.shape[0], 1, 1, 1)
        alpha = alpha.view(I.shape[0], 1, 1, 1).repeat(1, 1, I.shape[2], I.shape[3])
        gamma_1 = gamma.view(I.shape[0], 1, 1, 1).repeat(1, 1, I.shape[2], I.shape[3])
        C = alpha * I + 2 * gamma_1 * torch.matmul(M.transpose(2, 3), M)
        X = torch.matmul(pinv(C), torch.matmul(M.transpose(2, 3), Ak_L))
        gamma_2 = gamma.view(X.shape[0], 1, 1, 1).repeat(1, 1, X.shape[2], X.shape[3])
        N = 2 * gamma_2 * X
        return N