import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_msssim
import torchvision
from torch.nn import init
import time
from PIL import Image
import glob
import math
import pytorch_msssim


def compute_alpha(x):
    B,C,H,W = x.shape
    if C==1:
        grad_x = gradient_la(x)
    else:
        x_ycbcr = rgb2ycbcr(x)
        grad_x = gradient_la(x_ycbcr[:,0:1,:,:])
    if C==1:
        grad_blur_x = gradient_la(gaussian_blur(x))
    else:
        blur_ycbcr = rgb2ycbcr(gaussian_blur(x))
        grad_blur_x = gradient_la(blur_ycbcr[:,0:1,:,:])
    grad_change = torch.mean(torch.abs(grad_x-grad_blur_x), dim=[1, 2, 3])
    return grad_change


def gaussian_blur(x):
    blur_layer = get_gaussian_kernel(channels=x.shape[1], device=x.device)
    for k in range(2):
        x = blur_layer(x)
    return x


def get_gaussian_kernel(kernel_size=5, sigma=1, channels=3, device=0):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter.to(device)


def save_TensorImg(img_tensor, path, nrow=1):
    torchvision.utils.save_image(img_tensor, path, nrow=nrow)


def np_save_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img * 255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def test_norm(input_tensor):
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def rgb2ycbcr(img_rgb):
    img_rgb=torch.clamp(img_rgb, min=0,max=1)
    R = img_rgb[:, 0, :, :] * 255
    G = img_rgb[:, 1, :, :] * 255
    B = img_rgb[:, 2, :, :] * 255
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128.0
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
    img_ycbcr = torch.stack((Y, Cb, Cr), dim=1)
    return torch.clamp(img_ycbcr/255.0, min=0,max=1)


def ycbcr2rgb(img_ycbcr):
    img_ycbcr = torch.clamp(img_ycbcr, min=0, max=1)
    Y = img_ycbcr[:, 0, :, :] * 255
    Cb = img_ycbcr[:, 1, :, :] * 255
    Cr = img_ycbcr[:, 2, :, :] * 255
    R=1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    img_rgb = torch.stack((R, G, B), dim=1)
    return torch.clamp(img_rgb/255.0, min=0,max=1)


def ssim_loss(x, y):
    loss_ssim = 1 - pytorch_msssim.ssim(x, y, data_range=1, size_average=False)
    return loss_ssim


def gradient(x):
    sobel = torch.tensor([[0, -1, 0], [-1, 2, 0], [0, 0, 0]], device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    grad = F.conv2d(x, sobel, padding=1)
    return grad


def gradient_la(x):
    sobel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    grad = F.conv2d(x, sobel, padding=1)
    return grad


def nuclear_norm(x):
    batchsize = x.shape[0]
    for b in range(batchsize):
        x_each = x[b,0,:,:]
        loss_nuclear_norm = torch.linalg.norm(x_each, ord='nuc')
        if b==0:
            loss = loss_nuclear_norm
        else:
            loss = loss + loss_nuclear_norm
    return loss/batchsize


def rgb2gray(x):
    return 0.03 * x[:, 0:1, :, :] + 0.59 * x[:, 1:2, :, :] + 0.11 * x[:, 2:3, :, :]


def rgb2gray(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 0:1, :, :]
    B = tensor[:, 0:1, :, :]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B


def ycbcr2rgb(img_ycbcr):
    img_ycbcr = torch.clamp(img_ycbcr, min=0, max=1)
    Y = img_ycbcr[:, 0, :, :] * 255
    Cb = img_ycbcr[:, 1, :, :] * 255
    Cr = img_ycbcr[:, 2, :, :] * 255
    R=1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    img_rgb = torch.stack((R, G, B), dim=1)
    return torch.clamp(img_rgb/255.0, min=0,max=1)


def save_ckpt(state, is_best, experiment, epoch, ckpt_dir):
    filename = os.path.join(ckpt_dir, f'{experiment}_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        print(f'[BEST MODEL] Saving best model, obtained on epoch = {epoch + 1}')
        shutil.copy(filename, os.path.join(ckpt_dir, f'{experiment}_best_model.pth'))


def gamma_tensor(img, gamma=0.85):
    return torch.pow(img, gamma)


def gamma_correction(img, gamma):
    return np.power(img, gamma)


def gamma_like(img, enhanced):
    x, y = img.mean(), enhanced.mean()
    gamma = np.log(y) / np.log(x)
    return gamma_correction(img, gamma)


def to_numpy(t, squeeze=False, to_HWC=True):
    x = t.detach().cpu().numpy()
    if squeeze:
        x = x.squeeze()
    if to_HWC:
        x = x.transpose((1, 2, 0))
    return x


def make_grid(dataset, vsep=8):
    n = len(dataset)
    img = to_numpy(dataset[0]['img'])
    h, w, _ = img.shape
    grid = np.ones((n * h + vsep * (n - 1), 4 * w, 3), dtype=np.float32)
    return grid, vsep


def create_dir(path):
    'create directory if not exist'
    if isinstance(path, str):
        path = Path(path).expanduser().resolve()

    if path.exists():
        if path.is_dir():
            print('Output dir already exists.')
        else:
            sys.exit('[ERROR] You specified a file, not a folder. Please revise --outputDir')
    else:
        path.mkdir(parents=True)
    return path
