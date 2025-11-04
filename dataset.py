import glob
import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import *

def ori(x):
    return x


def apply_gamma_high(x):
    result = x/255.0 ** random.uniform(0.9, 1) *255.0
    return np.uint8(np.clip(result, 0, 255))


def gaussian_noise(x):
    x = np.float32(x)
    mean = 0
    sigma = random.uniform(0.05, 0.2)
    noise = (np.random.normal(mean, sigma, x.shape)) * 255
    return np.uint8(np.clip(x + noise, 0, 255))


def speckle_noise(image):
    mean = 0
    var = random.uniform(0.05, 0.2)
    image = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_image = image * (1 + noise)
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image


def miss(x):
    mask = np.random.choice([0, 1], size=x.shape, p=[0.4, 0.6]).astype(x.dtype)
    return x * mask


class Train_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/vis'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/ir'
        print(self.source2_dir)

        self.s1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.s2 = [im_name for im_name in os.listdir(self.source2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source1 = self.s1
        self.source2 = self.s2

        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        source1_path = os.path.join(self.source1_dir, name)
        img1_o = cv2.imread(source1_path)
        img1_o = cv2.cvtColor(img1_o, cv2.COLOR_BGR2RGB)

        source2_path = os.path.join(self.source2_dir, name)
        img2_o = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)

        shape=img1_o.shape

        ratio = random.uniform(0.4, 1.1)
        H = np.max([round(shape[1] * ratio), 156])
        W = np.max([round(shape[1] * ratio), 156])
        img1 = cv2.resize(img1_o, (W, H))
        img2 = cv2.resize(img2_o, (W, H))

        distortions = {1: gaussian_noise, 2: speckle_noise}

        img1_r = np.copy(img1)
        img2_r = np.copy(img2)

        dist1 = random.randint(1, 2)
        func1 = distortions.get(dist1)
        img1_r = func1(img1_r)

        dist2 = random.randint(1, 2)
        func2 = distortions.get(dist2)
        img2_r = func2(img2_r)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1_r = self.transform(img1_r)
            torch.random.manual_seed(seed)
            img2_r = self.transform(img2_r)
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'vis': img1_r, 'ir': img2_r, 'vis_ori': img1, 'ir_ori': img2}
        return sample


class Train_dataset_ori(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/vis'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/ir'
        print(self.source2_dir)

        self.s1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.s2 = [im_name for im_name in os.listdir(self.source2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source1 = self.s1
        self.source2 = self.s2

        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        source1_path = os.path.join(self.source1_dir, name)
        img1 = cv2.imread(source1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        source2_path = os.path.join(self.source2_dir, name)
        img2 = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)

        distortions = {1: gaussian_noise, 2: miss}

        img1_r = np.copy(img1)
        img2_r = np.copy(img2)

        dist1 = random.randint(1, 1)
        func1 = distortions.get(dist1)
        img1_r = func1(apply_gamma_high(img1_r))

        dist2 = random.randint(1, 1)
        func2 = distortions.get(dist2)
        img2_r = func2(img2_r)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1_r = self.transform(img1_r)
            torch.random.manual_seed(seed)
            img2_r = self.transform(img2_r)
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'vis_inference': img1, 'ir_inference': img2, 'vis': img1_r, 'ir': img2_r}
        return sample


class Test_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/vis'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/ir'
        print(self.source2_dir)

        self.source1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source2 = [im_name for im_name in os.listdir(self.source2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        img1 = cv2.imread(os.path.join(self.source1_dir, name))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(os.path.join(self.source2_dir, name), cv2.IMREAD_GRAYSCALE)

        height, width, _ = img1.shape
        if height % 8 != 0 or width % 8 != 0:
            img1 = img1[0:height//8*8, 0:width//8*8,:]
            img2 = img2[0:height//8*8, 0:width//8*8]

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'vis': img1,'ir':img2}
        return sample