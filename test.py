import os
import argparse
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from network.fusionnet import FusionNet
from utils import *
from dataset import *
from network.unfolding_inference import Inference_2D, Inference_3D
from thop import profile
from datetime import datetime

eps=1e-6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device num, cuda:0')
    parser.add_argument('--testDir', type=str, default='./datasets/test/RS', help='path to test images')
    parser.add_argument('--ckpt', type=str, default='xx.pth',
                        help='path to *_best_model.pth, e.g. LLVIP_M3FD.pth, RoadScene_MSRS.pth')
    parser.add_argument('--outputDir', type=str, default='./results/', help='path to save the results')
    args = parser.parse_args()
    return args

args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

model_F = FusionNet()
model_F.to(device)
model_F.eval()

fusion_checkpoint = torch.load('./ckpt/' + args.ckpt, map_location=device)
model_F.load_state_dict(fusion_checkpoint['model_F_state_dict'])

test_dataset = Test_dataset(args.testDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

os.makedirs(args.outputDir, exist_ok=True)

T=[]

with torch.no_grad():
    for i, sample in enumerate(test_loader):
        start_time = datetime.now()
        names = sample['name']
        vis = sample['vis'].to(device)
        ir = sample['ir'].to(device)
        pre_f, fused_img = model_F(vis, ir, pre_f=False)
        fused_img = test_norm(fused_img)

        time = datetime.now() - start_time
        time = time.total_seconds()

        T.append(time)

        if i > 1:
            T.append(time)
            print("\nElapsed_time: %s" % (T[i - 2]))

        flops, params = profile(model_F, (vis, ir, ))
        print("FLOPs: %.2f" % (flops/1e9), "Parameters: %.2f" % params)
        # params = sum(p.numel() for p in model_F.parameters() if p.requires_grad)
        # print(f"参数量: {params}")

        for name, _ in zip(names, fused_img):
            print(name)
            result = torch.cat((vis, ir.repeat(1, 3, 1, 1), fused_img), dim=3)
            torchvision.utils.save_image(fused_img, args.outputDir + '/' + name)

    print("Time mean :%s, std: %s\n" % (np.mean(T), np.std(T)))
