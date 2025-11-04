import os
import sys
import time
import shutil
import argparse
from datetime import datetime
from subprocess import call
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision
from dataset import *
from utils import *
from network.fusionnet import FusionNet
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from network.unfolding_inference import Inference_2D, Inference_3D, Ak_3D

eps = 1e-6

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=int, default=1, help='device name, cuda:0')
	parser.add_argument('--experiment', default='fusionnet', help='name of the experiment')
	parser.add_argument('--past_ckpt', default='1_100', help='ckpt name of the past model')
	parser.add_argument('--trainDir', type=str, default='./datasets/train', help='path to train images')
	parser.add_argument('--testDir', type=str, default='./datasets/test/', help='path to test images')
	parser.add_argument('--numEpoch', type=int, default=80)
	parser.add_argument('--patchsize', type=int, default=120)
	parser.add_argument('--batchsize', type=int, default=4)
	args = parser.parse_args()
	return args


def train(experiment, loaders, model_F, optimizer, vis_unfolding_inference, ir_unfolding_inference, writer, epoch, num_epochs, begin_time, begin_epoch, begin_step, device, **kwargs):
	model_F = model_F.train()

	print("begin epoch: ", begin_epoch)
	print("epoch: ", epoch)

	total_i = len(loaders['train'])
	step = begin_step

	if epoch ==0:
		if begin_step % (total_i + 1) == total_i:
			this_epoch = begin_epoch + 1
		else:
			this_epoch = begin_epoch
	else:
		this_epoch = begin_epoch + epoch

	print(f'--- Epoch {this_epoch} ---')

	this_epoch_end = 0
	for i, sample in enumerate(loaders['train']):
		this_epoch_step = step % total_i

		if this_epoch_end == 0:
			optimizer.zero_grad()
			vis_batch = sample['vis'].to(device)
			ir_batch = sample['ir'].to(device)
			patchsize = vis_batch.shape[-1]
			batchsize = vis_batch.shape[0]

			vis_ycbcr = rgb2ycbcr(vis_batch)
			vis_y = vis_ycbcr[:,0:1,:,:]
			vis_cb = vis_ycbcr[:, 1:2, :, :]
			vis_cr = vis_ycbcr[:, 2:3, :, :]

			w_alpha_vis = torch.round(compute_alpha(vis_batch) * 200)
			w_alpha_ir = torch.round(compute_alpha(ir_batch) * 80)
			w_gamma_vis = torch.round(compute_alpha(vis_batch) * 20)
			w_gamma_ir = torch.round(compute_alpha(ir_batch) * 16)

			vis_inference = vis_unfolding_inference.unfolding(vis_batch, w_alpha=w_alpha_vis, w_beta=2,
															w_gamma=w_gamma_vis, iter=20)

			ir_inference = ir_unfolding_inference.unfolding(ir_batch, w_alpha=w_alpha_ir, w_beta=2,
															w_gamma=w_gamma_ir, iter=20)

			vis_inference_ycbcr = rgb2ycbcr(vis_inference)
			vis_y_inference = vis_inference_ycbcr[:, 0:1, :, :]
			vis_cb_inference = vis_inference_ycbcr[:, 1:2, :, :]
			vis_cr_inference = vis_inference_ycbcr[:, 2:3, :, :]

			fused_max, fused_batch = model_F(vis_batch, ir_batch, pre_f=False)

			crit = nn.L1Loss()

			fused_ycbcr = rgb2ycbcr(fused_batch)
			fused_y = fused_ycbcr[:,0:1,:,:]

			'''intensity loss'''
			inten_max = torch.max(vis_y_inference, ir_inference)
			loss_y_inten = crit(inten_max, fused_y)

			'''gradient loss'''
			grad_vis_y_inference = gradient(vis_y_inference)
			grad_ir_inference = gradient(ir_inference)
			grad_fused_y = gradient(fused_y)
			grad_max_mask = (torch.abs(grad_vis_y_inference) > torch.abs(grad_ir_inference)).float()

			grad_max = grad_vis_y_inference * grad_max_mask + grad_ir_inference * (1 - grad_max_mask)
			loss_y_grad = crit(grad_max, grad_fused_y)

			'''cb cr loss'''
			fused_cb = fused_ycbcr[:, 1:2, :, :]
			fused_cr = fused_ycbcr[:, 2:3, :, :]

			loss_cbcr = crit(vis_cb_inference, fused_cb) + crit(vis_cr_inference, fused_cr)

			'''nuclear loss'''
			Ak_fused = Ak_3D(fused_batch, k1=patchsize//10, k2=patchsize//10, k3=2)
			loss_nuclear_norm = nuclear_norm(Ak_fused)

			loss = loss_y_inten + 1000 * loss_y_grad + 30 * loss_cbcr + 0.000001 * (this_epoch-1) * loss_nuclear_norm

			loss.backward()
			optimizer.step()

			timeElapsed = datetime.now() - begin_time

			if this_epoch_step % 10 == 0:
				print("\n vis alpha: ", w_alpha_vis)
				print("vis gamma: ", w_gamma_vis)
				print("ir alpha: ", w_alpha_ir)
				print("ir gamma: ", w_gamma_ir)
				if this_epoch_step == 0:
					print('k1=k2=36 Epoch: [%d/%d], Iter: [%d/%d], Step:%d, Loss: %.5f, Time: \n' % (
						this_epoch, num_epochs, total_i, total_i, step, loss.item()), timeElapsed)
				else:
					print('k1=k2=36 Epoch: [%d/%d], Iter: [%d/%d], Step:%d, Loss: %.5f, Time: ' % (
						this_epoch, num_epochs, this_epoch_step, total_i, step, loss.item()), timeElapsed)

				writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)
				writer.add_scalar(tag="loss_y_inten", scalar_value=loss_y_inten, global_step=step)
				writer.add_scalar(tag="loss_y_grad", scalar_value=loss_y_grad, global_step=step)
				writer.add_scalar(tag="loss_cbcr", scalar_value=loss_cbcr, global_step=step)
				writer.add_scalar(tag="loss_nuclear_norm", scalar_value=loss_nuclear_norm, global_step=step)

				N_show = 2
				writer.add_image("vis", torchvision.utils.make_grid(vis_batch[0:N_show, :, :, :]), global_step=step)
				writer.add_image("ir", torchvision.utils.make_grid(ir_batch[0:N_show, :, :, :]), global_step=step)

				writer.add_image("vis_y", torchvision.utils.make_grid(vis_y[0:N_show, :, :, :]),
								 global_step=step)

				writer.add_image("vis_inference", torchvision.utils.make_grid(vis_inference[0:N_show, :, :, :]),
								 global_step=step)

				writer.add_image("ir_inference", torchvision.utils.make_grid(ir_inference[0:N_show, :, :, :]), global_step=step)

				writer.add_image("fused", torchvision.utils.make_grid(fused_batch[0:N_show, :, :, :]), global_step=step)
				writer.add_image("fused_max", torchvision.utils.make_grid(fused_max[0:N_show, :, :, :]), global_step=step)
				writer.add_image("a_normal", torchvision.utils.make_grid(vis_batch[0:N_show, :, :, :]), global_step=step)

			if step % 100 ==0:
				state_dict = {'model_F_state_dict': model_F.state_dict(), 'step': step, 'epoch': this_epoch}
				torch.save(state_dict, './ckpt/'+ experiment+'/' + str(this_epoch) + '_' + str(step) + '_ckpt.pth')
			prev = datetime.now()

		if this_epoch_step == 0:
			break

		step = step + 1
	return step



args = parse_args()

if torch.cuda.is_available():
	device = torch.device(f'cuda:{args.device}')
	torch.cuda.manual_seed(1234)
else:
	device = torch.device('cpu')
	torch.manual_seed(1234)

train_dir = args.trainDir

trans_to_tensor = transforms.ToTensor()
trans_crop = transforms.RandomCrop(args.patchsize, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
trans_compose = transforms.Compose([trans_to_tensor, trans_crop])
train_dataset = Train_dataset(train_dir, transform=trans_compose)


trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)

loaders = {'train': trainloader}

hp = dict(lr=0.0001, wd=0, lr_decay_factor=0.995)

model_F = FusionNet()
model_F.to(device)

IR_Inference = Inference_2D().cuda()
IR_Inference.to(device)

VIS_Inference = Inference_3D().cuda()
VIS_Inference.to(device)

optimizer = optim.Adam(model_F.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

experiment=args.experiment

ckpt_dir = './ckpt/' + experiment +'/'
os.makedirs(ckpt_dir, exist_ok=True)
writer = SummaryWriter(log_dir='./logs/'+experiment)
num_epochs = args.numEpoch
checkpoint_path = ckpt_dir + args.past_ckpt + '_ckpt.pth'

if os.path.exists(checkpoint_path):
	print(f'---Continue Training---')
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model_F.load_state_dict(checkpoint['model_F_state_dict'])
	begin_epoch = checkpoint['epoch']
	begin_step = checkpoint['step']
	print("begin epoch: ", begin_epoch)
	print("begin step: ", begin_step)
else:
	begin_epoch = 1
	begin_step = 1

print(f'[START TRAINING JOB] on {datetime.now().strftime("%d %Y %H:%M:%S")}')
begin_time = datetime.now()



for epoch in range(num_epochs-begin_epoch+1):
	if epoch==0:
		step2 = train(experiment, loaders, model_F, optimizer, VIS_Inference, IR_Inference, writer, epoch, num_epochs,
					 begin_time, begin_epoch, begin_step, device)
	else:
		step2 = train(experiment, loaders, model_F, optimizer, VIS_Inference, IR_Inference, writer, epoch, num_epochs,
					 begin_time, begin_epoch, step1 + 1, device)
	step1 = step2


