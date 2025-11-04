# Deno-IF
Code of "Deno-IF: Unsupervised Noisy Visible and Infrared Image Fusion Method" (NeurIPS 2025 Spotlight).<br>
[Paper](https://github.com/hanna-xu/Deno-IF/blob/main/paper.pdf)

## Introduction
This paper proposes an unsupervised noisy visible and infrared image fusion method, termed as Deno-IF. It consists of two modules:

First, based on the convolutional low-rank property of high-quality data, the convolutional low-rank optimization module decomposes clean
component from the noisy input through convolution nuclear norm minimization in an unsupervised manner. The decomposed data provides the optimization guidance for the joint denoising and fusion module. 

Then, the joint denoising and fusion module takes noisy source images as input and outputs the fused image. The network includes intra-modal recovery and inter-modal recovery and fusion, with self- and cross-modal attention to deal with complex individual-modal and complementary multi-modal information. 

The framework of this method is shown below:
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/Deno-IF_framework.png" width="870" height="512"/></div>
<br>


## Recommended Environment
python=3.10<br>
pytorch=1.13<br>
pytorch-cuda=11.7<br>
numpy=1.24.4<br>
imageio=2.34.2<br>
opencv-python=4.10<br>
pandas=2.0.3<br>
pillow=10.4<br>
scikit-image=0.21<br>
scipy=1.10.1<br>

## __To train:__
* Prepare the training data:<br>
  Put the training data in `./datasets/train/vis/` and `./datasets/train/ir/`
* Run `python train.py`


## __To Test:__
* Prepare the test data:<br>
  Put the test data in `./datasets/test/vis/` and `./datasets/test/ir/`
* Choose the pretrained model:<br>
  Set `--ckpt` in `test.py` as one of the following file path:<br>
  i) `LLVIP_M3FD.pth` (for source images with relatively high resolution)<br>
  ii) `RoadScene_MSRS.pth` (for source images with relatively low resolution).
* Run `python test.py`
