# Deno-IF
Code of "Deno-IF: Unsupervised Noisy Visible and Infrared Image Fusion Method" (NeurIPS 2025 Spotlight).<br>
[Paper](https://github.com/hanna-xu/Deno-IF/blob/main/paper.pdf)

## Introduction
This paper proposes an unsupervised noisy visible and infrared image fusion method, termed as Deno-IF. It consists of two modules:

First, based on the convolutional low-rank property of high-quality data, the convolutional low-rank optimization module decomposes clean
component from the noisy input through convolution nuclear norm minimization in an unsupervised manner. The decomposed data provides the optimization guidance for the joint denoising and fusion module. 

Then, the joint denoising and fusion module takes noisy source images as input and outputs the fused image. The network includes intra-modal recovery and inter-modal recovery and fusion, with self- and cross-modal attention to deal with complex individual-modal and complementary multi-modal information. 

The framework of this method is shown below:
<div align=center><img src="[https://github.com/hanna-xu/others/blob/master/images/URFusion_framework.jpg](https://github.com/hanna-xu/others/blob/master/images/Deno-IF_framework.png)" width="870" height="512"/></div>
<br>
