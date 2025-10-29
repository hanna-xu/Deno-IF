# Deno-IF
Code of "Deno-IF: Unsupervised Noisy Visible and Infrared Image Fusion Method" (NeurIPS 2025 Spotlight).<br>
[Paper](https://github.com/hanna-xu/Deno-IF/blob/main/paper.pdf)

## Introduction
This paper proposes an unsupervised noisy visible and infrared image fusion method, termed as Deno-IF. It consists of two modules, including i) a convolutional low-rank optimization module and ii) a joint denoising and fusion module. 

Based on the convolutional lowrank property of high-quality data, the convolutional low-rank optimization module decomposes clean
component from the noisy input through convolution nuclear norm minimization in an unsupervised manner. The decomposed data provides the optimization guidance for the joint denoising and fusion module. 

The joint denoising and fusion module takes noisy source images as input and outputs the fused image. The network includes intra-modal recovery and inter-modal recovery and fusion, with self- and cross-modal attention to deal with complex individual-modal and complementary multi-modal information. 

The framework of this method is shown below:
