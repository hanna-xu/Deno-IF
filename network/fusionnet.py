import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.attn = MDTA(channels, num_heads)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(x.reshape(b, c, -1).transpose(-2, -1).contiguous().transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(x.reshape(b, c, -1).transpose(-2, -1).contiguous().transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class FusionNet(nn.Module):
    def __init__(self, num_blocks=[4, 4, 2], num_heads=[1, 2, 4], channels=[18, 36, 72, 144], num_refinement=2,
                 expansion_factor=2.66):
        super(FusionNet, self).__init__()

        self.embed_conv_vis = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.embed_conv_ir = nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders_vis = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        self.encoders_ir = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])


        self.reduces = nn.ModuleList([nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1, bias=False)])
        self.reduces.append(nn.Conv2d(channels[1] * 3, channels[1], kernel_size=1, bias=False))


        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])

        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[0] * 3, num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[0] * 3, num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[0] * 3, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, vis, ir, pre_f=True):
        fo_vis = self.embed_conv_vis(vis)
        fo_ir = self.embed_conv_ir(ir)

        vis_enc1 = self.encoders_vis[0](fo_vis)
        vis_enc2 = self.encoders_vis[1](self.downs[0](vis_enc1))
        vis_enc3 = self.encoders_vis[2](self.downs[1](vis_enc2))

        ir_enc1 = self.encoders_ir[0](fo_ir)
        ir_enc2 = self.encoders_ir[1](self.downs[0](ir_enc1))
        ir_enc3 = self.encoders_ir[2](self.downs[1](ir_enc2))

        reduce3 = self.reduces[0](torch.cat([vis_enc3, ir_enc3], dim=1))
        out_dec3 = self.decoders[0](reduce3)

        out_dec3_up = self.ups[1](out_dec3)
        reduce2 = self.reduces[1](torch.cat([out_dec3_up, vis_enc2, ir_enc2], dim=1))
        out_dec2 = self.decoders[1](reduce2)

        out_dec2_up = self.ups[2](out_dec2)
        out_dec1 = self.decoders[2](torch.cat([out_dec2_up, vis_enc1, ir_enc1], dim=1))

        fr = self.refinement(out_dec1)

        vis_ycbcr = rgb2ycbcr(vis)
        vis_y = vis_ycbcr[:, 0:1, :, :]
        vis_cb = vis_ycbcr[:, 1:2, :, :]
        vis_cr = vis_ycbcr[:, 2:3, :, :]
        max_y = (vis_y + ir) / 2
        max_ycbcr = torch.cat((max_y, vis_cb, vis_cr), dim=1)
        max_fused = ycbcr2rgb(max_ycbcr)
        if pre_f:
            out = torch.clamp(self.output(fr) + max_fused, 0, 1)
        else:
            out = torch.clamp(self.output(fr), 0, 1)
        return max_fused, out