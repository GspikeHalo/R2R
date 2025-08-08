"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), downsample=False):
        super().__init__()
        self.actv = actv
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        res = self._residual(x)
        skip = self._shortcut(x)
        return (res + skip) / math.sqrt(2)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.style_fc = nn.Linear(style_dim, in_ch)
        self.padding = kernel_size // 2
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.noise_weight = nn.Parameter(torch.zeros(out_ch))
        self.scale = 1 / math.sqrt(in_ch * kernel_size * kernel_size)

    def forward(self, x, s):
        B, C, H, W = x.shape
        style = self.style_fc(s).view(B, 1, C, 1, 1)
        w = self.weight * (style + 1)
        if self.demodulate:
            d = torch.rsqrt((w * w).sum([2, 3, 4]) + self.eps)
            w = w * d.view(B, -1, 1, 1, 1)
        w = w * self.scale
        x_ = x.view(1, -1, H, W)
        w_ = w.view(B * w.size(1), w.size(2), w.size(3), w.size(4))
        out = F.conv2d(x_, w_, padding=self.padding, groups=B)
        out = out.view(B, -1, H, W)
        out = out + self.bias.view(1, -1, 1, 1)
        noise = torch.randn(B, 1, H, W, device=out.device)
        out = out + noise * self.noise_weight.view(1, -1, 1, 1)
        return out

class ModulatedResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out)
        self.conv1 = ModulatedConv2d(dim_in, dim_out, 3, style_dim, demodulate=True)
        self.conv2 = ModulatedConv2d(dim_out, dim_out, 3, style_dim, demodulate=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        if self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic')

    def _shortcut(self, x):
        if self.upsample:
            x = self.up(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.upsample:
            x = self.up(x)
        x = self.conv1(x, s)
        x = self.actv(x)
        x = self.conv2(x, s)
        x = self.actv(x)
        return x

    def forward(self, x, s):
        res = self._residual(x, s)
        skip = self._shortcut(x)
        return (res + skip) / math.sqrt(2)

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_raw = nn.Conv2d(4, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_raw = nn.Conv2d(dim_in, 4, 1, 1, 0)

        repeat_num = int(math.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, actv=nn.LeakyReLU(0.2), downsample=True)
            )
            self.decode.insert(
                0, ModulatedResBlk(dim_out, dim_in, style_dim, upsample=True)
            )
            dim_in = dim_out

        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, actv=nn.LeakyReLU(0.2), downsample=False)
            )
            self.decode.insert(
                0, ModulatedResBlk(dim_out, dim_out, style_dim)
            )

    def forward(self, x, s):
        x = self.from_raw(x)
        skips = []
        for block in self.encode:
            skips.append(x)
            x = block(x)
        for idx, block in enumerate(self.decode):
            x = block(x, s)
            skip = skips[-idx-1]
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bicubic')
            x = x + skip
        return self.to_raw(x)

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema
