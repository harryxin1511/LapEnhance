import argparse
import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os
import random
from PIL import Image
from torch.utils import data


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding        outdim自己设的
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, out_dim=48):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
   #     self.H, self.W = img_size // patch_size, img_size // patch_size
        self.proj = nn.Conv2d(in_chans, out_dim, kernel_size=patch_size, stride=stride, padding = (patch_size // 2 ) )# 让输入和输出具有相同的高和宽)
        self.norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        # _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = torch.reshape(x,(x.shape[0], x.shape[2], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)))

        return x



