# utils/dwconv.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=k, stride=s,
                                padding=k // 2 if p is None else p,
                                groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))
