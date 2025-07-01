import torch
import torch.nn as nn
from .autopad import autopad

class DWConv(nn.Module):
    """
    Depthwise‑separable conv: DW → PW → BN → Act
    Exposes .conv and .bn so that Ultralytics’ fuse() can fold BN into conv.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        # depthwise conv
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        # pointwise conv
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

        # *** Alias pw as conv so fuse() sees it ***
        self.conv = self.pw

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)
