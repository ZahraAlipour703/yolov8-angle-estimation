import torch.nn as nn
from .autopad import autopad

class DWConv(nn.Module):
    """
    Depthwise + pointwise convolution block.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        # Depthwise conv (groups = c1)
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        # Pointwise conv
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)
