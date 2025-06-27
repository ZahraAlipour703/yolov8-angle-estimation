# custom_blocks.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad, Conv      # autopad & Conv are here
from ultralytics.nn.modules.block import C2f                # C2f is in block, not conv

class C3Ghost(nn.Module):
    """A GhostNet‐style C3 module."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(C2f(c_, c_, n, shortcut, g) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y = torch.cat((y1, y2), dim=1)
        return self.cv3(self.m(y))

# (Similarly define MobileViTBlock, BiFPN, DyHead here...)
