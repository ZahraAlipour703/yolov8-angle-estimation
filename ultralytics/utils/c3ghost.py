import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad, Conv, C2f  # reuse existing helpers

class C3Ghost(nn.Module):
    """Ghost‐based C3 block: replaces the pointwise conv in C3 with GhostConv"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import GhostConv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_, c_, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv3(torch.cat([self.cv1(x), self.cv2(self.cv1(x))], dim=1))
        return y + x if self.add else y
