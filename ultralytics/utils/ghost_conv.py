import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class GhostConv(nn.Module):
    """
    Ghost Convolution:
    Reduces FLOPs by generating half the channels via a cheap transform.
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        # hidden channels at least 1
        c_ = max(1, c2 // 2)
        # first Conv
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        # ensure groups positive and divides channels
        groups = 1 if c_ < 1 else min(c_, c_)
        self.cv2 = Conv(c_, c_, 5, 1, None, groups, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)
