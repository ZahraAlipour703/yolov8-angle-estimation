import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class DyConv(nn.Module):
    """
    DyConv: A tiny conditional convolution (a la CondConv / Dynamic Conv).
    We keep E experts and a per-sample soft routing to blend them.
    Args:
        in_channels:  int
        out_channels: int
        k:            kernel size
        stride:       stride
        padding:      padding (None→auto)
        groups:       number of experts
    """
    def __init__(self, in_channels, out_channels, k=3, stride=1, padding=None, groups=4, reduction=4):
        super().__init__()
        self.groups = groups
        # Shared routing MLP
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(in_channels // reduction, groups, bias=True)
        # Experts: group of small Conv layers
        pad = (k // 2 if padding is None else padding)
        self.experts = nn.ModuleList([
            Conv(in_channels, out_channels, k, stride, pad) for _ in range(groups)
        ])

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, _, _ = x.shape
        # routing weights
        w = self.pool(x).view(B, C)
        w = self.act(self.fc1(w))
        w = torch.softmax(self.fc2(w), dim=1)  # [B, groups]
        # compute each expert
        outs = [e(x).unsqueeze(1) for e in self.experts]  # list of [B,1,Co,H,W]
        outs = torch.cat(outs, dim=1)                     # [B,groups,Co,H,W]
        # blend
        w = w.view(B, self.groups, 1, 1, 1)               # [B,groups,1,1,1]
        out = (outs * w).sum(dim=1)                       # [B,Co,H,W]
        return out
