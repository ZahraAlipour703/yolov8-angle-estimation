import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    """
    Coordinate Attention — Hou et al., 2021.
    Captures long-range dependencies along one spatial direction
    and encodes precise positional information along the other.
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # Horizontal pooling: (B, C, H, 1)
        x_h = nn.functional.adaptive_avg_pool2d(x, (h, 1))
        # Vertical pooling: (B, C, 1, W)
        x_w = nn.functional.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        # Fuse
        y = torch.cat([x_h, x_w], dim=2)          # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # Split and re‐shape
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w
