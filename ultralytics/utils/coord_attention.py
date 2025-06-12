import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(channels // reduction, channels, 1)
        self.conv_w = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_gate(x) * x
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_gate(sa_input) * ca
        return sa
