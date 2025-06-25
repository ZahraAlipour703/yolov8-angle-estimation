# ultralytics/nn/modules/fpn.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, conv=Conv, epsilon=1e-4):
        """
        in_channels_list: list of channel dims for each input feature map [P3, P4, P5]
        out_channels: desired output channels after fusion
        """
        super().__init__()
        # learnable weights for each edge
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(3))
        # ensure numerical stability
        self.epsilon = epsilon
        # pointwise conv after fusion
        self.p3_conv = conv(sum(in_channels_list), out_channels, 1, 1)
    def forward(self, feats):
        # feats: list of [P3, P4, P5] feature maps
        P3, P4, P5 = feats
        # normalize weights
        w1 = torch.relu(self.w1)
        w1 = w1 / (w1.sum() + self.epsilon)
        # top-down fusion P5→P4→P3
        P5_up = nn.functional.interpolate(P5, scale_factor=2, mode='nearest')
        P4_td = w1[0]*P4 + w1[1]*P5_up
        # similarly bottom-up can be added...
        # for brevity, just demonstrate a single fusion
        fused = torch.cat([P3, 
                           nn.functional.interpolate(P4_td, scale_factor=2), 
                           nn.functional.interpolate(P5, scale_factor=4)], dim=1)
        return self.p3_conv(fused)
