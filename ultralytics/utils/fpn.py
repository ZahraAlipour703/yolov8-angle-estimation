# ultralytics/nn/modules/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, conv=Conv, epsilon=1e-4):
        super().__init__()
        # two sets of learnable edge weights
        self.w1 = nn.Parameter(torch.ones(2), requires_grad=True)  # top‑down
        self.w2 = nn.Parameter(torch.ones(3), requires_grad=True)  # bottom‑up
        self.epsilon = epsilon
        # pointwise conv after final fusion
        self.out_conv = conv(sum(in_channels_list), out_channels, 1, 1)

    def forward(self, feats):
        P3, P4, P5 = feats

        # --- Top‑down fusion pass ---
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum() + self.epsilon)
        P5_up = F.interpolate(P5, size=P4.shape[-2:], mode='nearest')
        P4_td = w1[0] * P4 + w1[1] * P5_up

        P4_up = F.interpolate(P4_td, size=P3.shape[-2:], mode='nearest')
        P3_td = w1[0] * P3 + w1[1] * P4_up  # reuse same w1 or define new

        # --- Bottom‑up fusion pass ---
        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum() + self.epsilon)
        P3_down = F.max_pool2d(P3_td, kernel_size=2, stride=2)
        P4_bu = w2[0] * P4_td + w2[1] * P3_down + w2[2] * P5

        P4_down = F.max_pool2d(P4_bu, kernel_size=2, stride=2)
        P5_bu = w2[0] * P5 + w2[1] * P4_down + w2[2] * 0  # or skip additional

        # --- Final fusion and conv ---
        fused = torch.cat([
            F.interpolate(P3_td, size=P3.shape[-2:], mode='nearest'),
            F.interpolate(P4_bu, size=P3.shape[-2:], mode='nearest'),
            F.interpolate(P5_bu, size=P3.shape[-2:], mode='nearest'),
        ], dim=1)
        return self.out_conv(fused)
