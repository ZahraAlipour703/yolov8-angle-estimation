import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad, Conv, C2f  # reuse existing helpers

class BiFPN(nn.Module):
    """Simplified single‐layer BiFPN"""
    def __init__(self, channels):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.conv3 = Conv(channels, channels, 1, 1)
        self.conv4 = Conv(channels, channels, 1, 1)
        self.conv5 = Conv(channels, channels, 1, 1)
        self.w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, feats):
        p3, p4, p5 = feats
        # top-down
        w = torch.relu(self.w1)
        w = w / (w.sum() + 1e-4)
        td4 = self.conv4(w[0]*p4 + w[1]*nn.functional.interpolate(p5, scale_factor=2))
        td3 = self.conv3(w[0]*p3 + w[1]*nn.functional.interpolate(td4, scale_factor=2))
        # bottom-up
        w2 = torch.relu(self.w2)
        w2 = w2 / (w2.sum() + 1e-4)
        bu4 = self.conv4(w2[0]*td4 + w2[1]*nn.functional.max_pool2d(td3, 2))
        bu5 = self.conv5(w2[0]*p5 + w2[1]*nn.functional.max_pool2d(bu4, 2))
        return bu4, bu5, td3
