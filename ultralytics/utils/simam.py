import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, x):
        b, c, h, w = x.size()
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        e = (x - mean) ** 2
        attn = e / (4 * (std + self.alpha) ** 2) + 0.5
        return x * torch.sigmoid(attn)
