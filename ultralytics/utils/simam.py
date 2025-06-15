import torch
import torch.nn as nn

class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM): per-pixel spatial attention without extra parameters.
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        # compute mean and variance per-channel
        x_flat = x.view(b, c, -1)
        mu = x_flat.mean(-1, keepdim=True)
        var = ((x_flat - mu)**2).mean(-1, keepdim=True)
        # energy: (x - mu)^2 / (4*(var+lambda)) + 0.5
        e = ((x_flat - mu)**2) / (4 * (var + self.e_lambda)) + 0.5
        a = torch.sigmoid(e.view(b, c, h, w))
        return x * a
