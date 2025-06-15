import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM) as described in "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks".
    Applies spatial attention by computing importance weights for each neuron based on variance.
    """
    def __init__(self, channels: int, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        # Compute mean per channel
        mu = x.mean(dim=[2, 3], keepdim=True)
        # Compute variance per element
        var = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)
        # Energy function: (x - mu)^2 / (4 * (var + lambda)) + 0.5
        energy = (x - mu) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        # Sigmoid to get attention map
        attention = torch.sigmoid(energy)
        return x * attention
