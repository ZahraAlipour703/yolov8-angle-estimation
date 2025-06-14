# ultralytics/utils/se.py
import torch
import torch.nn as nn

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) — Hu et al., 2018"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)
