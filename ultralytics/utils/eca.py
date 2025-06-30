import torch
import torch.nn as nn

class ECA(nn.Module):
    """Efficient Channel Attention (ECA) — Wang et al., 2020"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv across channels
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv(y)                         # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y
