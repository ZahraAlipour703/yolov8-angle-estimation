import torch
import torch.nn as nn

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018):
    Sequential channel and spatial attention to refine features.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg_pool = x.mean((2,3), keepdim=False)
        max_pool, _ = x.max((2,3), keepdim=False)
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att = self.sigmoid_channel(channel_att).view(b, c, 1, 1)
        x = x * channel_att
        # Spatial attention
        avg_pool = x.mean(1, keepdim=True)
        max_pool, _ = x.max(1, keepdim=True)
        spatial_att = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.conv_spatial(spatial_att)
        spatial_att = self.sigmoid_spatial(spatial_att)
        return x * spatial_att
