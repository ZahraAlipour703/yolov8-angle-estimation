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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = self.avg_pool(x)  # b, c, 1, 1
        max_pool = self.max_pool(x)  # b, c, 1, 1
        channel_att = self.conv1(avg_pool) + self.conv1(max_pool)
        channel_att = self.relu(channel_att)
        channel_att = self.conv2(channel_att)
        channel_att = self.sigmoid_channel(channel_att)
        x = x * channel_att
        # Spatial attention
        avg_pool = x.mean(1, keepdim=True)
        max_pool, _ = x.max(1, keepdim=True)
        spatial_att = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.conv_spatial(spatial_att)
        spatial_att = self.sigmoid_spatial(spatial_att)
        return x * spatial_att
