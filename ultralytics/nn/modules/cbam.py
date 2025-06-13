import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        return x * spatial_att
