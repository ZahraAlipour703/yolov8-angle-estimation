import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):  # Change 'inp' to 'channels'
        super().__init__()
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        
        # Horizontal pooling
        x_h = nn.AdaptiveAvgPool2d((h, 1))(x)
        # Vertical pooling
        x_w = nn.AdaptiveAvgPool2d((1, w))(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return identity * a_h * a_w
