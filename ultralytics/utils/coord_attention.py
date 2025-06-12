import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al., 2021):
    Encodes spatial information into channel attention by pooling along height and width axes separately.
    """
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # Changed to SiLU for YOLOv8 compatibility
        self.conv_h = nn.Conv2d(mip, inp, 1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, 1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # b,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # b,c,w,1 → b,c,1,w
        y = torch.cat([x_h, x_w], dim=2)  # b,c,h+w,1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_wimport torch
import torch.nn as nn

class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al., 2021):
    Encodes spatial information into channel attention by pooling along height and width axes separately.
    """
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # Changed to SiLU for YOLOv8 compatibility
        self.conv_h = nn.Conv2d(mip, inp, 1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, 1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # b,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # b,c,w,1 → b,c,1,w
        y = torch.cat([x_h, x_w], dim=2)  # b,c,h+w,1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w
