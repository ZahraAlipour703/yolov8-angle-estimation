import torch
import torch.nn as nn

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.cheap = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, kernel_size=k, stride=s, padding=k // 2 if p is None else p, groups=g, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // 2, c2 // 2, kernel_size=1, stride=1, padding=0, groups=g, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((self.cheap(x), self.cheap(x)), dim=1)
