import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import GhostConv

class GhostC2f(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = max(1, int(c2 * e))  # ✅ Make sure intermediate channels ≥ 1

        self.cv1 = GhostConv(c1, self.c, 1, 1)
        self.cv2 = GhostConv((2 if shortcut else 1) * self.c, c2, 1)

        self.m = nn.Sequential(
            GhostConv(self.c, self.c, 3, 1, g=g),
            GhostConv(self.c, self.c, 3, 1, g=g)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(y1)
        out = self.cv2(torch.cat((y1, y2), 1))
        return out + x if self.add else out
