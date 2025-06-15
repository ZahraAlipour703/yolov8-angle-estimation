import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import GhostConv

class GhostC2f(nn.Module):
    """
    Lightweight C2f module using GhostConv. Falls back for small channel counts.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # If output channels too small, fallback to identity or standard conv
        if c2 < 4:
            # Use a single GhostConv to match channels
            self.add = False
            self.module = GhostConv(c1, c2, 1, 1)
        else:
            self.c = max(2, int(c2 * e))  # intermediate channels
            c_hidden = max(1, self.c // 2)
            # primary projections
            self.cv1 = GhostConv(c1, self.c, 1, 1)
            self.cv2 = GhostConv((2 if shortcut else 1) * self.c, c2, 1)
            # refinement path
            safe_g = 1 if c_hidden < g or c_hidden % g != 0 else g
            self.m = nn.Sequential(
                GhostConv(self.c, self.c, 3, 1, g=safe_g),
                GhostConv(self.c, self.c, 3, 1, g=safe_g)
            )
            self.add = shortcut and c1 == c2
            self.module = None

    def forward(self, x):
        if self.module is not None:
            # ghost-c2f path
            y1 = self.cv1(x)
            y2 = self.m(y1)
            out = self.cv2(torch.cat((y1, y2), 1))
            return out + x if self.add else out
        else:
            # fallback
            return self.module(x)
