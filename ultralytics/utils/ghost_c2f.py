import torch.nn as nn
from ultralytics.utils.ghost_conv import GhostConv


class GhostC2f(nn.Module):
    """
    GhostC2f: replaces C2f but uses GhostConv for both branches.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # total output channels = c2
        # hidden branch channels
        c_ = max(1, int(c2 * e))
        # first GhostConv branch
        self.ghost1 = GhostConv(c1, c_, 1, 1, g)
        # second branch: either identity or another conv
        if shortcut:
            self.add = True
            # adjust channels if necessary
            self.ghost2 = GhostConv(c1, c_, 1, 1, g)
        else:
            self.add = False
            self.ghost2 = GhostConv(c_, c_, 1, 1, g)
        # final conv to project back to c2
        self.conv = nn.Conv2d(c_ * 2, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Identity()

    def forward(self, x):
        y1 = self.ghost1(x)
        y2 = self.ghost2(x if self.add else y1)
        y = torch.cat([y1, y2], dim=1)
        return self.act(self.bn(self.conv(y)))
