import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = max(1, c2 // 2)  # ensure c_ is at least 1
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        # Ensure groups is valid and doesn't exceed c_
        valid_groups = max(1, min(c_, c_))
        self.cv2 = Conv(c_, c_, 5, 1, None, valid_groups, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
