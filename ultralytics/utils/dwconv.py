import torch
import torch.nn as nn
from .autopad import autopad

class DWConv(nn.Module):
  def __init__(self, c1, c2, k = 3, s = 1, p = None, act = True):
    super().__init__()
    swlf.dw = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups = c1, bias = False)
    swlf.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias = False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
      return self.act(self.bn(self.pw(slef.dw(x))))
