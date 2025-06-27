import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import GhostConv

class GhostC2f(nn.Module):
    """
    GhostC2f: variant of C2f block using GhostConv layers to reduce computation.
    Splits channels, applies GhostConv and concatenates.
    """
    def __init__(self, channels: int, use_shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.use_shortcut = use_shortcut
        # first 1×1 GhostConv to reduce to hidden_channels
        self.cv1 = GhostConv(channels, hidden_channels, k=1, s=1, p=0)
        # second 1×1 GhostConv to expand back to channels
        self.cv2 = GhostConv(hidden_channels, channels, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.cv2(y)
        return x + y if self.use_shortcut else y
