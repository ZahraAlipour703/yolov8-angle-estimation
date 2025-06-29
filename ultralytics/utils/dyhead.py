# ultralytics/utils/dydetect.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect

class DyDetect(nn.Module):
    """
    Dynamic detection head: wraps a standard YOLOv8 Detect but allows
    optional per‐level weight modulation (if you extend to multi‐scale fusion).
    For now, simply forwards features to Detect.
    """
    def __init__(self, nc):
        """
        Args:
            nc (int): number of object classes
        """
        super().__init__()
        self.detect = Detect(nc)  # standard Detect head from ultralytics

    def forward(self, feats):
        """
        feats: list of feature maps [P3, P4, P5]
        returns: same output as Detect
        """
        return self.detect(feats)
