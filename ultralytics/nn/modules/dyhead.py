import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Pose

class DyHead(nn.Module):
    """
    A “dynamic” head that runs both Detect and Pose on the same features.
    Args:
        nc: number of classes
        kpt_shape: keypoints shape [n_keypoints, 3]
    """
    def __init__(self, nc, kpt_shape):
        super().__init__()
        # instantiate the two heads
        self.detect = Detect(nc)                # from ultralytics.nn.modules.head
        self.pose   = Pose(nc, kpt_shape)       # from ultralytics.nn.modules.head

    def forward(self, features):
        """
        features: list of feature maps [P3, P4, P5]
        returns a dict with detection and pose outputs
        """
        # detection returns list of (batch, anchors, outputs)
        det_out = self.detect(features)
        # pose expects same list of features
        pose_out = self.pose(features)
        return {"det": det_out, "pose": pose_out}
