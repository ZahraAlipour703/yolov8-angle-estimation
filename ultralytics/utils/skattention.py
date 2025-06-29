import torch
import torch.nn as nn

class SKAttention(nn.Module):
    """
    Selective Kernel Attention (Li et al. 2019)
    Fuses two parallel features by learning per-channel attention.
    Args:
        channels:   number of input/output channels
        M:          number of branches (here 2)
        G:          groups in fc
    """
    def __init__(self, channels, M=2, G=32, reduction=16):
        super().__init__()
        self.M = M
        self.channels = channels
        d = max(channels // reduction, G)
        # fc for attention
        self.fc   = nn.Linear(channels, d, bias=False)
        self.fcs  = nn.Linear(d, channels * M, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):
        # feats: list of M tensors, each [B,C,H,W]
        # stack: [B,M,C,H,W]
        U = torch.stack(feats, dim=1)
        # fuse by summation
        S = U.sum(dim=1)            # [B,C,H,W]
        # squeeze
        z = S.mean(-1).mean(-1)     # [B,C]
        z = self.fc(z)              # [B,d]
        a = self.fcs(z)             # [B, M*C]
        a = a.view(-1, self.M, self.channels)
        a = self.softmax(a)         # [B,M,C]
        # apply
        a = a.view(-1, self.M, self.channels, 1, 1)
        V = (U * a).sum(dim=1)      # [B,C,H,W]
        return V
