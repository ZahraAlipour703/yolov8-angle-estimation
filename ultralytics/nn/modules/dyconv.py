import torch
import torch.nn as nn
import torch.nn.functional as F

class DyConv(nn.Module):
    def __init__(self, channels, kernel_size, K=4, reduction=4):
        super().__init__()
        self.K = K
        # Expert kernels
        self.experts = nn.Parameter(torch.randn(K, channels, channels, kernel_size, kernel_size))
        # Routing network
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, K, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # 1) global pooling → routing weights α ∈ ℝ^(B×K)
        α = self.fc(self.gap(x).view(b, c))
        α = F.softmax(α, dim=1)  # each α[b] sums to 1

        # 2) mix the expert kernels
        #    experts: [K, C, C, k, k]
        #    α:       [B, K] → [B, K, 1, 1, 1, 1]
        ex = self.experts.unsqueeze(0)  # [1,K,C,C,k,k]
        α  = α.view(b, self.K, 1,1,1,1)
        W  = (α * ex).sum(dim=1)         # [B, C, C, k, k]

        # 3) apply per‑sample convolution via grouped conv
        x = x.view(1, b*c, h, w)                                     # [1, B*C, H, W]
        W = W.view(b*c, c, W.size(-2), W.size(-1))                  # [B*C, C, k, k]
        y = F.conv2d(x, W, groups=b).view(b, c, h, w)                # back to [B,C,H,W]
        return y
