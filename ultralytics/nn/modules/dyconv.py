import torch, torch.nn as nn

class DyConv(nn.Module):
    def __init__(self, channels, kernel_size, K=4, reduction=4):
        super().__init__()
        self.K = K
        # Expert kernels: K × (out_ch, in_ch, k, k)
        self.experts = nn.Parameter(torch.randn(K, channels, channels, kernel_size, kernel_size))
        # routing: global pool → FC → softmax over K
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, K, bias=False)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        # compute routing weights per‐batch
        pooled = self.gap(x).view(b, c)             # [B×C]
        alpha  = self.fc(pooled)                    # [B×K]
        alpha  = torch.softmax(alpha, dim=1)        # sum to 1
        # mix experts
        # reshape experts → [K, C, C, k, k], alpha → [B, K, 1,1,1,1]
        experts = self.experts.unsqueeze(0)         # [1,K,C,C,k,k]
        weights = (alpha.view(b, self.K, 1,1,1,1) * experts).sum(dim=1)
        # apply dynamic conv per‐sample
        # We can fold batch dimension by grouping, or apply in a loop (inefficient).
        # Here’s a simple grouped‐conv trick:
        x = x.view(1, b*c, h, w)
        w = weights.view(b*c, c, weights.size(-2), weights.size(-1))
        out = nn.functional.conv2d(x, w, groups=b).view(b, c, h, w)
        return out
