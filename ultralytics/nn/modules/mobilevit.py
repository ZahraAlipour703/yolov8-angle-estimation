import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad, Conv      # autopad & Conv are here
from ultralytics.nn.modules.block import C2f  

class MobileViTBlock(nn.Module):
    """Tiny MobileViT block: injects local→global self-attention"""
    def __init__(self, dim, ffn_dim, n_blocks=2):
        super().__init__()
        from ultralytics.nn.modules.transformer import TransformerLayer
        self.local = nn.Sequential(
            Conv(dim, dim, 3, 1),
            nn.BatchNorm2d(dim), nn.SiLU()
        )
        self.global_attn = nn.Sequential(*[
            TransformerLayer(dim, ffn_dim) for _ in range(n_blocks)
        ])
        self.project = Conv(dim, dim, 1, 1)

    def forward(self, x):
        y = self.local(x)
        # reshape to sequence: B,C,H*W → (H*W),B,C
        B, C, H, W = y.shape
        seq = y.flatten(2).permute(2,0,1)
        seq = self.global_attn(seq)
        seq = seq.permute(1,2,0).view(B, C, H, W)
        return self.project(seq + y)
