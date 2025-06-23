# Define the C2f_CA module
import torch
import torch.nn as nn
class C2f_CA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(C2f_CA, self).__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution
        self.ca = CoordinateAttention((2 + n) * self.c)  # Coordinate Attention
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final convolution
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )  # Bottleneck layers

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # Split into two parts
        for m in self.m:
            y.append(m(y[-1]))  # Apply bottlenecks sequentially
        cat_y = torch.cat(y, 1)  # Concatenate all outputs
        ca_y = self.ca(cat_y)  # Apply Coordinate Attention
        return self.cv2(ca_y)  # Final convolution
