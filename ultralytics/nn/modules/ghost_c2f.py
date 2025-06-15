class GhostC2f(nn.Module):
    """
    GhostC2f: variant of C2f block using GhostConv layers to reduce computation.
    Splits channels, applies GhostConv and concatenates.
    """
    def __init__(self, channels: int, use_shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.use_shortcut = use_shortcut
        # First GhostConv reduces to hidden_channels
        self.cv1 = GhostConv(channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        # Second GhostConv expands back to channels
        self.cv2 = GhostConv(hidden_channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.cv2(y)
        if self.use_shortcut:
            return x + y
        return y
