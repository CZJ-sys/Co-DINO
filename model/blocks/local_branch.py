"""Co-DINO's fixed zero-padded local depthwise branch."""

from torch import nn


class LocalDepthwiseBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.padding_mode = "zero"
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels,
            bias=False, padding_mode="zeros")

    def forward(self, x):
        return self.conv(x)
