"""A small residual CNN for combined 3x upscaling + moire/artifact removal.

Unlike the original per-pixel MLP (which only ever sees a 3x3 neighborhood
and has no way to recognize a spatial interference pattern), this is a
proper spatial model: convolutions give it a growing receptive field, and
it predicts a correction on top of a bicubic upscale of the input rather
than the image from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class TinySRNet(nn.Module):
    def __init__(self, scale=3, channels=48, num_blocks=6):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x):
        feat = self.act(self.head(x))
        feat = self.body_tail(self.blocks(feat)) + feat
        correction = self.upsample(feat)
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(correction + skip, 0, 1)
