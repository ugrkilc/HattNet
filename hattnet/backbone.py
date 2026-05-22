"""
HattNet — HattCNN Backbone

A small from-scratch CNN used for the auxiliary streams (mask and skeleton).
Five conv blocks: 1 -> 32 -> 64 -> 128 -> 256 -> 256 channels.
Each block: 2x (Conv3x3 + BN + ReLU), then MaxPool 2x2.
With 224x224 input, output feature map is 256 x 7 x 7.
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BN/ReLU, followed by 2x2 max-pool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class HattCNN(nn.Module):
    """5-block CNN trained from scratch (Kaiming init)."""

    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)
