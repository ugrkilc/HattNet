"""
HattNet — Attention Modules

Implements several attention mechanisms compared in the ablation study:
  - CBAM         (Woo et al., 2018 Cbam: Convolutional block attention module)
  - SE-Block     (Hu et al., 2018 Squeeze-and-excitation networks)
  - ECA          (Wang et al., 2020 ECA-Net: Efficient channel attention for deep convolutional neural networks)
  - SimAM        (Yang et al., 2021 Simam: A simple, parameter-free attention module for convolutional neural networks)
  - CoordAtt     (Hou et al., 2021 Coordinate attention for efficient mobile network design)  ← proposed
  - ChannelAttention1D : used for 'concat_with_attention' fusion
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------
# CBAM (Woo et al., 2018)
# --------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])
        max_pool = x.amax(dim=[2, 3])
        att = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        return x * att.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = torch.sigmoid(self.conv(concat))
        return x * att


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# --------------------------------------------------------------------
# SE-Block (Hu et al., 2018)
# --------------------------------------------------------------------
class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=[2, 3])
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# --------------------------------------------------------------------
# ECA (Wang et al., 2020)
# --------------------------------------------------------------------
class ECA(nn.Module):

    def __init__(self, in_channels, k_size=3):
        super().__init__()
        t = int(abs(math.log2(in_channels) / 2 + 0.5))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=[2, 3], keepdim=True).view(b, 1, c)
        y = torch.sigmoid(self.conv(y))
        return x * y.view(b, c, 1, 1)


# --------------------------------------------------------------------
# SimAM (Yang et al., 2021)
# --------------------------------------------------------------------
class SimAM(nn.Module):

    def __init__(self, in_channels=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * torch.sigmoid(y)


# --------------------------------------------------------------------
# Coordinate Attention (Hou et al., 2021) — PROPOSED
# --------------------------------------------------------------------
class CoordinateAttention(nn.Module):
    """
    Pools horizontal and vertical axes separately, producing two 1D
    direction-aware attention maps. Position-sensitive.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act   = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                          # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)              # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


# --------------------------------------------------------------------
# 1D Channel Attention (for fused feature vector)
# --------------------------------------------------------------------
class ChannelAttention1D(nn.Module):
    """Used in 'concat_with_attention' fusion to refine the concatenated vector."""

    def __init__(self, in_features, reduction=16):
        super().__init__()
        mid = max(in_features // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.mlp(x)


# --------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------
def get_attention_module(mode, in_channels, reduction=16):
    """
    Return the attention module for auxiliary streams based on mode.
    Returns None if no attention should be applied.
    """
    if mode in ['coord_attention', 'coord_attention_cross_attention',
                'coord_attention_gating', 'coord_attention_with_fusion','no,skeleton','no_mask','concat_only','concat_only_with_fusion']:
        return CoordinateAttention(in_channels, reduction=reduction)

    elif mode in ['cbam', 'cbam_cross_attention',
                  'cbam_gating', 'cbam_with_fusion']:
        return CBAM(in_channels, reduction=reduction)

    elif mode in ['simam', 'simam_cross_attention',
                  'simam_gating', 'simam_with_fusion']:
        return SimAM(in_channels)

    elif mode in ['eca', 'eca_cross_attention',
                  'eca_gating', 'eca_with_fusion']:
        return ECA(in_channels)

    elif mode in ['seblock', 'seblock_cross_attention',
                  'seblock_gating', 'seblock_with_fusion']:
        return SEBlock(in_channels, reduction=reduction)

    return None
