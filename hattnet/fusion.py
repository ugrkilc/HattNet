"""
HattNet — Feature Fusion Strategies

Strategies compared in the ablation study (alongside simple concatenation):
  - GatingFusion           : learnable scalar gates per stream
  - CrossAttentionFusion   : multi-head cross-attention across stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingFusion(nn.Module):
    """
    Learns a scalar weight per stream via softmax, then concatenates
    the weighted streams.
    """

    def __init__(self, dims):
        super().__init__()
        self.num_streams = len(dims)
        total = sum(dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(total, total // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total // 4, self.num_streams),
        )

    def forward(self, features):
        concat = torch.cat(features, dim=1)
        gates = F.softmax(self.gate_mlp(concat), dim=1)
        weighted = [f * gates[:, i:i + 1] for i, f in enumerate(features)]
        return torch.cat(weighted, dim=1)


class CrossAttentionFusion(nn.Module):
    """
    Projects each stream to a common dim, then performs multi-head
    self-attention across the stream tokens.
    """

    def __init__(self, dims, attn_dim=256, num_heads=4):
        super().__init__()
        self.projs   = nn.ModuleList([nn.Linear(d, attn_dim) for d in dims])
        self.attn    = nn.MultiheadAttention(attn_dim, num_heads, dropout=0.1)
        self.norm    = nn.LayerNorm(attn_dim)
        self.out_dim = attn_dim * len(dims)

    def forward(self, features):
        projected = [proj(f).unsqueeze(0) for proj, f in zip(self.projs, features)]
        stacked = torch.cat(projected, dim=0)
        attended, _ = self.attn(stacked, stacked, stacked)
        attended = self.norm(stacked + attended)
        result = attended.permute(1, 0, 2).reshape(attended.size(1), -1)
        return result
