"""
HattNet — Main Model

Tri-stream architecture for Arabic calligraphy style classification.

  - RGB texture stream  : ConvNext_Tiny backbone (ImageNet pretrained)
  - Mask stream         : HattCNN trained from scratch  + auxiliary attention
  - Skeleton stream     : HattCNN trained from scratch  + auxiliary attention

The texture stream uses no auxiliary attention so that pretrained
ImageNet representations are preserved. Features from all three streams
are fused via simple concatenation (by default) and passed to a classifier.

"""

import torch
import torch.nn as nn
from torchvision import models
import timm
from .config import TEXTURE_DIM, AUX_DIM, CBAM_REDUCTION
from .backbone import HattCNN
from .attention import (
    get_attention_module,
    ChannelAttention1D,
)
from .fusion import GatingFusion, CrossAttentionFusion


class HattNet(nn.Module):
    def __init__(self, mode='coord_attention', num_classes=12,
                 aux_dim=AUX_DIM, pretrained=True):
        super().__init__()
        self.mode = mode

        # Stream presence
        self.use_skeleton = mode != 'no_skeleton'
        self.use_mask     = mode != 'no_mask'

        # Fusion type
        self.fusion_type = self._determine_fusion_type(mode)

        # --- RGB texture stream ---
    
        self.texture_backbone = timm.create_model(
            'convnext_tiny', pretrained=pretrained, num_classes=0, global_pool=''
        )


        # No attention on the pretrained texture stream
        self.attention_texture = None

        # --- Mask stream ---
        if self.use_mask:
            self.mask_stream = HattCNN(in_channels=1, out_channels=aux_dim)
            self.attention_mask = get_attention_module(mode, aux_dim, CBAM_REDUCTION)
        else:
            self.attention_mask = None

        # --- Skeleton stream ---
        if self.use_skeleton:
            self.skeleton_stream = HattCNN(in_channels=1, out_channels=aux_dim)
            self.attention_skeleton = get_attention_module(mode, aux_dim, CBAM_REDUCTION)
        else:
            self.attention_skeleton = None

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Stream output dimensions
        dims = [TEXTURE_DIM]
        if self.use_mask:
            dims.append(aux_dim)
        if self.use_skeleton:
            dims.append(aux_dim)

        # Fusion module
        self.fusion, combined_dim = self._build_fusion(dims)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.Dropout(0.5),
            nn.Linear(combined_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    # ----------------------------------------------------------------
    # Init helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _determine_fusion_type(mode):
        gating_modes = ['coord_attention_gating', 'cbam_gating',
                        'simam_gating', 'eca_gating', 'seblock_gating']
        cross_modes  = ['coord_attention_cross_attention', 'cbam_cross_attention',
                        'simam_cross_attention', 'eca_cross_attention',
                        'seblock_cross_attention']
        fusion_modes = ['coord_attention_with_fusion', 'cbam_with_fusion',
                        'simam_with_fusion', 'eca_with_fusion',
                        'seblock_with_fusion', 'concat_only_with_fusion']

        if mode in gating_modes:
            return 'gating'
        if mode in cross_modes:
            return 'cross_attention'
        if mode in fusion_modes:
            return 'concat_with_attention'
        return 'concat'

    def _build_fusion(self, dims):
        if self.fusion_type == 'gating':
            fusion = GatingFusion(dims)
            combined_dim = sum(dims)
        elif self.fusion_type == 'cross_attention':
            fusion = CrossAttentionFusion(dims, attn_dim=256, num_heads=4)
            combined_dim = fusion.out_dim
        elif self.fusion_type == 'concat_with_attention':
            combined_dim = sum(dims)
            fusion = ChannelAttention1D(combined_dim, reduction=16)
        else:  # 'concat'
            fusion = None
            combined_dim = sum(dims)
        return fusion, combined_dim

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, rgb, mask, skeleton):
        # Texture stream (no attention)
        t = self.texture_backbone(rgb)               # (B, 768, 7, 7)

        t = self.gap(t).flatten(1)                   # (B, 768)
        features = [t]

        # Mask stream
        if self.use_mask:
            m = self.mask_stream(mask)
            if self.attention_mask is not None:
                m = self.attention_mask(m)
            m = self.gap(m).flatten(1)
            features.append(m)

        # Skeleton stream
        if self.use_skeleton:
            s = self.skeleton_stream(skeleton)
            if self.attention_skeleton is not None:
                s = self.attention_skeleton(s)
            s = self.gap(s).flatten(1)
            features.append(s)

        # Fusion
        if self.fusion_type in ['gating', 'cross_attention']:
            fused = self.fusion(features)
        else:
            fused = torch.cat(features, dim=1)
            if self.fusion_type == 'concat_with_attention':
                fused = self.fusion(fused)

        return self.classifier(fused)

    # ----------------------------------------------------------------
    # Param groups (for differential learning rates)
    # ----------------------------------------------------------------
    def get_backbone_params(self):
        return list(self.texture_backbone.parameters())

    def get_head_params(self):
        params = list(self.classifier.parameters())

        if self.use_mask:
            params += list(self.mask_stream.parameters())
            if self.attention_mask is not None:
                params += list(self.attention_mask.parameters())

        if self.use_skeleton:
            params += list(self.skeleton_stream.parameters())
            if self.attention_skeleton is not None:
                params += list(self.attention_skeleton.parameters())

        if self.fusion is not None:
            params += list(self.fusion.parameters())

        return params
