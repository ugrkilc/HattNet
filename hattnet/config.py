"""
HattNet — Configuration
"""

import torch

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
DATA_DIR     = "Hattnet12/"
MASK_DIR     = "Hattnet12_mask/"
SKELETON_DIR = "Hattnet12_skeleton/"

# --------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------
SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_CLASSES  = 12
K_FOLDS      = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Two-stage training
STAGE1_EPOCHS      = 15
STAGE1_LR          = 1e-3
STAGE2_EPOCHS      = 40
STAGE2_LR_BACKBONE = 1e-5
STAGE2_LR_HEAD     = 1e-4
PATIENCE           = 7 
MIN_STAGE2_EPOCHS  = 10

# Architecture dims
TEXTURE_DIM = 768           # convnext_tiny features output
AUX_DIM        = 256        # HattCNN output channels
CBAM_REDUCTION = 16         # Reduction ratio for attention modules

# --------------------------------------------------------------------
# Class names (folder index -> display name)
# --------------------------------------------------------------------
CLASS_NAMES_MAP = {
    "1":  "Jali Diwânî",
    "2":  "Diwânî",
    "3":  "Ijâzî (Ruqah')",
    "4":  "Kufic",
    "5":  "Maqli (Square kufic)",
    "6":  "Muhaqqaq",
    "7":  "Naskh",
    "8":  "Rayhan",
    "9":  "Ruq'ah",
    "10": "Thuluth",
    "11": "Ta'liq",
    "12": "Tawqi",
}

# --------------------------------------------------------------------
# Available ablation modes
# --------------------------------------------------------------------
AVAILABLE_MODES = [
    # Coordinate Attention (proposed)
    'coord_attention',
    'coord_attention_cross_attention',
    'coord_attention_gating',
    'coord_attention_with_fusion',

    # # # CBAM
    'cbam',
    'cbam_cross_attention',
    'cbam_gating',
    'cbam_with_fusion',

    # # # SimAM
    'simam',
    'simam_cross_attention',
    'simam_gating',
    'simam_with_fusion',

    # # # ECA
    'eca',
    'eca_cross_attention',
    'eca_gating',
    'eca_with_fusion',

    # # # SE-Block
    'seblock',
    'seblock_cross_attention',
    'seblock_gating',
    'seblock_with_fusion',

    # No attention
    'concat_only',
    'concat_only_with_fusion',

    # Component ablation
    'no_skeleton',
    'no_mask',
]
