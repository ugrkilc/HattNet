# HattNet

**A Tri-Stream Deep Learning Architecture for Arabic Calligraphy Style Classification**

This repository contains the official implementation and dataset for HattNet, a tri-stream architecture that classifies 12 styles of Arabic calligraphy by jointly processing RGB texture, binary text mask, and stroke skeleton representations.

## Highlights

- **HattSet-12**: an expert-labeled, publicly released dataset of 12 Arabic calligraphy styles, compiled from Turkey Manuscripts Institution archives.
- **HattNet**: a tri-stream architecture combining a pretrained DenseNet169 backbone with two HattCNN auxiliary streams (mask + skeleton), enhanced by Coordinate Attention.
- **90.29% ± 1.62%** accuracy under 5-fold cross-validation (+6.56 over the best single-backbone baseline).

## Project Structure

```
HattNet/
├── hattnet/
│   ├── config.py        # Hyperparameters
│   ├── data.py          # Dataset + synchronized augmentation
│   ├── attention.py     # CBAM, CoordAtt, SimAM, ECA, SE-Block
│   ├── fusion.py        # Gating / cross-attention fusion
│   ├── backbone.py      # LightCNN
│   ├── model.py         # HattNet main model
│   ├── train.py         # Training loop + fold runner
│   └── utils.py         # Plots & seeding
├── scripts/
│   ├── run_single.py    # Run one ablation mode
│   └── run_all.py       # Run every mode sequentially
├── requirements.txt
└── README.md
```

## Dataset Setup

Organize the dataset in three parallel folders:

```
orj_dataset/
    1/  *.jpg     # Jali Diwânî
    2/  *.jpg     # Diwânî
    ...
    12/ *.jpg     # Tawqi

orj_dataset_mask/        # precomputed binary masks
orj_dataset_skeleton/    # precomputed stroke skeletons
```

Mask and skeleton folders mirror the RGB folder structure (same class names, same filenames). Use the preprocessing script to generate them from the RGB images.

## Usage

### Run a single mode

```bash
# Proposed model (default)
python scripts/run_single.py --mode coord_attention

# Attention ablation
python scripts/run_single.py --mode cbam
python scripts/run_single.py --mode simam
python scripts/run_single.py --mode eca
python scripts/run_single.py --mode seblock

# Component ablation
python scripts/run_single.py --mode no_mask
python scripts/run_single.py --mode no_skeleton
python scripts/run_single.py --mode concat_only

# Fusion ablation
python scripts/run_single.py --mode coord_attention_gating
python scripts/run_single.py --mode coord_attention_cross_attention
```

### Run all modes sequentially

```bash
python scripts/run_all.py
```

## Citation

If you use HattNet or HattSet-12, please cite:

```bibtex
@article{hattnet2026,
  title  = {HattNet: A Multi-Stream Deep Learning Architecture
            for Arabic Calligraphy Style Classification},
  author = {...},
  journal= {...},
  year   = {2026}
}
```

## License

MIT
