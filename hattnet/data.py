"""
HattNet — Dataset & Augmentation

Defines:
  - SynchronizedAugmentation: applies identical geometric transforms
    to RGB, mask, and skeleton (preserving spatial correspondence).
  - TriStreamDataset: reads RGB / mask / skeleton from parallel folders.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


class SynchronizedAugmentation:
    """
    Applies augmentation to RGB, mask and skeleton with the same random
    parameters so geometric consistency across streams is preserved.
    """

    def __init__(self, img_size, training=True):
        self.img_size = img_size
        self.training = training

        # ImageNet normalize for RGB
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std  = [0.229, 0.224, 0.225]
        # Aux normalize (grayscale)
        self.aux_mean = [0.5]
        self.aux_std  = [0.5]

    def letterbox(self, img, target_size, mode='RGB'):
        """Resize preserving aspect ratio; pad rest with zeros."""
        w, h = img.size
        ratio = target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resample = Image.BILINEAR if mode == 'RGB' else Image.NEAREST
        img = img.resize((new_w, new_h), resample)
        fill = (0, 0, 0) if mode == 'RGB' else 0
        canvas = Image.new(mode, (target_size, target_size), fill)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas

    def __call__(self, rgb_img, mask_img=None, skel_img=None):
        """
        Returns: (rgb_tensor, mask_tensor, skel_tensor).
        If mask_img or skel_img is None, returns zero tensor for that stream.
        """
        if self.training:
            # Letterbox to slightly larger size for random crop
            target = self.img_size + 32
            rgb_img = self.letterbox(rgb_img, target, mode='RGB')
            if mask_img is not None:
                mask_img = self.letterbox(mask_img, target, mode='L')
            if skel_img is not None:
                skel_img = self.letterbox(skel_img, target, mode='L')

            # === Same random parameters for all streams ===
            # 1) Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                rgb_img, output_size=(self.img_size, self.img_size)
            )
            rgb_img = TF.crop(rgb_img, i, j, h, w)
            if mask_img is not None:
                mask_img = TF.crop(mask_img, i, j, h, w)
            if skel_img is not None:
                skel_img = TF.crop(skel_img, i, j, h, w)

            # 2) Random rotation
            angle = random.uniform(-15, 15)
            rgb_img = TF.rotate(rgb_img, angle,
                                interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            if mask_img is not None:
                mask_img = TF.rotate(mask_img, angle,
                                     interpolation=TF.InterpolationMode.NEAREST, fill=0)
            if skel_img is not None:
                skel_img = TF.rotate(skel_img, angle,
                                     interpolation=TF.InterpolationMode.NEAREST, fill=0)

            # 3) Random translation + shear
            tx = random.uniform(-0.1, 0.1) * self.img_size
            ty = random.uniform(-0.1, 0.1) * self.img_size
            shear = random.uniform(-10, 10)
            rgb_img = TF.affine(rgb_img, angle=0, translate=(tx, ty),
                                scale=1.0, shear=shear,
                                interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            if mask_img is not None:
                mask_img = TF.affine(mask_img, angle=0, translate=(tx, ty),
                                     scale=1.0, shear=shear,
                                     interpolation=TF.InterpolationMode.NEAREST, fill=0)
            if skel_img is not None:
                skel_img = TF.affine(skel_img, angle=0, translate=(tx, ty),
                                     scale=1.0, shear=shear,
                                     interpolation=TF.InterpolationMode.NEAREST, fill=0)

            # 4) Color jitter — RGB only
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )
            rgb_img = color_jitter(rgb_img)

            # 5) Random grayscale — RGB only
            if random.random() < 0.1:
                rgb_img = TF.rgb_to_grayscale(rgb_img, num_output_channels=3)

        else:
            # Test: just letterbox to target size, no augmentation
            rgb_img = self.letterbox(rgb_img, self.img_size, mode='RGB')
            if mask_img is not None:
                mask_img = self.letterbox(mask_img, self.img_size, mode='L')
            if skel_img is not None:
                skel_img = self.letterbox(skel_img, self.img_size, mode='L')

        # Convert to tensors and normalize
        rgb_tensor = TF.to_tensor(rgb_img)
        rgb_tensor = TF.normalize(rgb_tensor, self.rgb_mean, self.rgb_std)

        if mask_img is not None:
            mask_tensor = TF.to_tensor(mask_img)
            mask_tensor = TF.normalize(mask_tensor, self.aux_mean, self.aux_std)
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)

        if skel_img is not None:
            skel_tensor = TF.to_tensor(skel_img)
            skel_tensor = TF.normalize(skel_tensor, self.aux_mean, self.aux_std)
        else:
            skel_tensor = torch.zeros(1, self.img_size, self.img_size)

        return rgb_tensor, mask_tensor, skel_tensor


class TriStreamDataset(Dataset):
    """
    Reads RGB images from DATA_DIR, masks from MASK_DIR, and skeletons
    from SKELETON_DIR. Applies SynchronizedAugmentation to keep geometric
    consistency across the three streams.
    """

    def __init__(self,
                 rgb_root, mask_root, skeleton_root,
                 sync_aug,
                 indices=None,
                 use_mask=True,
                 use_skeleton=True):
        self.base = datasets.ImageFolder(rgb_root)
        self.indices = indices if indices is not None else list(range(len(self.base)))

        self.rgb_root      = Path(rgb_root)
        self.mask_root     = Path(mask_root)
        self.skeleton_root = Path(skeleton_root)

        self.sync_aug     = sync_aug
        self.use_mask     = use_mask
        self.use_skeleton = use_skeleton

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.base.samples[real_idx]
        img_path = Path(img_path)

        rgb_img = Image.open(img_path).convert("RGB")
        rel_path = img_path.relative_to(self.rgb_root)

        mask_img = None
        skel_img = None

        if self.use_mask:
            mask_img = Image.open(self.mask_root / rel_path).convert("L")
        if self.use_skeleton:
            skel_img = Image.open(self.skeleton_root / rel_path).convert("L")

        rgb_tensor, mask_tensor, skel_tensor = self.sync_aug(rgb_img, mask_img, skel_img)
        return rgb_tensor, mask_tensor, skel_tensor, label

    @property
    def targets(self):
        return [self.base.targets[i] for i in self.indices]


def get_sync_augmentations(img_size):
    """Returns (train_aug, test_aug) — both SynchronizedAugmentation instances."""
    train_aug = SynchronizedAugmentation(img_size, training=True)
    test_aug  = SynchronizedAugmentation(img_size, training=False)
    return train_aug, test_aug
