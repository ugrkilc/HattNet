"""
HattNet — Training & Evaluation

Two-stage training:
  Stage 1: freeze ConvNext_Tiny backbone, train heads (20 epochs).
  Stage 2: unfreeze all, fine-tune with differential learning rates,
           warmup + cosine schedule, early stopping (40 epochs max).

Public entry point: run_mode(mode) — runs a single ablation mode end-to-end.
"""

import os
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .data import TriStreamDataset, get_sync_augmentations
from .model import HattNet
from .utils import plot_fold_accuracies, plot_confusion, plot_all_fold_losses


# --------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for rgb, mask, skel, labels in tqdm(loader, desc="  Train", leave=False):
        rgb, mask, skel, labels = (rgb.to(device), mask.to(device),
                                   skel.to(device), labels.to(device))
        optimizer.zero_grad()
        logits = model(rgb, mask, skel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for rgb, mask, skel, labels in tqdm(loader, desc="  Eval ", leave=False):
        rgb, mask, skel, labels = (rgb.to(device), mask.to(device),
                                   skel.to(device), labels.to(device))
        logits = model(rgb, mask, skel)
        loss = criterion(logits, labels)

        running_loss += loss.item() * rgb.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# --------------------------------------------------------------------
# Single fold
# --------------------------------------------------------------------
def train_fold(fold, train_idx, test_idx, device, mode, output_dir):
    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold + 1}/{C.K_FOLDS}  [mode: {mode}]")
    print(f"{'=' * 60}")

    train_aug, test_aug = get_sync_augmentations(C.IMG_SIZE)
    use_skeleton = mode != 'no_skeleton'
    use_mask     = mode != 'no_mask'

    train_ds = TriStreamDataset(C.DATA_DIR, C.MASK_DIR, C.SKELETON_DIR,
                                sync_aug=train_aug, indices=train_idx,
                                use_skeleton=use_skeleton, use_mask=use_mask)
    test_ds  = TriStreamDataset(C.DATA_DIR, C.MASK_DIR, C.SKELETON_DIR,
                                sync_aug=test_aug, indices=test_idx,
                                use_skeleton=use_skeleton, use_mask=use_mask)

    targets = train_ds.targets
    class_counts = np.bincount(targets, minlength=C.NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    sampler = WeightedRandomSampler([weights[t] for t in targets], len(targets))

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

    model = HattNet(mode=mode, num_classes=C.NUM_CLASSES, aux_dim=C.AUX_DIM).to(device)

    if fold == 0:
        total_p = sum(p.numel() for p in model.parameters())
        bb_p    = sum(p.numel() for p in model.get_backbone_params())
        hd_p    = sum(p.numel() for p in model.get_head_params())
        print(f"\n  Total parameters : {total_p:,}")
        print(f"  Backbone         : {bb_p:,}")
        print(f"  Head+custom      : {hd_p:,}")
        print(f"  Mask             : {model.use_mask}")
        print(f"  Skeleton         : {model.use_skeleton}")
        print(f"  Fusion type      : {model.fusion_type}")

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    history = {
        "stage1_train_loss": [], "stage1_test_loss": [],
        "stage1_train_acc":  [], "stage1_test_acc":  [],
        "stage2_train_loss": [], "stage2_test_loss": [],
        "stage2_train_acc":  [], "stage2_test_acc":  [],
    }

    # ----- Stage 1: freeze backbone -----
    for p in model.get_backbone_params():
        p.requires_grad = False

    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.STAGE1_LR, weight_decay=1e-4,
    )
    trainable1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [Stage 1] Heads ({C.STAGE1_EPOCHS}ep, LR={C.STAGE1_LR})")
    print(f"  Trainable: {trainable1:,}")

    for epoch in range(1, C.STAGE1_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer1, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["stage1_train_loss"].append(tr_loss)
        history["stage1_test_loss"].append(te_loss)
        history["stage1_train_acc"].append(tr_acc)
        history["stage1_test_acc"].append(te_acc)

        mark = ""
        if te_acc > best_acc:
            best_acc = te_acc
            best_wts = copy.deepcopy(model.state_dict())
            mark = " *"
        print(f"    Ep {epoch:>2}/{C.STAGE1_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
              f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f}{mark}")
    print(f"  Stage 1 Best: {best_acc * 100:.2f}%")

    # ----- Stage 2: full fine-tune -----
    for p in model.parameters():
        p.requires_grad = True
    model.load_state_dict(best_wts)

    optimizer2 = optim.AdamW([
        {'params': model.get_backbone_params(), 'lr': C.STAGE2_LR_BACKBONE},
        {'params': model.get_head_params(),     'lr': C.STAGE2_LR_HEAD},
    ], weight_decay=1e-4)

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (C.STAGE2_EPOCHS - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lr_lambda)

    print(f"\n  [Stage 2] Fine-tune ({C.STAGE2_EPOCHS}ep)")

    patience_cnt = 0
    for epoch in range(1, C.STAGE2_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer2, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["stage2_train_loss"].append(tr_loss)
        history["stage2_test_loss"].append(te_loss)
        history["stage2_train_acc"].append(tr_acc)
        history["stage2_test_acc"].append(te_acc)

        scheduler.step()

        mark = ""
        if te_acc > best_acc:
            best_acc = te_acc
            best_wts = copy.deepcopy(model.state_dict())
            mark = " *"
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(f"    Ep {epoch:>2}/{C.STAGE2_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
              f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f}{mark}")

        if patience_cnt >= C.PATIENCE and epoch >= C.MIN_STAGE2_EPOCHS:
            print(f"    Early stopping at epoch {epoch}")
            break

    print(f"  Stage 2 Best: {best_acc * 100:.2f}%")

    model.load_state_dict(best_wts)
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)

    path = os.path.join(output_dir, f"fold{fold + 1}.pth")
    torch.save(best_wts, path)
    print(f"  Fold {fold + 1} Final: {best_acc * 100:.2f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_acc, preds, labels, history


# --------------------------------------------------------------------
# Full-mode runner
# --------------------------------------------------------------------
def run_mode(mode, output_root="results", summary_path="ablation_summary.txt"):
    output_dir = os.path.join(output_root, f"HattNet_{mode}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  HattNet — Mode: {mode}")
    print("=" * 60)
    print(f"  Device    : {C.DEVICE}")
    print(f"  Output    : {output_dir}/")
    print(f"  Stage 1   : {C.STAGE1_EPOCHS}ep, LR={C.STAGE1_LR}")
    print(f"  Stage 2   : {C.STAGE2_EPOCHS}ep, "
          f"bb={C.STAGE2_LR_BACKBONE}, hd={C.STAGE2_LR_HEAD}")

    dataset = datasets.ImageFolder(C.DATA_DIR)
    targets = np.array(dataset.targets)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [C.CLASS_NAMES_MAP.get(idx_to_class[i], idx_to_class[i])
                   for i in range(C.NUM_CLASSES)]
    print(f"  Dataset   : {len(dataset)} images, {C.NUM_CLASSES} classes")
    print(f"  Random seed: {C.SEED}")

    skf = StratifiedKFold(n_splits=C.K_FOLDS, shuffle=True, random_state=C.SEED)

    fold_accs, all_preds, all_labels, fold_histories = [], [], [], []
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        acc, preds, labels, history = train_fold(
            fold, train_idx, test_idx, C.DEVICE, mode, output_dir
        )
        fold_accs.append(acc)
        all_preds.extend(preds)
        all_labels.extend(labels)
        fold_histories.append(history)

    elapsed  = (time.time() - t0) / 60
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print(f"  HattNet [{mode}] — FINAL RESULTS")
    print("=" * 60)
    for i, a in enumerate(fold_accs):
        print(f"  Fold {i + 1}: {a * 100:.2f}%")
    print(f"\n  MEAN      : {mean_acc * 100:.2f}% +/- {std_acc * 100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Time      : {elapsed:.1f} min")

    print(f"\n{'=' * 60}\nCLASS-LEVEL REPORT:\n{'=' * 60}")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Write results to file
    with open(os.path.join(output_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"HattNet - Mode: {mode}\n")
        f.write("5-Fold Cross Validation Results\n")
        f.write("=" * 60 + "\n\n")
        for i, a in enumerate(fold_accs):
            f.write(f"Fold {i + 1}: {a * 100:.2f}%\n")
        f.write(f"\nMean: {mean_acc * 100:.2f}% +/- {std_acc * 100:.2f}%\n")
        f.write(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        f.write(f"Time: {elapsed:.1f} min\n\n")
        f.write(classification_report(all_labels, all_preds,
                                      target_names=class_names, digits=4))

    plot_fold_accuracies(fold_accs, mode, output_dir)
    plot_confusion(all_labels, all_preds, class_names, mode, output_dir)
    plot_all_fold_losses(fold_histories, mode, output_dir, C.K_FOLDS)

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"{mode:<32} | {mean_acc * 100:.2f}% +/- {std_acc * 100:.2f}% | "
            f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} | {elapsed:.1f}min\n"
        )

    print(f"\nOutputs: {output_dir}/")
    print(f"Summary: {summary_path}")
    return fold_accs
