"""
HattNet — Utilities

Random seed setup and plotting helpers (fold accuracy, confusion
matrix, training history).
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    """Set seed for reproducibility (note: cudnn is left non-deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_fold_accuracies(fold_accs, mode, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Fold {i + 1}" for i in range(len(fold_accs))],
        [a * 100 for a in fold_accs],
        color="#7F77DD",
        width=0.6,
    )
    mean = np.mean(fold_accs) * 100
    ax.axhline(y=mean, color="#E24B4A", linestyle="--", linewidth=2,
               label=f"Mean: {mean:.2f}%")
    for bar, acc in zip(bars, fold_accs):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{acc * 100:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"HattNet [{mode}] — 5-Fold CV", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fold_accuracies.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, labels, mode, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"HattNet [{mode}] — Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_fold_losses(fold_histories, mode, output_dir, k_folds):
    fig, axes = plt.subplots(2, k_folds, figsize=(5 * k_folds, 8), sharex=False)
    if k_folds == 1:
        axes = axes.reshape(2, 1)

    for i, hist in enumerate(fold_histories):
        # Top row: accuracy
        ax_acc = axes[0, i]
        train_accs = hist["stage1_train_acc"] + hist["stage2_train_acc"]
        test_accs  = hist["stage1_test_acc"]  + hist["stage2_test_acc"]
        epochs = list(range(1, len(train_accs) + 1))
        stage1_len = len(hist["stage1_train_acc"])

        ax_acc.plot(epochs, train_accs, label="Train", linewidth=1.5)
        ax_acc.plot(epochs, test_accs,  label="Test",  linewidth=1.5)
        ax_acc.axvline(stage1_len + 0.5, linestyle=":", linewidth=1, color="gray")
        ax_acc.set_title(f"Fold {i + 1} Accuracy")
        ax_acc.set_ylim([0, 1.0])
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="lower right")

        # Bottom row: loss
        ax_loss = axes[1, i]
        train_losses = hist["stage1_train_loss"] + hist["stage2_train_loss"]
        test_losses  = hist["stage1_test_loss"]  + hist["stage2_test_loss"]

        ax_loss.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
        ax_loss.plot(epochs, test_losses,  label="Val Loss",   linewidth=1.5)
        ax_loss.axvline(stage1_len + 0.5, linestyle=":", linewidth=1, color="gray")
        ax_loss.set_title(f"Fold {i + 1} Losses")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="upper right")

    plt.suptitle(f"HattNet [{mode}] — Training History", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, "training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
