"""
Evaluation utilities for skin lesion classification.

Computes balanced accuracy, weighted F1 score, per-class precision/recall,
and confusion matrix for the 8-class ISIC 2019 task.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.dataset import CLASS_NAMES


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Dictionary containing balanced_accuracy, weighted_f1,
        per_class_precision, and per_class_recall.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_prec = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(CLASS_NAMES))
    )
    per_class_rec = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(CLASS_NAMES))
    )

    return {
        "balanced_accuracy": bal_acc,
        "weighted_f1": w_f1,
        "per_class_precision": {
            name: float(p) for name, p in zip(CLASS_NAMES, per_class_prec)
        },
        "per_class_recall": {
            name: float(r) for name, r in zip(CLASS_NAMES, per_class_rec)
        },
    }


def print_classification_report(y_true: List[int], y_pred: List[int]) -> str:
    """Print a formatted sklearn classification report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        The classification report as a string.
    """
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    print(report)
    return report


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        save_path: If provided, save the figure to this path.

    Returns:
        The confusion matrix as a numpy array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - ISIC 2019 Classification")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)
    return cm
