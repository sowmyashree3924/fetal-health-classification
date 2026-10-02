"""Reusable evaluation helpers for the fetal-health classification project."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def summarize(y_true, y_pred) -> dict:
    """Overall metrics: weighted Precision/Recall/F1 plus Macro F1."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """Precision / recall / F1 / support for every class (one row per class)."""
    labels = sorted(pd.unique(np.asarray(y_true)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"Class": labels, "Precision": p, "Recall": r, "F1-score": f, "Support": s}
    )


def bootstrap_macro_f1_ci(y_true, y_pred, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap confidence interval for Macro F1 on a fixed test set."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """Draw (and optionally save) a confusion matrix; class ids are used as labels."""
    labels = sorted(pd.unique(np.asarray(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
