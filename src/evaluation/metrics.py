"""
Evaluation metrics for the Hateful Memes Challenge.

Primary metric  : AUROC  (area under the ROC curve)
Secondary metrics: Accuracy, F1 (binary), F1 (macro), classification report
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    logits: np.ndarray,   # (N, 2)  raw model logits
    labels: np.ndarray,   # (N,)    integer ground-truth labels
) -> dict:
    """
    Convert logits → probabilities → predictions and compute all metrics.

    Samples with label == -1 (test set sentinels) are automatically excluded.

    Returns a dict with keys:
        auroc, accuracy, f1, f1_macro, loss (placeholder), report, confusion
    """
    # Convert logits → softmax probabilities
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    ).numpy()                        # (N, 2)
    preds = np.argmax(probs, axis=1) # (N,)
    hateful_prob = probs[:, 1]       # probability of class "hateful"

    # Exclude unlabelled test-set samples
    valid = labels != -1
    if valid.sum() == 0:
        return {}

    y_true = labels[valid]
    y_pred = preds[valid]
    y_score = hateful_prob[valid]

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=["non-hateful", "hateful"],
            zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auroc"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────────────

def pretty_print_metrics(metrics: dict, prefix: str = "") -> None:
    """Print a compact one-line summary plus the full classification report."""
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}"
        f"AUROC: {metrics.get('auroc', 0):.4f}  "
        f"Acc: {metrics.get('accuracy', 0):.4f}  "
        f"F1: {metrics.get('f1', 0):.4f}  "
        f"F1-macro: {metrics.get('f1_macro', 0):.4f}"
    )
    if "report" in metrics:
        print(metrics["report"])
