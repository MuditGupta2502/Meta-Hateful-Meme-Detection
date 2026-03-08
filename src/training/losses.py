"""
Loss functions for hateful meme classification.

WeightedFocalLoss
-----------------
Combines class-weight rebalancing with focal modulation:

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

where
    p_t   = model's estimated probability for the true class
    α_t   = class weight for the true class
    γ     = focusing parameter (γ=0 reduces to weighted CE)

This addresses two issues in the dataset:
  1. Class imbalance  (5 450 non-hateful vs 3 050 hateful)
  2. Easy negatives dominating the gradient early in training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Args:
        class_weights : 1-D tensor of shape (num_classes,) or None
        gamma         : focusing exponent  (default 2.0)
        reduction     : "mean" | "sum" | "none"
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(2),
        )
        self.gamma = gamma
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,   # (B, num_classes)
        labels: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        # Per-sample cross-entropy (no reduction yet)
        ce = F.cross_entropy(
            logits, labels, weight=self.class_weights, reduction="none"
        )

        # p_t: probability assigned to the correct class
        p_t = torch.exp(-ce)

        # Focal modulation
        focal = ((1.0 - p_t) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
