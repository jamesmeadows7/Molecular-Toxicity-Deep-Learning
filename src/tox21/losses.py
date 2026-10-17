"""Masked, class-weighted BCE loss for multi-task training."""

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean BCE over labelled entries only, nan labels are masked out."""
    mask = ~torch.isnan(y)
    targets = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Per-task positive weight n_neg / n_pos."""
    n_pos = np.nansum(y_train, axis=0)
    n_neg = (~np.isnan(y_train)).sum(axis=0) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)
