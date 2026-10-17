"""Tests for masked BCE loss."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from tox21.losses import compute_pos_weight, masked_bce_with_logits


def test_nan_gives_finite_loss():
    logits = torch.tensor([[0.5, -1.0]])
    y = torch.tensor([[1.0, float("nan")]])
    assert torch.isfinite(masked_bce_with_logits(logits, y))


def test_masked_entries_ignored():
    y = torch.tensor([[1.0, float("nan")]])
    expected = F.binary_cross_entropy_with_logits(torch.tensor(0.5), torch.tensor(1.0))
    assert_close(masked_bce_with_logits(torch.tensor([[0.5, -1.0]]), y), expected)
    assert_close(masked_bce_with_logits(torch.tensor([[0.5, 100.0]]), y), expected)


def test_masked_entries_have_zero_gradient():
    logits = torch.tensor([[0.5, -1.0]], requires_grad=True)
    y = torch.tensor([[1.0, float("nan")]])
    masked_bce_with_logits(logits, y).backward()
    grad = logits.grad
    assert grad is not None
    assert grad[0, 1] == 0
    assert grad[0, 0] != 0


def test_standard_bce_without_nan():
    logits = torch.tensor([[0.5, -1.0], [2.0, 0.3]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert_close(
        masked_bce_with_logits(logits, y), F.binary_cross_entropy_with_logits(logits, y)
    )


def test_pos_weight_scales_positive_labels():
    logits = torch.tensor([[0.5]])
    y = torch.tensor([[1.0]])
    pos_weight = torch.tensor([3.0])
    assert_close(
        masked_bce_with_logits(logits, y, pos_weight),
        3 * masked_bce_with_logits(logits, y),
    )


def test_compute_pos_weight():
    y = np.array(
        [
            [1, 1],
            [0, 1],
            [0, 0],
            [0, np.nan],
        ]
    )
    assert_close(compute_pos_weight(y), torch.tensor([3.0, 0.5]))
