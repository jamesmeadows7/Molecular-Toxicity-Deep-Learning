"""Tests for model definitions."""

import torch

from tox21.models import MLP


def test_mlp_output_shape():
    assert MLP()(torch.rand(4, 2048)).shape == (4, 12)
