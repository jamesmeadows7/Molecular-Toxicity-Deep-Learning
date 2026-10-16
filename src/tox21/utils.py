"""Reproducibility and device helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for random, numpy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return Apple Silicon GPU if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
