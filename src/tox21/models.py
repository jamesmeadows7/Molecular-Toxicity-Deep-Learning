"""Model definitions: multi-task MLP."""

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-task MLP: shared hidden layers, one logit output per task."""

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dims: tuple[int, ...] = (512, 256),
        n_tasks: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_tasks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map fingerprints (batch, in_dim) to logits (batch, n_tasks)."""
        return self.net(x)
