"""Model definitions: multi-task MLP and GIN."""

import torch
from torch import nn
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.pool import global_add_pool


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


class GIN(nn.Module):
    """Multi-task GIN: message passing over atoms, sum pooling, one logit per task."""

    def __init__(
        self,
        in_dim: int = 25,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_tasks: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(prev, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))  # ty: ignore[call-non-callable]
            prev = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Map a batch of graphs to logits (n_molecules, n_tasks)."""
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        return self.head(x)
