"""Tests for model definitions."""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tox21.data import build_graphs
from tox21.models import GIN, MLP


def test_mlp_output_shape():
    assert MLP()(torch.rand(4, 2048)).shape == (4, 12)


def test_gin_output_shape():
    graphs = build_graphs(
        ["CCO", "c1ccccc1", "[Na+]"], np.full((3, 12), np.nan, dtype=np.float32)
    )
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    assert GIN()(batch.x, batch.edge_index, batch.batch).shape == (3, 12)
