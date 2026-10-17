"""Tests for data loading, featurisation and the scaffold split."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from tox21.data import (
    DEGREES,
    ELEMENTS,
    FORMAL_CHARGES,
    HYBRIDISATIONS,
    atom_features,
    build_graphs,
    load_tox21,
    murcko_scaffolds,
    one_hot,
    scaffold_split,
    smiles_to_graph,
)


@pytest.fixture(scope="module")
def smiles():
    return load_tox21()["smiles"]


@pytest.fixture(scope="module")
def scaffolds(smiles):
    return np.array(murcko_scaffolds(smiles))


@pytest.fixture(scope="module")
def split(smiles):
    return scaffold_split(smiles)


def test_no_scaffold_in_multiple_splits(split, scaffolds):
    train, val, test = (set(scaffolds[idx]) for idx in split)
    assert not train & val
    assert not train & test
    assert not val & test


def test_every_molecule_assigned_once(split, smiles):
    assert np.array_equal(np.sort(np.concatenate(split)), np.arange(len(smiles)))


def test_split_sizes(split, smiles):
    n = len(smiles)
    train, val, test = split
    assert abs(len(train) / n - 0.8) < 0.02
    assert abs(len(val) / n - 0.1) < 0.02
    assert abs(len(test) / n - 0.1) < 0.02


def test_split_is_deterministic(smiles, split):
    train1, val1, test1 = split
    train2, val2, test2 = scaffold_split(smiles)
    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)


def test_one_hot_choice():
    result = one_hot("N", ["C", "N", "O"])
    assert result == [0, 1, 0, 0]


def test_one_hot_other():
    result = one_hot("Hg", ["C", "N", "O"])
    assert result == [0, 0, 0, 1]


def test_atom_features_ethanol_oxygen():
    atom = Chem.MolFromSmiles("CCO").GetAtomWithIdx(2)
    encoding = atom_features(atom)
    expected = (
        one_hot("O", ELEMENTS)
        + one_hot(1, DEGREES)
        + one_hot(0, FORMAL_CHARGES)
        + one_hot(Chem.rdchem.HybridizationType.SP3, HYBRIDISATIONS)
        + [0]
    )
    assert len(encoding) == 25
    assert encoding == expected


def test_atom_features_benzene_carbon():
    atom = Chem.MolFromSmiles("c1ccccc1").GetAtomWithIdx(0)
    encoding = atom_features(atom)
    expected = (
        one_hot("C", ELEMENTS)
        + one_hot(2, DEGREES)
        + one_hot(0, FORMAL_CHARGES)
        + one_hot(Chem.rdchem.HybridizationType.SP2, HYBRIDISATIONS)
        + [1]
    )
    assert len(encoding) == 25
    assert encoding == expected


def test_smiles_to_graph_ethanol():
    graph = smiles_to_graph("CCO")
    x = graph.x
    assert x is not None
    assert x.shape == (3, 25)
    edge_index = graph.edge_index
    assert edge_index is not None
    assert edge_index.shape == (2, 4)
    edges = set(map(tuple, edge_index.t().tolist()))
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_smiles_to_graph_single_atom():
    graph = smiles_to_graph("[Na+]")
    x = graph.x
    assert x is not None
    assert x.shape == (1, 25)
    edge_index = graph.edge_index
    assert edge_index is not None
    assert edge_index.shape == (2, 0)


def test_smiles_to_graph_benzene():
    graph = smiles_to_graph("c1ccccc1")
    x = graph.x
    assert x is not None
    assert x.shape == (6, 25)
    edge_index = graph.edge_index
    assert edge_index is not None
    assert edge_index.shape == (2, 12)


def test_build_graphs_label_shapes():
    smiles = ["CCO", "c1ccccc1", "[Na+]"]
    Y = np.full((3, 12), np.nan, dtype=np.float32)
    Y[:, 0] = [1, 0, 1]
    graphs = build_graphs(smiles, Y)
    for graph in graphs:
        y = graph.y
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 12)
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    assert batch.y.shape == (3, 12)
    assert torch.isnan(batch.y[:, 1:]).all()
