"""Tests for data loading and the scaffold split."""

import numpy as np
import pytest

from tox21.data import load_tox21, murcko_scaffolds, one_hot, scaffold_split


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
