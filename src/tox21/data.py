"""Tox21 dataset loading, featurisation and scaffold splitting."""

import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.rdBase import BlockLogs
from torch_geometric.data import Data

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

ELEMENTS = ["C", "O", "N", "Cl", "S", "F", "Br", "P", "I"]
DEGREES = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-1, 0, 1]
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def download_tox21() -> Path:
    """Download the Tox21 CSV into data/."""
    path = DATA_DIR / "tox21.csv.gz"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_tox21() -> pl.DataFrame:
    """Load Tox21 as a DataFrame, dropping SMILES that RDKit cannot parse."""
    df = pl.read_csv(download_tox21())
    # silence RDKit per-molecule warnings
    with BlockLogs():
        valid = [Chem.MolFromSmiles(s) is not None for s in df["smiles"]]
    return df.filter(pl.Series(valid))


def morgan_fingerprints(
    smiles: list[str] | pl.Series, radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Compute Morgan bit fingerprints."""
    fingerprints = []
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    with BlockLogs():
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fingerprints.append(gen.GetFingerprintAsNumPy(mol))
    return np.stack(fingerprints)


def one_hot(value, choices: list) -> list[int]:
    """One-hot encode values over choices."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom: Chem.Atom) -> list[int]:
    """Return a 25-long feature vector for an RDKit atom."""
    return (
        one_hot(atom.GetSymbol(), ELEMENTS)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot(atom.GetHybridization(), HYBRIDISATIONS)
        + [int(atom.GetIsAromatic())]
    )


def smiles_to_graph(smiles: str) -> Data:
    """Convert a SMILES string to a PyG graph."""
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32
    )
    pairs = []
    for bond in mol.GetBonds():
        pairs.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        pairs.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
    if pairs:
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def murcko_scaffolds(smiles: list[str] | pl.Series) -> list[str]:
    """Return the Bemis-Murcko scaffold SMILES of each molecule."""
    with BlockLogs():
        return [
            MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
            for s in smiles
        ]


def scaffold_split(
    smiles: list[str] | pl.Series, frac_train: float = 0.8, frac_val: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split molecules into train/val/test so no scaffold spans two splits."""
    groups = (
        pl.DataFrame({"scaffold": murcko_scaffolds(smiles)})
        .with_row_index("idx")
        .group_by("scaffold")
        .agg(pl.col("idx"))
        .with_columns(size=pl.col("idx").list.len())
        .sort(["size", "scaffold"], descending=[True, False])
        .get_column("idx")
        .to_list()
    )

    n = len(smiles)
    train_size = frac_train * n
    val_size = frac_val * n

    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= train_size:
            train.extend(group)
        elif len(val) + len(group) <= val_size:
            val.extend(group)
        else:
            test.extend(group)

    return np.array(train), np.array(val), np.array(test)
