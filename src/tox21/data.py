import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.rdBase import BlockLogs

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
