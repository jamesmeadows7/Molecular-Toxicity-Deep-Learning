import urllib.request
from pathlib import Path

import polars as pl

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
    """Load Tox21 as a DataFrame."""
    return pl.read_csv(download_tox21())
