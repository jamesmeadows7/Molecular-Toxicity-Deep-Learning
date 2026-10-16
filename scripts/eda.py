"""Exploratory data analysis."""

import polars as pl
from rdkit import Chem

from tox21.data import TASKS, download_tox21

df = pl.read_csv(download_tox21())  # raw data
print(f"Loaded {df.height} molecules, {df.width} columns")

# --- Labels per Task ---
# labels are 0, 1, or null
# count, sum and mean skip nulls

summary = (
    df.select(TASKS)
    .unpivot(variable_name="task", value_name="label")
    .group_by("task")
    .agg(
        n_labelled=pl.col("label").count(),
        n_missing=pl.col("label").null_count(),
        pct_missing=(pl.col("label").is_null().mean() * 100).round(3),
        n_positive=pl.col("label").sum(),
        pos_rate=pl.col("label").mean().round(3),
    )
    .sort("pos_rate")
)

with pl.Config(tbl_rows=20, tbl_cols=10):
    print(summary)

# --- Molecule Parsing ---
# MolFromSmiles returns None when smiles is invalid

molecules = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]

df = df.with_columns(parsed=pl.Series([mol is not None for mol in molecules]))

failed = df.filter(~pl.col("parsed"))
print(f"{failed.height} SMILES failed to parse")
print(failed.select("mol_id", "smiles"))

# --- Duplicates and Molecule Size ---
# canonical smiles will reveal hidden duplicates

df = df.with_columns(
    canonical=pl.Series(
        [Chem.MolToSmiles(mol) if mol is not None else None for mol in molecules]
    ),
    n_atoms=pl.Series(
        [mol.GetNumAtoms() if mol is not None else None for mol in molecules]
    ),
)

n_parsed = df["canonical"].drop_nulls().len()
n_unique = df["canonical"].drop_nulls().n_unique()
print(f"{n_parsed} parsed, {n_unique} unique, {n_parsed - n_unique} duplicates")

print(df["n_atoms"].describe())

# a '.' separates disconnected fragments
n_fragmented = df["smiles"].str.contains(".", literal=True).sum()
print(f"{n_fragmented} SMILES contain multiple fragments")
