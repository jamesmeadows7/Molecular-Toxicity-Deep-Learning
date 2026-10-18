"""GIN hyperparameter grid (3 seeds, pos_weight), ranked on val AUC."""

import copy
from itertools import product

import polars as pl
import yaml

from tox21.train import run

with open("configs/gnn.yaml") as f:
    base = yaml.safe_load(f)

all_results = []
grid_settings = list(product([128, 256], [3, 5], [0.2, 0.5]))
for i, (hidden_dim, n_layers, dropout) in enumerate(grid_settings, start=1):
    print(
        f"[{i}/{len(grid_settings)}] hidden_dim={hidden_dim} n_layers={n_layers} dropout={dropout}"
    )

    config = copy.deepcopy(base)
    config["pos_weight"] = [True]
    config["seeds"] = [0, 1, 2]
    config["output"] = f"results/grid/gnn_h{hidden_dim}_l{n_layers}_d{dropout}.csv"

    model_params = config["model_params"]
    model_params["hidden_dim"] = hidden_dim
    model_params["n_layers"] = n_layers
    model_params["dropout"] = dropout

    results = run(config).with_columns(
        hidden_dim=pl.lit(hidden_dim),
        n_layers=pl.lit(n_layers),
        dropout=pl.lit(dropout),
    )
    all_results.append(results)

grid = pl.concat(all_results).filter(pl.col("split") == "val")
grid.write_csv("results/gnn_grid.csv")

per_seed = grid.group_by("hidden_dim", "n_layers", "dropout", "seed").agg(
    mean_auc=pl.col("auc").mean()
)
summary = (
    per_seed.group_by("hidden_dim", "n_layers", "dropout")
    .agg(mean=pl.col("mean_auc").mean().round(3), std=pl.col("mean_auc").std().round(4))
    .sort("mean", descending=True)
)
with pl.Config(tbl_rows=10):
    print(summary)
