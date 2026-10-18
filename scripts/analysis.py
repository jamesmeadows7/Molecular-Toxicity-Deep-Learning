"""Combine all model results into summary tables."""

from pathlib import Path

import polars as pl

MODEL_NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "mlp": "MLP",
    "gnn": "GIN",
}


def fmt(mean: float, std: float | None) -> str:
    """Format mean ± std to 3dp."""
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


results = pl.concat(
    [pl.read_csv(path) for path in sorted(Path("results").glob("metrics_*.csv"))]
)

# find best settings on validation
val_means = (
    results.filter(pl.col("split") == "val")
    .group_by("model", "class_weight", "seed")
    .agg(mean_auc=pl.col("auc").mean())
    .group_by("model", "class_weight")
    .agg(val_auc=pl.col("mean_auc").mean())
)
chosen = (
    val_means.sort("val_auc", descending=True)
    .group_by("model")
    .first()
    .select("model", "class_weight")
    .sort("model")
)
print(chosen)

final = results.join(chosen, on=["model", "class_weight"], how="inner")
print(f"{final.height} rows")

summary = (
    final.group_by("model", "seed", "split")
    .agg(mean_auc=pl.col("auc").mean())
    .group_by("model", "split")
    .agg(mean=pl.col("mean_auc").mean(), std=pl.col("mean_auc").std())
    .pivot(on="split", index="model", values=["mean", "std"])
    .sort(pl.col("model").cast(pl.Enum(MODEL_NAMES.keys())))
)

print(summary)
summary.write_csv("results/summary.csv")

print("| Model | Val AUC | Test AUC |")
print("|---|---|---|")
for row in summary.iter_rows(named=True):
    val = fmt(row["mean_val"], row["std_val"])
    test = fmt(row["mean_test"], row["std_test"])
    print(f"| {MODEL_NAMES[row['model']]} | {val} | {test} |")
