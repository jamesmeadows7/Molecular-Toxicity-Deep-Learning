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

# per task analysis
per_task = (
    final.filter(pl.col("split") == "test")
    .group_by("model", "task")
    .agg(auc=pl.col("auc").mean())
    .pivot(on="model", index="task", values="auc")
    .select("task", *MODEL_NAMES.keys())
    .with_columns(mean=pl.mean_horizontal(*MODEL_NAMES.keys()))
    .sort("mean", descending=True)
)

with pl.Config(tbl_rows=13):
    print(per_task.with_columns(pl.exclude("task").round(3)))


# GIN vs RF
gin_vs_rf = (
    final.filter(pl.col("split") == "test", pl.col("model").is_in(["gnn", "rf"]))
    .group_by("task", "model")
    .agg(mean=pl.col("auc").mean(), lo=pl.col("auc").min(), hi=pl.col("auc").max())
    .pivot(on="model", index="task", values=["mean", "lo", "hi"])
)
gin_vs_rf = gin_vs_rf.with_columns(
    diff=pl.col("mean_gnn") - pl.col("mean_rf"),
    consistent=(pl.col("lo_gnn") > pl.col("hi_rf"))
    | (pl.col("hi_gnn") < pl.col("lo_rf")),
).sort("diff", descending=True)

gin_vs_rf.write_csv("results/gin_vs_rf.csv")
with pl.Config(tbl_rows=13):
    print(
        gin_vs_rf.select(
            "task", "mean_rf", "mean_gnn", "diff", "consistent"
        ).with_columns(pl.col(pl.Float64).round(3))
    )


counts = gin_vs_rf.select(
    gin_higher=(pl.col("diff") > 0).sum(),
    gin_consistent=(pl.col("consistent") & (pl.col("diff") > 0)).sum(),
    rf_consistent=(pl.col("consistent") & (pl.col("diff") < 0)).sum(),
).row(0, named=True)

print(
    f"GIN higher on {counts['gin_higher']} of 12 tasks; consistent wins: "
    f"GIN {counts['gin_consistent']}, RF {counts['rf_consistent']}"
)
