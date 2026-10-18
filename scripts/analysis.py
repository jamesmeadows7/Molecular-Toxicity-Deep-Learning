"""Combine all model results into summary tables and figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# --- Settings ---

MODEL_NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "mlp": "MLP",
    "gnn": "GIN",
}
GIN_COLOUR = "#2a78d6"
RF_COLOUR = "#eb6834"
CHANCE_COLOUR = "#898781"
BAND_COLOUR = "#f0efec"

# --- Helpers ---


def fmt(mean: float, std: float | None) -> str:
    """Format mean ± std to 3dp."""
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def plot_gin_vs_rf(table: pl.DataFrame, path: str) -> None:
    """Dot plot of RF vs GIN mean test AUC per task."""
    rows = table.sort("diff", descending=True).to_dicts()
    y = range(len(rows))

    fig, ax = plt.subplots()
    ax.axvline(0.5, color=CHANCE_COLOUR, linestyle="--", linewidth=1, zorder=1)
    for i in range(0, len(rows), 2):
        ax.axhspan(i - 0.5, i + 0.5, color=BAND_COLOUR, zorder=0)
    for offset, model, colour, label in [
        (-0.15, "rf", RF_COLOUR, "Random forest"),
        (0.15, "gnn", GIN_COLOUR, "GIN"),
    ]:
        ax.errorbar(
            [r[f"mean_{model}"] for r in rows],
            [i + offset for i in y],
            xerr=[r[f"std_{model}"] for r in rows],
            fmt="o",
            color=colour,
            markersize=7,
            elinewidth=2,
            capsize=0,
            label=label,
            zorder=3,
        )
    ax.set_yticks(list(y), [r["task"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Test ROC-AUC")
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 1. Load results and chose each model's best settings based on validation ---

results = pl.concat(
    [pl.read_csv(path) for path in sorted(Path("results").glob("metrics_*.csv"))]
)

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


# --- 2. Main results table ---

summary = (
    final.group_by("model", "seed", "split")
    .agg(mean_auc=pl.col("auc").mean())
    .group_by("model", "split")
    .agg(mean=pl.col("mean_auc").mean(), std=pl.col("mean_auc").std())
    .pivot(on="split", index="model", values=["mean", "std"])
    .select("model", "mean_val", "std_val", "mean_test", "std_test")
    .sort(pl.col("model").cast(pl.Enum(MODEL_NAMES.keys())))
)

print(summary)
summary.with_columns(pl.col(pl.Float64).round(4)).write_csv("results/summary.csv")

print("| Model | Val AUC | Test AUC |")
print("|---|---|---|")
for row in summary.iter_rows(named=True):
    val = fmt(row["mean_val"], row["std_val"])
    test = fmt(row["mean_test"], row["std_test"])
    print(f"| {MODEL_NAMES[row['model']]} | {val} | {test} |")


# 3. --- Per-task test AUC ---

per_task = (
    final.filter(pl.col("split") == "test")
    .group_by("model", "task")
    .agg(auc=pl.col("auc").mean())
    .pivot(on="model", index="task", values="auc")
    .select("task", *MODEL_NAMES.keys())
    .with_columns(mean=pl.mean_horizontal(*MODEL_NAMES.keys()))
    .sort("mean", descending=True)
)
per_task.with_columns(pl.col(pl.Float64).round(4)).write_csv(
    "results/per_task_test.csv"
)

with pl.Config(tbl_rows=13):
    print(per_task.with_columns(pl.exclude("task").round(3)))


# 4. --- GIN vs random forest per task ---

gin_vs_rf = (
    final.filter(pl.col("split") == "test", pl.col("model").is_in(["gnn", "rf"]))
    .group_by("task", "model")
    .agg(
        mean=pl.col("auc").mean(),
        lo=pl.col("auc").min(),
        hi=pl.col("auc").max(),
        std=pl.col("auc").std(),
    )
    .pivot(on="model", index="task", values=["mean", "lo", "hi", "std"])
    .select(
        "task",
        "mean_rf",
        "std_rf",
        "mean_gnn",
        "std_gnn",
        "lo_rf",
        "hi_rf",
        "lo_gnn",
        "hi_gnn",
    )
)
gin_vs_rf = gin_vs_rf.with_columns(
    diff=pl.col("mean_gnn") - pl.col("mean_rf"),
    consistent=(pl.col("lo_gnn") > pl.col("hi_rf"))
    | (pl.col("hi_gnn") < pl.col("lo_rf")),
).sort("diff", descending=True)

gin_vs_rf.with_columns(pl.col(pl.Float64).round(4)).write_csv("results/gin_vs_rf.csv")
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
    f"GIN higher on {counts['gin_higher']} of {gin_vs_rf.height} tasks; consistent wins: "
    f"GIN {counts['gin_consistent']}, RF {counts['rf_consistent']}"
)

# --- 5. Figures ---

Path("results/figures").mkdir(parents=True, exist_ok=True)
plot_gin_vs_rf(gin_vs_rf, "results/figures/gin_vs_rf_per_task.png")
