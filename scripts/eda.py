"""Exploratory data analysis."""

import polars as pl

from tox21.data import TASKS, load_tox21

df = load_tox21()
print(df.shape)

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
