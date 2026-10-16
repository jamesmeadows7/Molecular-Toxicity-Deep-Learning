"""Random forest baseline: one model per task on Morgan fingerprints."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from tox21.data import TASKS, load_tox21, morgan_fingerprints, scaffold_split
from tox21.evaluate import per_task_auc

df = load_tox21()

X = morgan_fingerprints(df["smiles"])
Y = df.select(TASKS).to_numpy()

train, val, test = scaffold_split(df["smiles"])

rows = []

for class_weight in [None, "balanced"]:
    for seed in range(5):
        val_scores = np.zeros((len(val), len(TASKS)))
        test_scores = np.zeros((len(test), len(TASKS)))

        for j, task in enumerate(TASKS):
            labelled_train = train[~np.isnan(Y[train, j])]
            clf = RandomForestClassifier(
                n_estimators=500,
                n_jobs=-1,
                class_weight=class_weight,
                random_state=seed,
            ).fit(X[labelled_train], Y[labelled_train, j])

            val_scores[:, j] = clf.predict_proba(X[val])[:, 1]
            test_scores[:, j] = clf.predict_proba(X[test])[:, 1]

        val_aucs = per_task_auc(Y[val], val_scores)
        test_aucs = per_task_auc(Y[test], test_scores)

        print(
            f"class_weight={class_weight} seed={seed}: val {np.mean(val_aucs):.3f}, test {np.mean(test_aucs):.3f}"
        )

        for split, aucs in [("val", val_aucs), ("test", test_aucs)]:
            for task, auc in zip(TASKS, aucs):
                rows.append(
                    {
                        "model": "rf",
                        "class_weight": str(class_weight),
                        "seed": seed,
                        "split": split,
                        "task": task,
                        "auc": auc,
                    }
                )

Path("results").mkdir(exist_ok=True)
results = pl.DataFrame(rows)
results.write_csv("results/metrics_rf.csv")

per_seed = results.group_by("class_weight", "seed", "split").agg(
    mean_auc=pl.col("auc").mean()
)
summary = (
    per_seed.group_by("class_weight", "split")
    .agg(
        mean=pl.col("mean_auc").mean().round(3),
        std=pl.col("mean_auc").std().round(4),
    )
    .sort("split", "class_weight")
)
print(summary)
