"""Logistic regression baseline: one model per task on Morgan fingerprints."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from tox21.data import TASKS, load_tox21, morgan_fingerprints, scaffold_split
from tox21.evaluate import per_task_auc

df = load_tox21()

X = morgan_fingerprints(df["smiles"])
Y = df.select(TASKS).to_numpy()

train, val, test = scaffold_split(df["smiles"])

rows = []

for class_weight in [None, "balanced"]:
    val_scores = np.zeros((len(val), len(TASKS)))
    test_scores = np.zeros((len(test), len(TASKS)))

    for j, task in enumerate(TASKS):
        labelled_train = train[~np.isnan(Y[train, j])]
        clf = LogisticRegression(max_iter=2000, class_weight=class_weight).fit(
            X[labelled_train], Y[labelled_train, j]
        )

        val_scores[:, j] = clf.predict_proba(X[val])[:, 1]
        test_scores[:, j] = clf.predict_proba(X[test])[:, 1]

    val_aucs = per_task_auc(Y[val], val_scores)
    test_aucs = per_task_auc(Y[test], test_scores)

    print(
        f"class_weight={class_weight}: val {np.mean(val_aucs):.3f}, test {np.mean(test_aucs):.3f}"
    )

    for split, aucs in [("val", val_aucs), ("test", test_aucs)]:
        for task, auc in zip(TASKS, aucs):
            rows.append(
                {
                    "model": "lr",
                    "class_weight": str(class_weight),
                    "seed": 0,
                    "split": "val",
                    "task": task,
                    "auc": auc,
                }
            )

Path("results").mkdir(exist_ok=True)
pl.DataFrame(rows).write_csv("results/metrics_lr.csv")
