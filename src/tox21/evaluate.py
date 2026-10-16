"""Model evaluation with per-task ROC-AUC."""

import numpy as np
from sklearn.metrics import roc_auc_score


def per_task_auc(y_true: np.ndarray, y_score: np.ndarray) -> list[float]:
    """Compute ROC-AUC for each task, ignoring missing values."""
    auc_scores = []
    _, n_tasks = y_true.shape
    for j in range(n_tasks):
        labelled = ~np.isnan(y_true[:, j])
        y = y_true[labelled, j]
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Task {j} needs both classes to compute AUC, got {np.unique(y)}"
            )
        auc_scores.append(float(roc_auc_score(y, y_score[labelled, j])))
    return auc_scores
