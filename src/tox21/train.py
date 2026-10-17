"""Train the multi-task MLP with masked BCE and early stopping on validation ROC-AUC."""

import copy
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tox21.data import TASKS, load_tox21, morgan_fingerprints, scaffold_split
from tox21.evaluate import per_task_auc
from tox21.losses import compute_pos_weight, masked_bce_with_logits
from tox21.models import MLP
from tox21.utils import get_device, set_seed

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor | None,
    device: torch.device,
) -> float:
    """Train one pass and return the mean batch loss."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = masked_bce_with_logits(model(xb), yb, pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return predicted probabilities."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X.to(device))).cpu().numpy()


def train_and_evaluate(
    X: torch.Tensor,
    Y: torch.Tensor,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
    pos_weight: torch.Tensor | None,
    device: torch.device,
) -> tuple[list[float], list[float], int]:
    """Train one MLP with early stopping, returning per-task val AUCs, test AUCs and the best epoch."""
    set_seed(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(X[train], Y[train]), batch_size=BATCH_SIZE, shuffle=True
    )
    Y_val = Y[val].numpy()
    Y_test = Y[test].numpy()

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_since_best = 0

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(model, loader, optimizer, pos_weight, device)
        val_auc = np.mean(per_task_auc(Y_val, predict(model, X[val], device)))

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_aucs = per_task_auc(Y_val, predict(model, X[val], device))
    test_aucs = per_task_auc(Y_test, predict(model, X[test], device))
    return val_aucs, test_aucs, best_epoch


def main() -> None:
    """Train the MLP over 5 seeds, with and without pos_weight."""
    device = get_device()

    df = load_tox21()
    X = torch.from_numpy(morgan_fingerprints(df["smiles"]).astype(np.float32))
    Y = torch.from_numpy(df.select(TASKS).to_numpy().astype(np.float32))
    train, val, test = scaffold_split(df["smiles"])

    rows = []
    for use_pos_weight in [False, True]:
        pos_weight = (
            compute_pos_weight(Y[train].numpy()).to(device) if use_pos_weight else None
        )
        for seed in range(5):
            val_aucs, test_aucs, best_epoch = train_and_evaluate(
                X, Y, train, val, test, seed, pos_weight, device
            )
            print(
                f"pos_weight={use_pos_weight} seed={seed}: best epoch {best_epoch}, val {np.mean(val_aucs):.3f}, test {np.mean(test_aucs):.3f}"
            )
            for split, aucs in [("val", val_aucs), ("test", test_aucs)]:
                for task, auc in zip(TASKS, aucs):
                    rows.append(
                        {
                            "model": "mlp",
                            "class_weight": "balanced" if use_pos_weight else "None",
                            "seed": seed,
                            "split": split,
                            "task": task,
                            "auc": auc,
                        }
                    )

    Path("results").mkdir(exist_ok=True)
    results = pl.DataFrame(rows)
    results.write_csv("results/metrics_mlp.csv")

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


if __name__ == "__main__":
    main()
