"""Train a model on Tox21 with masked BCE and early stopping. Run with --config <yaml>."""

import argparse
import copy
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch

from tox21.data import TASKS, load_tox21, morgan_fingerprints, scaffold_split
from tox21.evaluate import per_task_auc
from tox21.losses import compute_pos_weight, masked_bce_with_logits
from tox21.models import MLP
from tox21.utils import get_device, set_seed


def forward_batch(
    model: nn.Module, batch: Batch | list[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a PyG Batch or [x, y] list to device."""
    if isinstance(batch, Batch):  # PyG batch
        batch = batch.to(device)  # ty: ignore[unresolved-attribute]
        return model(batch.x, batch.edge_index, batch.batch), batch.y
    xb, yb = batch  # MLP batch
    return model(xb.to(device)), yb.to(device)


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
    for batch in loader:
        logits, y = forward_batch(model, batch, device)
        loss = masked_bce_with_logits(logits, y, pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return predicted probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = forward_batch(model, batch, device)
            probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


def train_and_evaluate(
    X: torch.Tensor,
    Y: torch.Tensor,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
    pos_weight: torch.Tensor | None,
    model_params: dict,
    training: dict,
    device: torch.device,
) -> tuple[list[float], list[float], int]:
    """Train one MLP with early stopping, returning per-task val AUCs, test AUCs and the best epoch."""
    set_seed(seed)
    model = MLP(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    train_loader = DataLoader(
        TensorDataset(X[train], Y[train]),
        batch_size=training["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[val], Y[val]),
        batch_size=training["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X[test], Y[test]),
        batch_size=training["batch_size"],
        shuffle=False,
    )
    Y_val = Y[val].numpy()
    Y_test = Y[test].numpy()

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_since_best = 0

    for epoch in range(training["max_epochs"]):
        train_one_epoch(model, train_loader, optimizer, pos_weight, device)
        val_auc = np.mean(per_task_auc(Y_val, predict(model, val_loader, device)))

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= training["patience"]:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_aucs = per_task_auc(Y_val, predict(model, val_loader, device))
    test_aucs = per_task_auc(Y_test, predict(model, test_loader, device))
    return val_aucs, test_aucs, best_epoch


def run(config: dict) -> None:
    """Train the configured model over seeds, with and without pos_weight."""
    device = get_device()

    model_params = config["model_params"]
    training = config["training"]

    df = load_tox21()
    X = torch.from_numpy(morgan_fingerprints(df["smiles"]).astype(np.float32))
    Y = torch.from_numpy(df.select(TASKS).to_numpy().astype(np.float32))
    train, val, test = scaffold_split(df["smiles"])

    rows = []
    for use_pos_weight in [False, True]:
        pos_weight = (
            compute_pos_weight(Y[train].numpy()).to(device) if use_pos_weight else None
        )
        for seed in config["seeds"]:
            val_aucs, test_aucs, best_epoch = train_and_evaluate(
                X, Y, train, val, test, seed, pos_weight, model_params, training, device
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

    Path(config["output"]).parent.mkdir(exist_ok=True)
    results = pl.DataFrame(rows)
    results.write_csv(config["output"])

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


def main() -> None:
    """Parse config and run the experiment."""
    parser = argparse.ArgumentParser(description="Train a model on Tox21.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config)


if __name__ == "__main__":
    main()
