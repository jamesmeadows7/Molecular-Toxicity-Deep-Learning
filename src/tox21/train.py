"""Train the multi-task MLP with masked BCE and early stopping on validation ROC-AUC."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tox21.data import TASKS, load_tox21, morgan_fingerprints, scaffold_split
from tox21.evaluate import per_task_auc
from tox21.losses import masked_bce_with_logits
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


def main() -> None:
    """Train MLP on the scaffold split and print loss and val AUC per epoch."""
    device = get_device()

    df = load_tox21()
    X = torch.from_numpy(morgan_fingerprints(df["smiles"]).astype(np.float32))
    Y = torch.from_numpy(df.select(TASKS).to_numpy().astype(np.float32))
    train, val, test = scaffold_split(df["smiles"])
    Y_val = Y[val].numpy()
    Y_test = Y[test].numpy()

    set_seed(0)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(X[train], Y[train]), batch_size=BATCH_SIZE, shuffle=True
    )
    pos_weight = None

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_since_best = 0

    for epoch in range(MAX_EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, pos_weight, device)
        val_auc = np.mean(per_task_auc(Y_val, predict(model, X[val], device)))
        print(f"epoch {epoch:2d}  loss {loss:.4f}  val AUC {val_auc:.3f}")

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
    val_auc = np.mean(per_task_auc(Y_val, predict(model, X[val], device)))
    test_auc = np.mean(per_task_auc(Y_test, predict(model, X[test], device)))
    print(f"best epoch {best_epoch} val AUC {val_auc:.3f} test AUC {test_auc:.3f}")


if __name__ == "__main__":
    main()
