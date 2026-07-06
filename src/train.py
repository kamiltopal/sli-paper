"""Fixed-budget training loop (protocol §3.5).

Budget fairness: every run gets the SAME optimizer, epochs, batch size,
and learning rate. Model selection = best validation MSE epoch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(pair, batch_size, shuffle):
    x, y = pair
    ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        se += ((pred - yb) ** 2).sum().item()
        ae += (pred - yb).abs().sum().item()
        n += yb.numel()
    return se / n, ae / n   # MSE, MAE


def train_model(model, data, cfg: TrainConfig):
    """data: dict from src.data.prepare (train/val/test pairs).
    Optionally data may contain 'train_aug' (inputs, targets) that is
    CONCATENATED to real training windows — the augmentation hook.

    Returns dict(test_mse, test_mae, val_mse, best_epoch).
    """
    seed_everything(cfg.seed)
    device = cfg.device
    model = model.to(device)

    xtr, ytr = data["train"]
    if "train_aug" in data and data["train_aug"] is not None:
        xa, ya = data["train_aug"]
        xtr = np.concatenate([xtr, xa], axis=0)
        ytr = np.concatenate([ytr, ya], axis=0)

    tr = _loader((xtr, ytr), cfg.batch_size, shuffle=True)
    va = _loader(data["val"], cfg.batch_size, shuffle=False)
    te = _loader(data["test"], cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.MSELoss()

    best_val, best_state, best_ep = float("inf"), None, -1
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        val_mse, _ = evaluate(model, va, device)
        if val_mse < best_val:
            best_val, best_ep = val_mse, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, te, device)
    return {"test_mse": test_mse, "test_mae": test_mae,
            "val_mse": best_val, "best_epoch": best_ep}
