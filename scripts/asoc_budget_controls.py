"""ASOC major revision: step-matched vs epoch-matched augmentation controls.

Additive runner; original training code and the 12,294-run CSVs are untouched.
Run from repository root, after scripts.validate_backbone has generated data.

Four conditions per experiment:
  epoch_baseline   10 epochs on real windows (original update budget)
  step_baseline    real windows, same optimizer steps as 1:1 augmented data
  duplicate_real   duplicated real windows, same steps as 1:1 augmented data
  vae_augmented    real + VAE windows, same steps as the two matched controls

The VAE generator is target-only. In --input-mode all its auxiliary windows
are randomly paired with synthetic targets in the ORIGINAL implementation.
Do not infer aligned multivariate augmentation effects from that condition.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
KEY_COLS = ("file", "regime", "arch", "rep", "pool_seed", "input_mode",
            "epochs", "batch_size", "lr", "vae_epochs", "condition")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(*, allow_cpu: bool = False, gpu_index: int = 0) -> torch.device:
    """Require CUDA by default; explicit CPU fallback for environments without CUDA.

    The original gen_vae() chooses the active CUDA device when CUDA is
    available, so we select the device before generating the synthetic pool.
    """
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if not (0 <= gpu_index < count):
            raise ValueError(f"--gpu-index {gpu_index} is invalid: {count} CUDA device(s) available")
        torch.cuda.set_device(gpu_index)
        return torch.device(f"cuda:{gpu_index}")
    if allow_cpu:
        if gpu_index != 0:
            raise ValueError("--gpu-index requires CUDA")
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA GPU not available to this PyTorch environment. "
        "Install a CUDA-enabled PyTorch build/check GPU drivers; "
        "or explicitly pass --allow-cpu for a CPU-only smoke test."
    )


def only_channels(pair, mode: str):
    x, y = pair
    return (x[:, :, :1], y) if mode == "target" else (x, y)


def count_steps(n_real: int, n_aug: int, batch_size: int, epochs: int):
    if min(n_real, n_aug, batch_size, epochs) <= 0:
        raise ValueError("Counts, batch size and epochs must be positive")
    return (epochs * math.ceil(n_real / batch_size),
            epochs * math.ceil((n_real + n_aug) / batch_size))


def _loader(pair, batch_size: int, shuffle: bool):
    x, y = pair
    ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _evaluate(model: nn.Module, pair, batch_size: int, device: torch.device):
    model.eval()
    se, ae, count = 0.0, 0.0, 0
    for xb, yb in _loader(pair, batch_size, False):
        xb, yb = xb.to(device), yb.to(device)
        residual = model(xb) - yb
        se += residual.square().sum().item()
        ae += residual.abs().sum().item()
        count += yb.numel()
    return se / count, ae / count


def fit_condition(*, arch: str, train_pair, val_pair, test_pair,
                  init_seed: int, max_steps: int, validation_interval: int,
                  batch_size: int, lr: float, device: torch.device,
                  model_factory: Callable | None = None) -> dict:
    """Seed before model construction; select checkpoint on validation only.

    Original (shorter-budget) baseline uses its own epoch length for validation.
    Equal-step conditions validate every augmented epoch-equivalent, equally.
    """
    seed_everything(init_seed)
    if model_factory is None:
        from src.models import build
        model_factory = build
    model = model_factory(arch, train_pair[0].shape[1], train_pair[1].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _loader(train_pair, batch_size, True)
    iterator = iter(train_loader)
    best_val = float("inf")
    best_step = -1
    best_state = None
    curve = []
    samples_seen = 0
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        samples_seen += len(xb)
        xb, yb = xb.to(device), yb.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        if step % validation_interval == 0 or step == max_steps:
            vmse, _ = _evaluate(model, val_pair, batch_size, device)
            curve.append([step, vmse])
            if vmse < best_val:
                best_val = vmse
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    test_mse, test_mae = _evaluate(model, test_pair, batch_size, device)
    return {
        "test_mse": test_mse, "test_mae": test_mae,
        "val_mse": best_val, "best_step": best_step,
        "optimizer_steps": max_steps, "samples_seen": samples_seen,
        "train_windows": len(train_pair[0]), "validation_checkpoints": len(curve),
        "validation_curve": json.dumps(curve),
        "train_seconds": round(time.perf_counter() - t0, 3),
    }


def _read_completed(out: Path):
    if not out.exists():
        return set()
    df = pd.read_csv(out)
    if not set(KEY_COLS).issubset(df.columns):
        raise ValueError(f"Existing output lacks revision key columns: {out}")
    return set(map(tuple, df.loc[:, list(KEY_COLS)].itertuples(index=False, name=None)))


def _pool_seed_from_path(path: Path) -> int:
    m = re.search(r"_s(\d+)\.npy$", path.name)
    if not m:
        raise ValueError("Cannot infer pool seed from filename; pass --pool-seed")
    return int(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="om0.5_plain_s0.npy",
                   help="Synthetic .npy filename under data/synthetic, or absolute path")
    p.add_argument("--arch", choices=("dlinear", "patchtst", "itransformer"), default="dlinear")
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.10)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=None,
                   help="VAE pool seed, default series seed encoded in filename")
    p.add_argument("--input-mode", choices=("target", "all"), default="target")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu-index", type=int, default=0,
                   help="CUDA device index; uses GPU 0 by default")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow explicit CPU fallback when CUDA is unavailable (slow)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "asoc_revision" / "budget_controls.csv")
    args = p.parse_args()
    if args.epochs <= 0 or args.vae_epochs <= 0 or args.batch_size <= 0 or args.rep < 0 or args.lr <= 0:
        p.error("epochs, vae-epochs, batch-size, and lr must be positive; rep >= 0")
    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / "data" / "synthetic" / path
    if not path.is_file():
        p.error(f"Missing {path}; run python -m scripts.validate_backbone first")
    try:
        device = select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    print(f"DEVICE: {device} | GPU: {gpu_name} | "
          f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda or 'not built'}")
    if device.type != "cuda":
        print("WARNING: CPU fallback requested; do not compare runtime with GPU runs.")
    pool_seed = _pool_seed_from_path(path) if args.pool_seed is None else args.pool_seed

    from src.data import WindowConfig, prepare, chrono_split, apply_regime, standardize
    from src.generators import gen_vae

    X = np.load(path).astype(np.float32)
    cfg = WindowConfig(lookback=96, horizon=24)
    prepared = prepare(X, regime=args.regime, cfg=cfg)
    real = prepared["train"]
    n_real = len(real[0])
    # Match the exact training slice and standardizer of the original runner.
    train_end, _ = chrono_split(len(X))
    train_start = apply_regime(train_end, args.regime, cfg.lookback)
    standardized, _, _ = standardize(X, slice(train_start, train_end))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    synthetic = gen_vae(standardized[train_start:train_end], cfg,
                        ratio=1.0, seed=pool_seed, epochs=args.vae_epochs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = round(time.perf_counter() - t0, 3)
    if len(synthetic[0]) != n_real:
        raise RuntimeError(f"Expected {n_real} VAE windows, got {len(synthetic[0])}")
    original_steps, matched_steps = count_steps(n_real, len(synthetic[0]),
                                                 args.batch_size, args.epochs)
    baseline_interval = math.ceil(n_real / args.batch_size)
    matched_interval = math.ceil((n_real + len(synthetic[0])) / args.batch_size)

    duplicate = (np.concatenate([real[0], real[0]]),
                 np.concatenate([real[1], real[1]]))
    augmented = (np.concatenate([real[0], synthetic[0]]),
                 np.concatenate([real[1], synthetic[1]]))
    conditions = {
        "epoch_baseline": (real, original_steps, baseline_interval),
        "step_baseline": (real, matched_steps, matched_interval),
        "duplicate_real": (duplicate, matched_steps, matched_interval),
        "vae_augmented": (augmented, matched_steps, matched_interval),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _read_completed(args.out)
    if args.input_mode == "all":
        print("WARNING: original VAE uses randomly paired auxiliary windows; H3 inference invalid here.")

    for name in CONDITIONS:
        key = (path.name, args.regime, args.arch, args.rep, pool_seed, args.input_mode,
               args.epochs, args.batch_size, args.lr, args.vae_epochs, name)
        if key in done:
            print(f"SKIP {name}: existing result")
            continue
        pair, steps, interval = conditions[name]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        res = fit_condition(
            arch=args.arch, train_pair=only_channels(pair, args.input_mode),
            val_pair=only_channels(prepared["val"], args.input_mode),
            test_pair=only_channels(prepared["test"], args.input_mode),
            init_seed=args.rep, max_steps=steps, validation_interval=interval,
            batch_size=args.batch_size, lr=args.lr, device=device)
        record = dict(zip(KEY_COLS, key))
        record.update(res)
        record.update({
            "device": str(device), "gpu_name": gpu_name,
            "pytorch_version": str(torch.__version__),
            "pytorch_cuda_version": torch.version.cuda or "none",
            "gpu_peak_memory_mb": (
                round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
                if device.type == "cuda" else 0.0),
            "original_steps": original_steps, "matched_steps": matched_steps,
            "generator_seconds": generation_seconds,
            "vae_aux_alignment": "random_unmatched" if args.input_mode == "all" else "not_applicable",
            "window_length": cfg.lookback, "forecast_horizon": cfg.horizon,
        })
        pd.DataFrame([record]).to_csv(args.out, mode="a", index=False,
                                      header=not args.out.exists())
        print(f"{name:17s} steps={steps:5d} val={res['val_mse']:.6f} test={res['test_mse']:.6f}")
    print(f"Wrote/verified four conditions in {args.out}")


if __name__ == "__main__":
    main()
