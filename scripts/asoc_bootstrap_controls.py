"""ASOC Phase 4: boundary-controlled block bootstrap vs residual-window bootstrap.

All new outputs are additive under results/asoc_revision; archived results are
never modified. Forecaster updates and validation checkpoints are identical.
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.asoc_budget_controls import fit_condition, only_channels, select_device
from scripts.asoc_bootstrap_generators import (bootstrap_pair_and_audit,
    residual_window_pool)
from src.data import WindowConfig, prepare, chrono_split, apply_regime, standardize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/asoc_revision/bootstrap_controls.csv")
CONDITIONS = ("step_real", "duplicate_real", "bootstrap_original",
              "bootstrap_safe", "bootstrap_residual")
VERSION = "phase4_window_bootstrap_v1"
KEY_COLS = ("file", "regime", "arch", "rep", "pool_seed", "epochs", "batch_size",
            "lr", "block", "trend_window", "periods", "version", "condition")


def _done(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if not set(KEY_COLS).issubset(df):
        raise ValueError(f"Incompatible output file: {path}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate experiment keys; audit CSV before resuming")
    return set(df[list(KEY_COLS)].itertuples(index=False, name=None))


def _concat(real, syn):
    if (len(syn[0]) != len(syn[1]) or real[0].shape[1:] != syn[0].shape[1:] or
            real[1].shape[1:] != syn[1].shape[1:] or
            not all(np.isfinite(a).all() for a in (*real, *syn))):
        raise ValueError("Invalid train pool")
    return np.concatenate([real[0], syn[0]]), np.concatenate([real[1], syn[1]])


def run_one(*, filename: str, regime: float, rep: int, pool_seed: int,
            arch: str, epochs: int, batch_size: int, lr: float, block: int,
            trend_window: int, periods: tuple[int, ...], gpu_index: int,
            allow_cpu: bool, out: Path):
    if arch not in ("dlinear", "patchtst"):
        raise ValueError("Phase 4 uses target-only forecasters dlinear/patchtst")
    if (regime not in (0.1, 0.25, 1.0) or rep < 0 or pool_seed < 0 or epochs < 1
            or batch_size < 1 or lr <= 0 or block < 120 or trend_window % 2 != 1
            or trend_window < 3 or not periods or any(p < 2 for p in periods)):
        raise ValueError("Invalid Phase 4 parameters")
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / "data" / "synthetic" / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing series: {path}")
    common = dict(file=path.name, regime=regime, arch=arch, rep=rep, pool_seed=pool_seed,
                  epochs=epochs, batch_size=batch_size, lr=lr, block=block,
                  trend_window=trend_window, periods=",".join(map(str, periods)), version=VERSION)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    missing = [c for c in CONDITIONS if tuple({**common, "condition": c}[k] for k in KEY_COLS) not in done]
    if not missing:
        print(f"SKIP: five conditions already complete for {path.name} / {regime} / {rep}", flush=True)
        return
    device = select_device(allow_cpu=allow_cpu, gpu_index=gpu_index)
    print(f"DEVICE: {device} | PyTorch: {torch.__version__}", flush=True)
    X = np.load(path).astype(np.float32)
    if X.ndim != 2 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("Expected finite multivariate synthetic stream")
    cfg = WindowConfig(lookback=96, horizon=24)
    data = prepare(X, regime, cfg)
    real = data["train"]
    t1, _ = chrono_split(len(X))
    start = apply_regime(t1, regime, cfg.lookback)
    std, _, _ = standardize(X, slice(start, t1))
    segment = std[start:t1]
    t0 = time.perf_counter()
    orig, safe, audit = bootstrap_pair_and_audit(segment, cfg, block=block, seed=pool_seed)
    bootstrap_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    residual, res_info = residual_window_pool(
        segment, cfg, n=len(orig[0]), seed=pool_seed,
        trend_window=trend_window, periods=periods)
    residual_seconds = time.perf_counter() - t0
    if len(safe[0]) != len(orig[0]) or len(residual[0]) != len(orig[0]):
        raise RuntimeError("Synthetic pools have unequal cardinalities")
    n_real, n_syn = len(real[0]), len(orig[0])
    matched_steps = epochs * math.ceil((n_real + n_syn) / batch_size)
    interval = math.ceil((n_real + n_syn) / batch_size)
    # Duplicating ALL real windows preserves the original real-data negative
    # control (n_real extra). The difference is one window at ratio=1 under
    # the frozen bootstrap implementation; all optimizer STEPS are identical.
    conditions = {
        "step_real": real,
        "duplicate_real": _concat(real, real),
        "bootstrap_original": _concat(real, orig),
        "bootstrap_safe": _concat(real, safe),
        "bootstrap_residual": _concat(real, residual),
    }
    audit_fields = audit.as_dict()
    for condition in CONDITIONS:
        key = tuple({**common, "condition": condition}[k] for k in KEY_COLS)
        if key in done:
            print(f"SKIP {condition}", flush=True)
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = fit_condition(
            arch=arch, train_pair=only_channels(conditions[condition], "target"),
            val_pair=only_channels(data["val"], "target"),
            test_pair=only_channels(data["test"], "target"),
            init_seed=rep, max_steps=matched_steps,
            validation_interval=interval, batch_size=batch_size, lr=lr, device=device)
        row = {**common, "condition": condition, **metrics,
               **audit_fields, **res_info,
               "matched_steps": matched_steps, "n_real": n_real, "n_synthetic": n_syn,
               "input_mode": "target", "lookback": cfg.lookback, "horizon": cfg.horizon,
               "device": str(device),
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "none",
               "pytorch_version": str(torch.__version__),
               "bootstrap_generator_seconds": round(bootstrap_seconds, 4),
               "residual_generator_seconds": round(residual_seconds, 4),
               "gpu_peak_memory_mb": (round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                                      if device.type == "cuda" else 0.0)}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{condition:22s} steps={matched_steps:5d} "
              f"val={metrics['val_mse']:.6f} test={metrics['test_mse']:.6f}", flush=True)
    print(f"Wrote/verified 5 conditions: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="om0.5_plain_s0.npy")
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.10)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=None)
    p.add_argument("--arch", choices=("dlinear", "patchtst"), default="dlinear")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--block", type=int, default=200)
    p.add_argument("--trend-window", type=int, default=169)
    p.add_argument("--periods", default="24,168")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    try:
        periods = tuple(int(x.strip()) for x in args.periods.split(","))
    except ValueError:
        p.error("--periods must be comma-separated integers")
    m = re.search(r"_s(\d+)\.npy$", Path(args.file).name)
    pool_seed = args.pool_seed if args.pool_seed is not None else (int(m.group(1)) if m else 0)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    run_one(filename=args.file, regime=args.regime, rep=args.rep,
            pool_seed=pool_seed, arch=args.arch, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, block=args.block,
            trend_window=args.trend_window, periods=periods,
            gpu_index=args.gpu_index, allow_cpu=args.allow_cpu, out=out)


if __name__ == "__main__":
    main()
