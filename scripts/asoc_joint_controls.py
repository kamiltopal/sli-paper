"""ASOC Phase 3B: joint-window VAE versus original VAE and equal-step controls.

No original generators or 12,294 archived experiment rows are modified.
Each run is one paired cell with five iTransformer forecasters and a single
train-only random pool per generator. Run in a fresh subprocess via
`scripts.asoc_joint_matrix` for Windows/CUDA crash recovery.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.asoc_budget_controls import count_steps, fit_condition, select_device
from scripts.asoc_joint_vae import (gen_joint_vae, shuffle_synthetic_aux,
                                    cross_channel_diagnostics)
from src.data import WindowConfig, prepare, chrono_split, apply_regime, standardize
from src.generators import gen_vae

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/asoc_revision/joint_vae_controls.csv")
CONDITIONS = (
    "itr_all_step",
    "itr_all_duplicate",
    "itr_all_vae_original",
    "itr_all_vae_joint",
    "itr_all_vae_joint_shuffled",
)
EXPERIMENT_VERSION = "phase3b_joint_window_v1"
KEY_COLS = (
    "file", "regime", "rep", "pool_seed", "epochs", "vae_epochs",
    "batch_size", "lr", "latent", "hidden", "horizon_weight",
    "experiment_version", "condition",
)


def _completed_keys(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    missing = set(KEY_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Existing {path} has incompatible columns: {sorted(missing)}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError(f"Duplicate result rows in {path}; audit before continuing")
    return set(df.loc[:, list(KEY_COLS)].itertuples(index=False, name=None))


def _add(real, extra):
    x, y = real
    sx, sy = extra
    if x.shape[1:] != sx.shape[1:] or y.shape[1:] != sy.shape[1:] or len(x) != len(sx):
        raise ValueError("Expected 1:1 real and synthetic pools with matching shape")
    if not all(np.isfinite(z).all() for z in (x, y, sx, sy)):
        raise ValueError("Non-finite data in an augmentation pool")
    return np.concatenate((x, sx)), np.concatenate((y, sy))


def run_one(*, filename: str, regime: float, rep: int, pool_seed: int,
            epochs: int, vae_epochs: int, batch_size: int, lr: float,
            latent: int, hidden: int, horizon_weight: float,
            gpu_index: int, allow_cpu: bool, out: Path):
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / "data" / "synthetic" / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing series: {path}")
    if not (0 < regime <= 1) or min(epochs, vae_epochs, batch_size, latent, hidden) <= 0:
        raise ValueError("Invalid experiment settings")

    key_common = dict(file=path.name, regime=regime, rep=rep, pool_seed=pool_seed,
                      epochs=epochs, vae_epochs=vae_epochs,
                      batch_size=batch_size, lr=lr, latent=latent,
                      hidden=hidden, horizon_weight=horizon_weight,
                      experiment_version=EXPERIMENT_VERSION)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(out)
    pending = [condition for condition in CONDITIONS
               if tuple({**key_common, "condition": condition}[k] for k in KEY_COLS) not in done]
    if not pending:
        print(f"SKIP all five conditions already completed: {path.name} regime={regime} rep={rep}", flush=True)
        return

    device = select_device(allow_cpu=allow_cpu, gpu_index=gpu_index)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    print(f"DEVICE: {device} | GPU: {gpu_name} | torch: {torch.__version__}", flush=True)
    array = np.load(path).astype(np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite (T,C>=2) series")
    cfg = WindowConfig(lookback=96, horizon=24)
    data = prepare(array, regime, cfg)
    real = data["train"]
    t1, _ = chrono_split(len(array))
    tstart = apply_regime(t1, regime, cfg.lookback)
    standardized, _, _ = standardize(array, slice(tstart, t1))
    train_segment = standardized[tstart:t1]

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result, round(time.perf_counter() - tick, 3)

    # Original VAE is the frozen 30-epoch target-only generator, with randomly
    # paired real auxiliary windows. It intentionally remains UNMODIFIED.
    original, target_generator_seconds = timed(lambda: gen_vae(
        train_segment, cfg, ratio=1.0, seed=pool_seed, epochs=vae_epochs,
        latent=latent))
    # Joint VAE is fitted on exactly the same TRAIN windows (no val/test).
    (joint, joint_stats), joint_generator_seconds = timed(lambda: gen_joint_vae(
        real, seed=pool_seed, epochs=vae_epochs, device=device, latent=latent,
        hidden=hidden, horizon_weight=horizon_weight))
    joint_shuffled = shuffle_synthetic_aux(joint, seed=pool_seed)
    for name, pair in [("original", original), ("joint", joint), ("joint_shuffled", joint_shuffled)]:
        if len(pair[0]) != len(real[0]):
            raise RuntimeError(f"{name}: pool size must equal real train windows")
    # Crucial exact data-identity check for the alignment negative control.
    if not np.array_equal(joint[0][:, :, 0], joint_shuffled[0][:, :, 0]) or not np.array_equal(joint[1], joint_shuffled[1]):
        raise RuntimeError("Joint/shuffled must share exactly the synthetic targets and labels")

    original_steps, matched_steps = count_steps(len(real[0]), len(joint[0]), batch_size, epochs)
    validation_interval = math.ceil((2 * len(real[0])) / batch_size)
    conditions = {
        "itr_all_step": real,
        "itr_all_duplicate": _add(real, real),
        "itr_all_vae_original": _add(real, original),
        "itr_all_vae_joint": _add(real, joint),
        "itr_all_vae_joint_shuffled": _add(real, joint_shuffled),
    }
    diagnostics = {
        "itr_all_vae_original": cross_channel_diagnostics(real, original),
        "itr_all_vae_joint": cross_channel_diagnostics(real, joint),
        "itr_all_vae_joint_shuffled": cross_channel_diagnostics(real, joint_shuffled),
    }
    for condition in CONDITIONS:
        key = tuple({**key_common, "condition": condition}[k] for k in KEY_COLS)
        if key in done:
            print(f"SKIP {condition}: already recorded", flush=True)
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = fit_condition(
            arch="itransformer", train_pair=conditions[condition],
            val_pair=data["val"], test_pair=data["test"],
            init_seed=rep, max_steps=matched_steps,
            validation_interval=validation_interval,
            batch_size=batch_size, lr=lr, device=device,
        )
        row = {**key_common, "condition": condition, **metrics}
        row.update(dict(arch="itransformer", input_mode="all",
                        original_steps=original_steps, matched_steps=matched_steps,
                        target_generator_seconds=target_generator_seconds,
                        joint_generator_seconds=joint_generator_seconds,
                        joint_last_batch_loss=joint_stats["joint_last_batch_loss"],
                        joint_generator_parameters=joint_stats["joint_generator_parameters"],
                        generator_train_windows=joint_stats["joint_generator_training_windows"],
                        device=str(device), gpu_name=gpu_name,
                        pytorch_version=str(torch.__version__),
                        gpu_peak_memory_mb=(round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                                            if device.type == "cuda" else 0.0),
                        lag_corr_gap=np.nan, future_assoc_gap=np.nan,
                        zero_lag_corr_gap=np.nan,
                        window_length=cfg.lookback, forecast_horizon=cfg.horizon))
        row.update(diagnostics.get(condition, {}))
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{condition:28s} steps={matched_steps:5d} "
              f"val={metrics['val_mse']:.6f} test={metrics['test_mse']:.6f}", flush=True)
    print(f"Completed/verified {len(CONDITIONS)} conditions in {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="om0.5_plain_s0.npy")
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.10)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon-weight", type=float, default=0.5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if (args.rep < 0 or args.epochs < 1 or args.vae_epochs < 1 or args.batch_size < 1
            or args.lr <= 0 or not 0 < args.horizon_weight < 1):
        p.error("Invalid training parameters")
    inferred = re.search(r"_s(\d+)\.npy$", Path(args.file).name)
    pool_seed = args.pool_seed if args.pool_seed is not None else (int(inferred.group(1)) if inferred else 0)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    run_one(filename=args.file, regime=args.regime, rep=args.rep,
            pool_seed=pool_seed, epochs=args.epochs, vae_epochs=args.vae_epochs,
            batch_size=args.batch_size, lr=args.lr, latent=args.latent,
            hidden=args.hidden, horizon_weight=args.horizon_weight,
            gpu_index=args.gpu_index, allow_cpu=args.allow_cpu, out=out)


if __name__ == "__main__":
    main()
