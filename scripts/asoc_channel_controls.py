"""ASOC revision phase 3: matched-step channel/pairing ablations.

This experiment does not alter the original VAE. It uses its *same target pool*
and compares its original random auxiliary pairing with an exploratory
nearest-real-context pairing. The latter is NOT ground-truth temporal alignment.

Usage (from project root):
  python -m scripts.asoc_channel_controls --file om0.5_plain_s0.npy --regime 0.10 --rep 0
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
from scipy.spatial import cKDTree

from scripts.asoc_budget_controls import (
    count_steps, fit_condition, only_channels, select_device,
)
from src.data import WindowConfig, prepare, chrono_split, apply_regime, standardize
from src.generators import gen_vae

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/asoc_revision/channel_controls.csv")
CONDITIONS = (
    "itr_target_step",
    "itr_target_duplicate",
    "itr_target_vae",
    "itr_all_step",
    "itr_all_duplicate",
    "itr_all_duplicate_permuted",
    "itr_all_vae_random",
    "itr_all_vae_context",
)
KEY_COLS = (
    "file", "regime", "rep", "pool_seed", "epochs", "vae_epochs",
    "batch_size", "lr", "context_bins", "condition",
)


def context_features(x: np.ndarray, bins: int = 12) -> np.ndarray:
    """Extract target *lookback-only* features; no forecast labels or test data.

    Input shape (N,L,C), with target at channel zero. Target lookback is
    represented by coarse bin means and stds, global trend and last level.
    Features are normalized using REAL TRAIN features before neighbor search.
    """
    if x.ndim != 3 or x.shape[0] < 1:
        raise ValueError("Expected nonempty (windows, lookback, channels)")
    n, length, _ = x.shape
    if bins < 1 or length % bins != 0:
        raise ValueError("context_bins must be a positive divisor of lookback")
    target = np.asarray(x[:, :, 0], dtype=np.float64)
    blocks = target.reshape(n, bins, length // bins)
    vals = np.concatenate([
        blocks.mean(axis=2),
        blocks.std(axis=2),
        target[:, -1:],
        (target[:, -1:] - target[:, :1]),
    ], axis=1)
    if not np.isfinite(vals).all():
        raise ValueError("Non-finite synthetic or real target context")
    return vals


def nearest_context_aux(real_x: np.ndarray, synthetic_pair,
                        bins: int = 12):
    """Pair VAE target context with real auxiliary windows by nearest target context.

    Outputs the original synthetic target windows/labels unchanged and only
    substitutes auxiliary channels from a real training window. This is an
    approximate observational pairing, NOT true time-index alignment and
    NOT a validated conditional generator.
    """
    syn_x, syn_y = synthetic_pair
    if real_x.ndim != 3 or real_x.shape[2] < 2:
        raise ValueError("Need at least one auxiliary channel")
    if syn_x.shape[1:] != real_x.shape[1:] or len(syn_x) != len(syn_y):
        raise ValueError("Incompatible synthetic/real window shapes")
    f_real = context_features(real_x, bins=bins)
    f_syn = context_features(syn_x, bins=bins)
    mu = f_real.mean(axis=0)
    sd = np.maximum(f_real.std(axis=0), 1e-6)
    tree = cKDTree((f_real - mu) / sd)
    distances, indices = tree.query((f_syn - mu) / sd, k=1, workers=1)
    matched = syn_x.copy()
    matched[:, :, 1:] = real_x[np.asarray(indices, dtype=int), :, 1:]
    return (matched, syn_y.copy()), {
        "context_distance_median": float(np.median(distances)),
        "context_unique_real_fraction": float(len(np.unique(indices)) / len(indices)),
    }


def permute_duplicate_aux(real_pair, pool_seed: int):
    """Negative control: identical target windows, permuted aux ONLY in added half."""
    x, y = real_pair
    n = len(x)
    if n < 2 or x.shape[2] < 2:
        raise ValueError("At least 2 real windows and 2 channels required")
    doubled_x = np.concatenate([x, x.copy()], axis=0)
    doubled_y = np.concatenate([y, y.copy()], axis=0)
    ids = np.random.default_rng(pool_seed + 123_457).permutation(n)
    if np.array_equal(ids, np.arange(n)):
        ids = np.roll(ids, 1)
    doubled_x[n:, :, 1:] = x[ids, :, 1:]
    return doubled_x, doubled_y


def correlation_gap(real_x: np.ndarray, synthetic_x: np.ndarray) -> dict:
    """Descriptive contemporaneous target/aux correlations, train-only.

    It is NOT a measure of forecasting benefit or true future-validity.
    """
    if real_x.shape[2] != synthetic_x.shape[2]:
        raise ValueError("Need same channel count")
    def corr(x, aux_idx):
        target = x[:, :, 0].reshape(-1).astype(np.float64)
        aux = x[:, :, aux_idx].reshape(-1).astype(np.float64)
        if min(np.std(target), np.std(aux)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(target, aux)[0, 1])
    real_corr = [corr(real_x, c) for c in range(1, real_x.shape[2])]
    syn_corr = [corr(synthetic_x, c) for c in range(1, synthetic_x.shape[2])]
    diff = np.abs(np.asarray(real_corr) - np.asarray(syn_corr))
    return {
        "real_target_aux_corr": json.dumps(real_corr),
        "synth_target_aux_corr": json.dumps(syn_corr),
        "mean_abs_corr_gap": float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan"),
    }


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    missing = set(KEY_COLS) - set(d.columns)
    if missing:
        raise ValueError(f"Existing {path} missing required columns: {sorted(missing)}")
    if d.duplicated(list(KEY_COLS)).any():
        raise ValueError(f"Duplicate condition keys in {path}; audit before continuing")
    return set(d.loc[:, list(KEY_COLS)].itertuples(index=False, name=None))


def run_one(*, filename: str, regime: float, rep: int, pool_seed: int,
            epochs: int, vae_epochs: int, batch_size: int, lr: float,
            context_bins: int, gpu_index: int, allow_cpu: bool,
            out: Path) -> None:
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / "data" / "synthetic" / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; use scripts.asoc_prepare_backbone first")
    device = select_device(allow_cpu=allow_cpu, gpu_index=gpu_index)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    print(f"DEVICE: {device} | GPU: {gpu_name} | PyTorch {torch.__version__}", flush=True)
    data_array = np.load(path).astype(np.float32)
    if data_array.ndim != 2 or data_array.shape[1] < 2:
        raise ValueError("Phase 3 requires multivariate (T, C>=2) series")
    wcfg = WindowConfig(lookback=96, horizon=24)
    data = prepare(data_array, regime=regime, cfg=wcfg)
    real = data["train"]
    t1, _ = chrono_split(len(data_array))
    tstart = apply_regime(t1, regime, wcfg.lookback)
    standardized, _, _ = standardize(data_array, slice(tstart, t1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    vae_random = gen_vae(standardized[tstart:t1], wcfg, ratio=1.0,
                         seed=pool_seed, epochs=vae_epochs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gen_seconds = round(time.perf_counter() - t0, 3)
    if len(vae_random[0]) != len(real[0]):
        raise RuntimeError("VAE pool must be 1:1 with real windows")
    t0 = time.perf_counter()
    vae_context, ctx_stats = nearest_context_aux(real[0], vae_random,
                                                 bins=context_bins)
    context_seconds = round(time.perf_counter() - t0, 3)
    duplicated = (np.concatenate([real[0], real[0]], axis=0),
                  np.concatenate([real[1], real[1]], axis=0))
    duplicate_permuted = permute_duplicate_aux(real, pool_seed=pool_seed)
    vae_random_all = (np.concatenate([real[0], vae_random[0]], axis=0),
                      np.concatenate([real[1], vae_random[1]], axis=0))
    vae_context_all = (np.concatenate([real[0], vae_context[0]], axis=0),
                       np.concatenate([real[1], vae_context[1]], axis=0))
    permuted_all = duplicate_permuted
    original_steps, matched_steps = count_steps(len(real[0]), len(vae_random[0]),
                                                batch_size, epochs)
    interval = math.ceil((len(real[0]) + len(vae_random[0])) / batch_size)
    # All conditions share EXACTLY the same optimizer-step and validation budget.
    conditions = {
        "itr_target_step": (real, "target"),
        "itr_target_duplicate": (duplicated, "target"),
        "itr_target_vae": (vae_random_all, "target"),
        "itr_all_step": (real, "all"),
        "itr_all_duplicate": (duplicated, "all"),
        "itr_all_duplicate_permuted": (permuted_all, "all"),
        "itr_all_vae_random": (vae_random_all, "all"),
        "itr_all_vae_context": (vae_context_all, "all"),
    }
    aux_gaps = {
        "itr_all_duplicate": correlation_gap(real[0], duplicated[0][len(real[0]):]),
        "itr_all_duplicate_permuted": correlation_gap(real[0], permuted_all[0][len(real[0]):]),
        "itr_all_vae_random": correlation_gap(real[0], vae_random[0]),
        "itr_all_vae_context": correlation_gap(real[0], vae_context[0]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out)
    for condition in CONDITIONS:
        key = (path.name, regime, rep, pool_seed, epochs, vae_epochs,
               batch_size, lr, context_bins, condition)
        if key in done:
            print(f"SKIP {condition}: already recorded", flush=True)
            continue
        pair, input_mode = conditions[condition]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = fit_condition(
            arch="itransformer", train_pair=only_channels(pair, input_mode),
            val_pair=only_channels(data["val"], input_mode),
            test_pair=only_channels(data["test"], input_mode),
            init_seed=rep, max_steps=matched_steps,
            validation_interval=interval, batch_size=batch_size,
            lr=lr, device=device,
        )
        row = dict(zip(KEY_COLS, key))
        row.update(result)
        row.update({
            "arch": "itransformer", "input_mode": input_mode,
            "original_steps": original_steps, "matched_steps": matched_steps,
            "generator_seconds": gen_seconds, "context_match_seconds": context_seconds,
            "context_distance_median": ctx_stats["context_distance_median"],
            "context_unique_real_fraction": ctx_stats["context_unique_real_fraction"],
            "device": str(device), "gpu_name": gpu_name,
            "pytorch_version": str(torch.__version__),
            "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 2)
            if device.type == "cuda" else 0.0,
            "real_target_aux_corr": "", "synth_target_aux_corr": "",
            "mean_abs_corr_gap": float("nan"),
            "paired_aux_method": (
                "nearest_real_target_context_approx" if condition == "itr_all_vae_context"
                else "random_real_aux" if condition == "itr_all_vae_random"
                else "permuted_real_aux" if condition == "itr_all_duplicate_permuted"
                else "real_original_aux" if input_mode == "all" else "target_only"
            ),
        })
        row.update(aux_gaps.get(condition, {}))
        pd.DataFrame([row]).to_csv(out, mode="a", index=False,
                                   header=not out.exists())
        print(f"{condition:27s} steps={matched_steps:5d} "
              f"val={result['val_mse']:.6f} test={result['test_mse']:.6f}", flush=True)
    print(f"Completed/verified {len(CONDITIONS)} conditions -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="om0.5_plain_s0.npy")
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.10)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--context-bins", type=int, default=12)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true", help="CPU-only smoke testing")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if args.rep < 0 or args.epochs < 1 or args.vae_epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("rep>=0 and epochs, vae_epochs, batch_size, lr must be positive")
    if args.context_bins < 1 or 96 % args.context_bins:
        p.error("--context-bins must divide the fixed lookback 96")
    filename = Path(args.file)
    pool_seed = (int(m.group(1)) if (m := re.search(r"_s(\d+)\.npy$", filename.name)) else 0)
    if args.pool_seed is not None:
        pool_seed = args.pool_seed
    out = args.out if args.out.is_absolute() else ROOT / args.out
    run_one(filename=str(filename), regime=args.regime, rep=args.rep,
            pool_seed=pool_seed, epochs=args.epochs, vae_epochs=args.vae_epochs,
            batch_size=args.batch_size, lr=args.lr, context_bins=args.context_bins,
            gpu_index=args.gpu_index, allow_cpu=args.allow_cpu, out=out)


if __name__ == "__main__":
    main()
