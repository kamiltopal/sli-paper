"""ASOC Phase 7: real-domain, optimizer-step-matched augmentation controls.

Additive experiment: never alters original real_runs.csv or prepared arrays.
Target-only mode uses the original window VAE; multivariate iTransformer
mode uses the *joint-window* VAE, never randomly paired auxiliary channels.
Controls are stored only once per model seed (pool_seed=-1); independently
seeded synthetic pools are stored separately, preventing pseudo-replication.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.asoc_budget_controls import (count_steps, fit_condition,
                                          only_channels, select_device)
from src.data import WindowConfig, apply_regime, chrono_split, prepare, standardize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/asoc_revision/real_budget_controls.csv")
VERSION = "phase7_real_equal_step_v1"
DOMAINS = ("weather", "traffic", "electricity", "epias")
CONDITIONS = ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
CONTROL_CONDITIONS = CONDITIONS[:3]
CONTROL_POOL_SEED = -1
KEY = ("experiment_version", "domain", "data_sha256", "regime", "arch",
       "input_mode", "rep", "epochs", "batch_size", "lr", "vae_epochs",
       "latent", "hidden", "horizon_weight", "condition", "pool_seed")


def parse_ints(spec: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in spec.split(","))
    if not values or any(x < 0 for x in values) or len(set(values)) != len(values):
        raise ValueError("Expected distinct nonnegative integer seeds")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_completed(path: Path) -> tuple[set[tuple], pd.DataFrame]:
    if not path.exists():
        return set(), pd.DataFrame(columns=KEY)
    df = pd.read_csv(path)
    missing = set(KEY) - set(df.columns)
    if missing:
        raise ValueError(f"Output schema mismatch: {sorted(missing)}")
    if df.duplicated(list(KEY)).any():
        raise ValueError(f"Duplicate run keys in {path}; audit before resuming")
    invalid = set(df["condition"].unique()) - set(CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown conditions: {invalid}")
    bad = df.loc[((df.condition == "vae_augmented") & (df.pool_seed < 0)) |
                 ((df.condition != "vae_augmented") & (df.pool_seed != CONTROL_POOL_SEED))]
    if len(bad):
        raise ValueError("Pool seeds must be -1 for controls and nonnegative for VAE")
    return set(df.loc[:, list(KEY)].itertuples(index=False, name=None)), df


def build_key(common: dict, condition: str, rep: int, pool_seed: int) -> tuple:
    row = {**common, "condition": condition, "rep": rep, "pool_seed": pool_seed}
    return tuple(row[k] for k in KEY)


def _combine(real, extra):
    rx, ry = real
    sx, sy = extra
    if sx.shape != rx.shape or sy.shape != ry.shape or not all(
            np.isfinite(v).all() for v in (rx, ry, sx, sy)):
        raise ValueError("Expected finite 1:1 synthetic pools with matching dimensions")
    return np.concatenate((rx, sx)), np.concatenate((ry, sy))


def run_group(*, domain: str, regime: float, arch: str, input_mode: str,
              reps: tuple[int, ...], pool_seeds: tuple[int, ...], epochs: int,
              vae_epochs: int, batch_size: int, lr: float, latent: int,
              hidden: int, horizon_weight: float, gpu_index: int,
              allow_cpu: bool, out: Path, data_dir: Path | None = None) -> None:
    if domain not in DOMAINS or regime not in (0.10, 0.25, 1.0):
        raise ValueError("Unsupported domain or regime")
    if arch not in ("dlinear", "patchtst", "itransformer"):
        raise ValueError("Unsupported architecture")
    if input_mode not in ("target", "all") or (input_mode == "all" and arch != "itransformer"):
        raise ValueError("All-channel mode requires iTransformer; no random auxiliary pairing")
    if input_mode == "all" and regime < 1.0 and domain != "epias":
        raise ValueError(
            "Benchmark .npy auxiliary channels were selected from the full 70% train span, "
            "not this scarce training slice. Use --input-mode target, or validate a "
            "separately versioned, regime-internal covariate array first. "
            "All-channel runs on these benchmark arrays are permitted only at regime=1.00."
        )
    if (not reps or not pool_seeds or min(reps) < 0 or min(pool_seeds) < 0 or
            min(epochs, vae_epochs, batch_size, latent, hidden) < 1 or
            lr <= 0 or not 0 < horizon_weight < 1):
        raise ValueError("Invalid training or seed parameters")
    if len(set(reps)) != len(reps) or len(set(pool_seeds)) != len(pool_seeds):
        raise ValueError("Repeated seeds")
    directory = data_dir if data_dir is not None else ROOT / "data" / "real"
    path = directory / f"{domain}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python -m scripts.prepare_real")
    digest = sha256(path)
    common = dict(experiment_version=VERSION, domain=domain, data_sha256=digest,
                  regime=regime, arch=arch, input_mode=input_mode,
                  epochs=epochs, batch_size=batch_size, lr=lr,
                  vae_epochs=vae_epochs, latent=latent, hidden=hidden,
                  horizon_weight=horizon_weight)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, prior = read_completed(out)
    # Fail closed if the same domain was regenerated or changed between runs.
    changed = prior[(prior.domain == domain) & (prior.data_sha256 != digest)]
    if len(changed):
        raise ValueError(f"Data changed for {domain}; use a NEW output CSV and audit provenance")
    expected = [build_key(common, c, rep, CONTROL_POOL_SEED)
                for rep in reps for c in CONTROL_CONDITIONS]
    expected += [build_key(common, "vae_augmented", rep, ps)
                 for ps in pool_seeds for rep in reps]
    missing = [key for key in expected if key not in done]
    print(f"{domain} regime={regime} arch={arch}/{input_mode}: "
          f"{len(missing)}/{len(expected)} results to compute", flush=True)
    if not missing:
        return
    device = select_device(allow_cpu=allow_cpu, gpu_index=gpu_index)
    print(f"Device: {device} / {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
          flush=True)
    array = np.load(path).astype(np.float32)
    if array.ndim != 2 or array.shape[1] < 1 or len(array) < 500 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite (T,C) real data, T>=500: {path}")
    if input_mode == "all" and array.shape[1] < 2:
        raise ValueError("Joint VAE requires at least two channels")
    cfg = WindowConfig(lookback=96, horizon=24)
    data = prepare(array, regime=regime, cfg=cfg)
    n = len(data["train"][0])
    original_steps, matched_steps = count_steps(n, n, batch_size, epochs)
    original_interval = math.ceil(n / batch_size)
    matched_interval = math.ceil(2 * n / batch_size)
    t1, _ = chrono_split(len(array))
    tstart = apply_regime(t1, regime, cfg.lookback)
    Xs, mu, sd = standardize(array, slice(tstart, t1))
    segment = np.asarray(Xs[tstart:t1], dtype=np.float32)
    target_mean, target_sd = float(mu[0, 0]), float(sd[0, 0])
    real = only_channels(data["train"], input_mode)
    val = only_channels(data["val"], input_mode)
    test = only_channels(data["test"], input_mode)
    doubled = _combine(real, real)

    def record(condition: str, rep: int, pool_seed: int, pair, steps: int,
               interval: int, generator_seconds: float, generator_stats: dict):
        key = build_key(common, condition, rep, pool_seed)
        if key in done:
            print(f"SKIP {condition} rep={rep} pool={pool_seed}", flush=True)
            return
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = fit_condition(arch=arch, train_pair=pair, val_pair=val,
                                test_pair=test, init_seed=rep, max_steps=steps,
                                validation_interval=interval, batch_size=batch_size,
                                lr=lr, device=device)
        row = {**common, "rep": rep, "condition": condition,
               "pool_seed": pool_seed, **metrics,
               "mse_raw": metrics["test_mse"] * target_sd ** 2,
               "mae_raw": metrics["test_mae"] * target_sd,
               "val_mse_raw": metrics["val_mse"] * target_sd ** 2,
               "target_mean": target_mean, "target_sd": target_sd,
               "data_length": len(array), "train_start": tstart,
               "train_end": t1, "real_windows": n,
               "val_windows": len(val[0]), "test_windows": len(test[0]),
               "original_steps": original_steps, "matched_steps": matched_steps,
               "generator_seconds": generator_seconds,
               "generator_name": ("none" if condition != "vae_augmented"
                                  else "vae_target_only" if input_mode == "target"
                                  else "vae_joint_window"),
               "generator_last_loss": generator_stats.get("joint_last_batch_loss", np.nan),
               "device": str(device), "pytorch_version": str(torch.__version__),
               "gpu_peak_memory_mb": (
                   round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
                   if device.type == "cuda" else 0.0),
               "lookback": cfg.lookback, "horizon": cfg.horizon,
               "normalization": "per_regime_train_segment_only",
               "val_checkpoint_selection": "minimum_validation_mse"}
        pd.DataFrame([row]).to_csv(out, mode="a", index=False,
                                    header=not out.exists())
        done.add(key)
        print(f"  {condition:16s} rep={rep} pool={pool_seed:2d} "
              f"steps={steps:5d} test={metrics['test_mse']:.6f} "
              f"raw_mse={row['mse_raw']:.6f}", flush=True)

    for rep in reps:
        record("epoch_baseline", rep, CONTROL_POOL_SEED, real,
               original_steps, original_interval, np.nan, {})
        record("step_baseline", rep, CONTROL_POOL_SEED, real,
               matched_steps, matched_interval, np.nan, {})
        record("duplicate_real", rep, CONTROL_POOL_SEED, doubled,
               matched_steps, matched_interval, np.nan, {})

    for ps in pool_seeds:
        pending = [rep for rep in reps if build_key(common, "vae_augmented", rep, ps) not in done]
        if not pending:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        if input_mode == "target":
            from src.generators import gen_vae
            synth = gen_vae(segment, cfg, ratio=1.0, seed=ps,
                            epochs=vae_epochs, latent=latent)
            syn_stats = {}
        else:
            from scripts.asoc_joint_vae import gen_joint_vae
            synth, syn_stats = gen_joint_vae(data["train"], seed=ps,
                                             epochs=vae_epochs, latent=latent,
                                             hidden=hidden, horizon_weight=horizon_weight,
                                             device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gen_seconds = round(time.perf_counter() - tick, 3)
        aug = _combine(real, only_channels(synth, input_mode))
        for rep in pending:
            record("vae_augmented", rep, ps, aug, matched_steps,
                   matched_interval, gen_seconds, syn_stats)
    print(f"Complete: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=DOMAINS, default="electricity")
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.25)
    p.add_argument("--arch", choices=("dlinear", "patchtst", "itransformer"), default="dlinear")
    p.add_argument("--input-mode", choices=("target", "all"), default="target")
    p.add_argument("--reps", default="0")
    p.add_argument("--pool-seeds", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon-weight", type=float, default=0.5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true", help="CPU fallback for smoke tests")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=None, help="Real .npy directory")
    args = p.parse_args()
    try:
        reps = parse_ints(args.reps)
        pools = parse_ints(args.pool_seeds)
        run_group(domain=args.domain, regime=args.regime, arch=args.arch,
                  input_mode=args.input_mode, reps=reps, pool_seeds=pools,
                  epochs=args.epochs, vae_epochs=args.vae_epochs,
                  batch_size=args.batch_size, lr=args.lr, latent=args.latent,
                  hidden=args.hidden, horizon_weight=args.horizon_weight,
                  gpu_index=args.gpu_index, allow_cpu=args.allow_cpu,
                  out=args.out if args.out.is_absolute() else ROOT / args.out,
                  data_dir=args.data_dir)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
