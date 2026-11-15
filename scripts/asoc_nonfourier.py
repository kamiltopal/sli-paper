"""Phase 12: pre-fixed non-Fourier stochastic dynamics, matched-step VAE comparison.

No old result/data files are changed. Every simulation is deterministic per
family/series ID, and all candidate series are pre-specified before outcomes.

Pilot:
  python -m scripts.asoc_nonfourier --families nonlinear_ar --ids 0 --regimes 0.10 --reps 0 --pool-seeds 0 --dry-run
  python -m scripts.asoc_nonfourier --families nonlinear_ar --ids 0 --regimes 0.10 --reps 0 --pool-seeds 0 --limit 1
Expanded, pre-specified matrix:
  python -m scripts.asoc_nonfourier --families nonlinear_ar,markov_ar --ids 0,1,2 --regimes 0.10,1.00 --reps 0 --pool-seeds 0
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

from scripts.asoc_budget_controls import count_steps, fit_condition, select_device
from src.data import WindowConfig, apply_regime, chrono_split, prepare, standardize

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/asoc_revision/nonfourier")
OUT = BASE / "nonfourier_runs.csv"
DATA_DIR = BASE / "data"
VERSION = "phase12_nonfourier_fixed_v1"
FAMILIES = ("nonlinear_ar", "markov_ar")
IDS = (0, 1, 2)
LENGTH = 12000
BURN_IN = 500
CFG = WindowConfig(96, 24)
CONDITIONS = ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
CONTROLS = CONDITIONS[:3]
KEY = ("experiment_version", "family", "series_id", "data_sha256", "length", "regime",
       "arch", "rep", "pool_seed", "condition", "epochs", "batch_size", "lr",
       "vae_epochs", "latent", "lookback", "horizon")


def parse_csv_choices(s: str, allowed: tuple[str, ...]):
    values = tuple(part.strip() for part in s.split(","))
    if not values or any(v not in allowed for v in values) or len(values) != len(set(values)):
        raise ValueError(f"Expected distinct values from: {allowed}")
    return values


def parse_ids(s: str, allowed=IDS):
    try:
        result = tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise ValueError("Invalid integer IDs") from exc
    if not result or len(result) != len(set(result)) or any(i not in allowed for i in result):
        raise ValueError(f"IDs must be unique and from {allowed}")
    return result


def series_seed(family: str, series_id: int) -> int:
    if family not in FAMILIES or series_id not in IDS:
        raise ValueError("Unknown family or series ID")
    return (41000 if family == "nonlinear_ar" else 51000) + series_id


def simulate(family: str, series_id: int, length: int = LENGTH) -> np.ndarray:
    """Fixed, stable, stochastic non-sinusoidal DGPs, with 500-point burn-in.

    nonlinear_ar: tanh AR with bilinear interaction + Gaussian innovations.
    markov_ar: hidden two-state Markov-switching AR(1), heterogeneous level,
    autoregressive coefficient and innovation variance.
    The latent state is NOT made available to any forecaster or generator.
    """
    if length < 2000:
        raise ValueError("length must be >=2000 for L=96,H=24, and scarce training")
    rng = np.random.default_rng(series_seed(family, series_id))
    total = length + BURN_IN
    eps = rng.standard_normal(total)
    x = np.zeros(total, dtype=np.float64)
    if family == "nonlinear_ar":
        for t in range(2, total):
            a, b = x[t - 1], x[t - 2]
            x[t] = (0.67 * np.tanh(1.5 * a) - 0.23 * np.tanh(b)
                    + 0.16 * np.tanh(2.7 * a * b) + 0.28 * eps[t])
    elif family == "markov_ar":
        state = 0
        for t in range(1, total):
            stay = 0.985 if state == 0 else 0.978
            if rng.random() > stay:
                state = 1 - state
            mu, phi, sigma = ((-0.7, 0.80, 0.23) if state == 0
                              else (1.1, 0.38, 0.48))
            x[t] = mu + phi * (x[t - 1] - mu) + sigma * eps[t]
    else:
        raise ValueError(f"Unknown family: {family}")
    arr = x[BURN_IN:].astype(np.float32)[:, None]
    if not np.isfinite(arr).all() or np.std(arr[:, 0]) < 0.01:
        raise ValueError("Invalid generated dynamics")
    return arr


def array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def series_path(data_dir: Path, family: str, series_id: int) -> Path:
    return data_dir / f"{family}_{series_id:02d}.npy"


def ensure_series(data_dir: Path, family: str, series_id: int, *, write: bool) -> tuple[np.ndarray, str]:
    """Regenerate deterministically and refuse changes to any existing .npy.

    On a dry run, do not write anything. On a real run, only CREATE missing
    dedicated Phase 12 arrays; never overwrite (or modify any earlier data).
    """
    expected = simulate(family, series_id)
    digest = array_sha256(expected)
    path = series_path(data_dir, family, series_id)
    if path.exists():
        actual = np.load(path, allow_pickle=False)
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"Existing series differs from fixed protocol: {path}")
    elif write:
        data_dir.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as fh:
            np.save(fh, expected)
    return expected, digest


def read_runs(path: Path) -> tuple[set[tuple], pd.DataFrame]:
    if not path.exists():
        return set(), pd.DataFrame(columns=list(KEY))
    data = pd.read_csv(path)
    if missing := set(KEY).difference(data.columns):
        raise ValueError(f"Missing run columns: {sorted(missing)}")
    if data.duplicated(list(KEY)).any():
        raise ValueError("Duplicate run IDs; audit output before resume")
    if not data.condition.isin(CONDITIONS).all():
        raise ValueError("Unknown run condition")
    if ((data.condition == "vae_augmented") & (data.pool_seed < 0)).any() or (
        (data.condition != "vae_augmented") & (data.pool_seed != -1)).any():
        raise ValueError("Invalid pool seed for control / synthetic run")
    if not (data.experiment_version == VERSION).all():
        raise ValueError("Output includes incompatible protocol versions")
    return set(data[list(KEY)].itertuples(index=False, name=None)), data


def fixed_row(family, series_id, digest, regime, arch, rep, pool_seed,
              condition, *, length=LENGTH, epochs=10, batch_size=64,
              lr=0.001, vae_epochs=30, latent=16):
    return dict(experiment_version=VERSION, family=family, series_id=series_id,
                data_sha256=digest, length=length, regime=regime, arch=arch,
                rep=rep, pool_seed=pool_seed, condition=condition, epochs=epochs,
                batch_size=batch_size, lr=lr, vae_epochs=vae_epochs, latent=latent,
                lookback=CFG.lookback, horizon=CFG.horizon)


def key_of(row: dict) -> tuple:
    return tuple(row[k] for k in KEY)


def expected_rows(*, family, series_id, digest, regime, arch, reps, pool_seeds,
                  length=LENGTH, epochs=10, batch_size=64, lr=0.001,
                  vae_epochs=30, latent=16):
    kw = dict(family=family, series_id=series_id, digest=digest, regime=regime,
              arch=arch, length=length, epochs=epochs, batch_size=batch_size,
              lr=lr, vae_epochs=vae_epochs, latent=latent)
    for rep in reps:
        for c in CONTROLS:
            yield fixed_row(rep=rep, pool_seed=-1, condition=c, **kw)
        for ps in pool_seeds:
            yield fixed_row(rep=rep, pool_seed=ps, condition="vae_augmented", **kw)


def _combine(a, b):
    x, y = a
    sx, sy = b
    if x.shape != sx.shape or y.shape != sy.shape or not all(
        np.isfinite(k).all() for k in (x, y, sx, sy)
    ):
        raise ValueError("Synthetic windows must match real windows 1:1 and be finite")
    return np.concatenate([x, sx]), np.concatenate([y, sy])


def simple_references(train_pair, test_pair) -> dict[str, float]:
    """Forecast skill diagnostic only; direct AR(2) fits TRAIN windows alone."""
    tx, ty = train_pair
    x, y = test_pair
    last = np.broadcast_to(x[:, -1:, 0], y.shape)
    mean = np.broadcast_to(float(np.mean(ty)), y.shape)
    # Direct multi-horizon lag-2 linear least squares, with ridge regularizer.
    A = np.column_stack([tx[:, -2, 0], tx[:, -1, 0], np.ones(len(tx))])
    B = np.column_stack([x[:, -2, 0], x[:, -1, 0], np.ones(len(x))])
    regularizer = np.diag([1e-3, 1e-3, 0.0])
    coef = np.linalg.solve(A.T @ A + regularizer, A.T @ ty)
    pred_ar2 = B @ coef
    def mse(pred):
        return float(np.mean((pred.astype(np.float64) - y) ** 2))
    return dict(persistence_mse=mse(last), trainmean_mse=mse(mean),
                direct_ar2_mse=mse(pred_ar2))


def run_group(*, family, series_id, digest, array, regime, arch, reps,
              pool_seeds, epochs, batch_size, lr, vae_epochs, latent,
              device, out: Path, done: set[tuple]) -> None:
    cfg = CFG
    data = prepare(array, regime, cfg)
    real, val, test = data["train"], data["val"], data["test"]
    n = len(real[0])
    original, matched = count_steps(n, n, batch_size, epochs)
    original_interval, matched_interval = math.ceil(n / batch_size), math.ceil(2*n / batch_size)
    train_end, _ = chrono_split(len(array))
    start = apply_regime(train_end, regime, cfg.lookback)
    standardized, mu, sd = standardize(array, slice(start, train_end))
    segment = np.asarray(standardized[start:train_end], dtype=np.float32)
    refs = simple_references(real, test)
    doubled = _combine(real, real)

    def record(base, pair, steps, interval, gen_seconds):
        k = key_of(base)
        if k in done:
            return
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = fit_condition(arch=arch, train_pair=pair, val_pair=val,
                                test_pair=test, init_seed=base["rep"],
                                max_steps=steps, validation_interval=interval,
                                batch_size=batch_size, lr=lr, device=device)
        scale = float(sd[0, 0])
        row = {**base, **metrics, **refs,
               "mse_raw": metrics["test_mse"] * scale**2,
               "mae_raw": metrics["test_mae"] * scale,
               "target_sd": scale, "target_mean": float(mu[0, 0]),
               "real_windows": n, "train_start": start,
               "train_end": train_end, "val_windows": len(val[0]),
               "test_windows": len(test[0]),
               "original_steps": original, "matched_steps": matched,
               "generator_seconds": gen_seconds,
               "device": str(device), "torch_version": str(torch.__version__),
               "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                    if device.type == "cuda" else 0.,
               "normalization": "selected_training_slice_only",
               "val_checkpoint_selection": "minimum_validation_mse"}
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False, mode="a", header=not out.exists())
        done.add(k)
        print(f"  {family}/{series_id} r={regime:.2f} {arch}: "
              f"{base['condition']:16} rep={base['rep']} ps={base['pool_seed']} "
              f"steps={steps} test_mse={metrics['test_mse']:.6f}", flush=True)

    def row(rep, ps, cond):
        return fixed_row(family, series_id, digest, regime, arch, rep, ps, cond,
                         length=len(array), epochs=epochs, batch_size=batch_size,
                         lr=lr, vae_epochs=vae_epochs, latent=latent)

    for rep in reps:
        record(row(rep, -1, "epoch_baseline"), real, original, original_interval, np.nan)
        record(row(rep, -1, "step_baseline"), real, matched, matched_interval, np.nan)
        record(row(rep, -1, "duplicate_real"), doubled, matched, matched_interval, np.nan)

    for ps in pool_seeds:
        pending = [rep for rep in reps if key_of(row(rep, ps, "vae_augmented")) not in done]
        if not pending:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        from src.generators import gen_vae
        synth = gen_vae(segment, cfg, ratio=1.0, seed=ps, epochs=vae_epochs, latent=latent)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gen_seconds = round(time.perf_counter() - t0, 3)
        augmented = _combine(real, synth)
        for rep in pending:
            record(row(rep, ps, "vae_augmented"), augmented,
                   matched, matched_interval, gen_seconds)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", default="nonlinear_ar,markov_ar")
    p.add_argument("--ids", default="0,1,2")
    p.add_argument("--regimes", default="0.10,1.00")
    p.add_argument("--archs", default="itransformer")
    p.add_argument("--reps", default="0")
    p.add_argument("--pool-seeds", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Max incomplete series/regime/arch groups")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args(argv)
    try:
        families = parse_csv_choices(args.families, FAMILIES)
        ids = parse_ids(args.ids)
        regimes = tuple(float(r) for r in parse_csv_choices(args.regimes,
                                 ("0.10", "0.1", "1.00", "1.0")))
        if len(regimes) != len(set(regimes)):
            raise ValueError("Duplicate training regimes")
        archs = parse_csv_choices(args.archs, ("itransformer", "dlinear", "patchtst"))
        from scripts.asoc_real_controls import parse_ints
        reps, pools = parse_ints(args.reps), parse_ints(args.pool_seeds)
        if min(args.epochs, args.vae_epochs, args.batch_size, args.latent) < 1 or args.lr <= 0 or args.limit < 0:
            raise ValueError("Invalid training budget or limit")
        out = args.out if args.out.is_absolute() else ROOT / args.out
        data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
        done, previous = read_runs(out)
        groups = []
        for family in families:
            for sid in ids:
                arr, digest = ensure_series(data_dir, family, sid, write=False)
                stale = previous[(previous.family == family) & (previous.series_id == sid) &
                                 (previous.data_sha256 != digest)]
                if len(stale):
                    raise ValueError(f"Series {family}/{sid} SHA differs from existing results")
                for regime in regimes:
                    for arch in archs:
                        expected = list(expected_rows(family=family, series_id=sid, digest=digest,
                            regime=regime, arch=arch, reps=reps, pool_seeds=pools,
                            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            vae_epochs=args.vae_epochs, latent=args.latent))
                        missing = sum(key_of(item) not in done for item in expected)
                        if missing:
                            groups.append((family, sid, digest, arr, regime, arch, missing))
        print(f"PLAN: {len(groups)} incomplete groups / "
              f"{len(families)*len(ids)*len(regimes)*len(archs)} configured; "
              f"{sum(g[-1] for g in groups)} new runs; out={out}", flush=True)
        for i, g in enumerate(groups, 1):
            print(f"  {i:03}: {g[0]}/{g[1]:02} regime={g[4]:.2f} "
                  f"arch={g[5]} pending={g[-1]}", flush=True)
        if args.dry_run or not groups:
            return
        device = select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
        print(f"DEVICE: {device}", flush=True)
        for family, sid, digest, _, regime, arch, _ in groups[:args.limit or None]:
            array, checked = ensure_series(data_dir, family, sid, write=True)
            if digest != checked:
                raise ValueError("Generated data changed during planning")
            run_group(family=family, series_id=sid, digest=digest, array=array,
                      regime=regime, arch=arch, reps=reps, pool_seeds=pools,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, vae_epochs=args.vae_epochs,
                      latent=args.latent, device=device, out=out, done=done)
        print("Complete or saved partial progress; rerun to resume.", flush=True)
    except (ValueError, FileNotFoundError) as err:
        p.error(str(err))


if __name__ == "__main__":
    main()
