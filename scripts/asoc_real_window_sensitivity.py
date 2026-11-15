"""Phase 9 (exploratory): Weather lookback/horizon sensitivity, equal-step controls.

New runs only for L/H 48/24 and 96/48. Phase 7's 96/24 is read-only.
No historical outputs or prepared arrays are modified.

Pilot:
  python -m scripts.asoc_real_window_sensitivity --dry-run
  python -m scripts.asoc_real_window_sensitivity --limit 1
  python -m scripts.asoc_real_window_sensitivity --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.asoc_budget_controls import count_steps, fit_condition, only_channels, select_device
from scripts.asoc_real_controls import ROOT, parse_ints, sha256
from src.data import WindowConfig, apply_regime, chrono_split, prepare, standardize

VERSION = "phase9_weather_window_sensitivity_v1"
DOMAIN, REGIME, ARCH, INPUT_MODE = "weather", 0.10, "itransformer", "target"
CONFIGS = ((48, 24), (96, 48))
REFERENCE = (96, 24)
CONDITIONS = ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
KEY = ("experiment_version", "domain", "data_sha256", "regime", "arch", "input_mode",
       "lookback", "horizon", "rep", "pool_seed", "condition", "epochs", "vae_epochs",
       "batch_size", "lr", "latent")
OUT = Path("results/asoc_revision/window_sensitivity_v1.csv")
POOLS = Path("results/asoc_revision/window_sensitivity_pools_v1")
REPORT = Path("results/asoc_revision/window_sensitivity_report_v1")
REFERENCE_CSV = Path("results/asoc_revision/real_budget_controls.csv")


def parse_configs(raw: str) -> tuple[tuple[int, int], ...]:
    result = []
    for spec in raw.split(","):
        try:
            a, b = spec.strip().split(":")
            cfg = (int(a), int(b))
        except ValueError as exc:
            raise ValueError("Use --configs 48:24,96:48") from exc
        if cfg not in CONFIGS or cfg in result:
            raise ValueError(f"Expected unique configurations in {CONFIGS}; 96:24 is read-only")
        result.append(cfg)
    if not result:
        raise ValueError("No configurations")
    return tuple(result)


def settings(*, epochs, vae_epochs, batch_size, lr, latent):
    if min(epochs, vae_epochs, batch_size, latent) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid training settings")
    return dict(epochs=epochs, vae_epochs=vae_epochs, batch_size=batch_size,
                lr=lr, latent=latent)


def meta(*, digest, lookback, horizon, rep=0, pool_seed=-1, condition="step_baseline", **kw):
    return dict(experiment_version=VERSION, domain=DOMAIN, data_sha256=digest,
                regime=REGIME, arch=ARCH, input_mode=INPUT_MODE, lookback=lookback,
                horizon=horizon, rep=rep, pool_seed=pool_seed, condition=condition, **kw)


def rowkey(row):
    return tuple(row[x] for x in KEY)


def read_runs(out: Path, digest: str) -> tuple[set, pd.DataFrame]:
    if not out.exists():
        return set(), pd.DataFrame(columns=KEY)
    df = pd.read_csv(out)
    required = set(KEY) | {"optimizer_steps", "test_mse", "test_mae", "mse_raw", "mae_raw",
                           "target_sd", "train_start", "train_end", "matched_steps", "original_steps", "pool_sha256"}
    if missing := required - set(df):
        raise ValueError(f"Output schema mismatch: {sorted(missing)}")
    if df.duplicated(list(KEY)).any():
        raise ValueError("Duplicate run keys; inspect the output before resuming")
    if not (df.data_sha256 == digest).all():
        raise ValueError("Dataset SHA changed; do not mix two Weather files")
    if not (df.experiment_version == VERSION).all():
        raise ValueError("Mixed experiment versions")
    if (set(df.condition) - set(CONDITIONS) or
        ((df.condition == "vae_augmented") & (df.pool_seed < 0)).any() or
        ((df.condition != "vae_augmented") & (df.pool_seed != -1)).any()):
        raise ValueError("Invalid condition or pool seed")
    vae = df[df.condition == "vae_augmented"]
    group = ["lookback", "horizon", "data_sha256", "vae_epochs", "latent", "pool_seed"]
    if len(vae) and "pool_sha256" in vae:
        if (vae.groupby(group, dropna=False).pool_sha256.nunique(dropna=False) > 1).any():
            raise ValueError("One pool seed corresponds to differing synthetic samples")
    return set(df[list(KEY)].itertuples(index=False, name=None)), df


def expected(*, lookback, horizon, reps, pool_seeds, digest, **kw):
    for rep in reps:
        for c in CONDITIONS[:3]:
            yield meta(digest=digest, lookback=lookback, horizon=horizon,
                       rep=rep, condition=c, **kw)
        for ps in pool_seeds:
            yield meta(digest=digest, lookback=lookback, horizon=horizon,
                       rep=rep, pool_seed=ps, condition="vae_augmented", **kw)


def active_timeline_check(raw_csv: Path, array_rows: int, train_start: int) -> str:
    """No calendar gaps in any active train/validation/test indices; earlier anomalies allowed."""
    if not raw_csv.exists():
        return "raw_csv_unavailable"
    ts = pd.to_datetime(pd.read_csv(raw_csv, usecols=["date"])["date"], errors="coerce")
    if len(ts) != array_rows:
        raise ValueError("Weather raw timestamp count does not match prepared array")
    active = ts.iloc[train_start:]
    if active.isna().any() or active.duplicated().any() or not (
        active.diff().iloc[1:] == pd.Timedelta(minutes=10)).all():
        raise ValueError("Weather active train/val/test has irregular timestamps")
    return "active_10min_regular"


def pool_metadata(*, digest, lookback, horizon, train_start, train_end,
                  vae_epochs, latent):
    return dict(version=VERSION, data_sha256=digest, domain=DOMAIN, regime=REGIME,
                lookback=lookback, horizon=horizon, train_start=train_start,
                train_end=train_end, vae_epochs=vae_epochs, latent=latent,
                generator="src.generators.gen_vae", ratio=1.0)


def pool_path(directory: Path, parameters: dict, seed: int) -> Path:
    token = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16]
    return directory / f"{DOMAIN}_L{parameters['lookback']}_H{parameters['horizon']}_s{seed}_{token}.npz"


def array_digest(xs: np.ndarray, ys: np.ndarray) -> str:
    h = hashlib.sha256()
    for x in (xs, ys):
        h.update(str(x.shape).encode())
        h.update(str(x.dtype).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def load_or_generate_pool(path: Path, parameters: dict, segment: np.ndarray,
                          cfg: WindowConfig, pool_seed: int,
                          n: int, device: torch.device):
    if path.exists():
        with np.load(path, allow_pickle=False) as obj:
            x, y = obj["x"], obj["y"]
            saved = str(obj["metadata"].item())
            stamp = str(obj["pool_sha256"].item())
        if saved != json.dumps(parameters, sort_keys=True):
            raise ValueError(f"Pool metadata mismatch: {path}")
        seconds = 0.0
        if stamp != array_digest(x, y):
            raise ValueError(f"Corrupted cached pool: {path}")
        cached = True
    else:
        from src.generators import gen_vae
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        x, y = gen_vae(segment, cfg, ratio=1.0, seed=pool_seed,
                       epochs=parameters["vae_epochs"], latent=parameters["latent"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        cached = False
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != (n, cfg.lookback, segment.shape[1]) or y.shape != (n, cfg.horizon):
        raise ValueError("VAE pool has unexpected shape")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite synthetic windows")
    stamp = array_digest(x, y)
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".temporary")
        try:
            with temp.open("wb") as f:
                np.savez_compressed(f, x=x, y=y,
                                    metadata=json.dumps(parameters, sort_keys=True),
                                    pool_sha256=stamp)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return (x, y), stamp, round(seconds, 3)


def run_config(*, cfg, reps, pool_seeds, digest, data_path, raw_csv, out,
               pool_dir, gpu_index, allow_cpu, dry_run, **kw):
    L, H = cfg
    work = list(expected(lookback=L, horizon=H, reps=reps, pool_seeds=pool_seeds,
                         digest=digest, **kw))
    done, prior = read_runs(out, digest)
    missing = [x for x in work if rowkey(x) not in done]
    print(f"Weather L={L} H={H}: {len(missing)}/{len(work)} pending", flush=True)
    if not missing or dry_run:
        return len(missing)
    device = select_device(allow_cpu=allow_cpu, gpu_index=gpu_index)
    X = np.load(data_path).astype(np.float32)
    if X.ndim != 2 or X.shape[1] < 1 or len(X) < 500 or not np.isfinite(X).all():
        raise ValueError("Invalid real data shape or non-finite values")
    window = WindowConfig(lookback=L, horizon=H)
    data = prepare(X, regime=REGIME, cfg=window)
    t1, t2 = chrono_split(len(X))
    train_start = apply_regime(t1, REGIME, L)
    if (t1 - train_start < L + H or t2 - t1 < H or len(X) - t2 < H):
        raise ValueError("Training period too short")
    cadence = active_timeline_check(raw_csv, len(X), train_start)
    if cadence != "active_10min_regular":
        print("WARNING: raw Weather time axis not available for cadence check", flush=True)
    Xs, mu, sd = standardize(X, slice(train_start, t1))
    segment = np.asarray(Xs[train_start:t1], dtype=np.float32)
    real = only_channels(data["train"], INPUT_MODE)
    val = only_channels(data["val"], INPUT_MODE)
    test = only_channels(data["test"], INPUT_MODE)
    n = len(real[0])
    if n != t1 - train_start - L - H + 1:
        raise ValueError("Unexpected real-window count")
    if not np.allclose(data["mu"], mu) or not np.allclose(data["sd"], sd):
        raise ValueError("Train-only standardization mismatch")
    baseline_steps, matched_steps = count_steps(n, n, kw["batch_size"], kw["epochs"])
    early_interval = math.ceil(n / kw["batch_size"])
    equal_interval = math.ceil(2 * n / kw["batch_size"])
    doubled = (np.concatenate([real[0], real[0]]),
               np.concatenate([real[1], real[1]]))
    target_sd = float(sd[0, 0])

    def save(common, pair, steps, interval, pool_sha="", gen_seconds=0.0, from_cache=False):
        k = rowkey(common)
        if k in done:
            return
        result = fit_condition(arch=ARCH, train_pair=pair, val_pair=val,
                               test_pair=test, init_seed=common["rep"],
                               max_steps=steps, validation_interval=interval,
                               batch_size=kw["batch_size"], lr=kw["lr"], device=device)
        if result["optimizer_steps"] != steps or not (np.isfinite(result["test_mse"]) and
                                                           np.isfinite(result["test_mae"])):
            raise ValueError("Trainer returned inconsistent steps or loss")
        row = {**common, **result, "mse_raw": result["test_mse"] * target_sd ** 2,
               "mae_raw": result["test_mae"] * target_sd,
               "val_mse_raw": result["val_mse"] * target_sd ** 2,
               "target_sd": target_sd, "target_mean": float(mu[0, 0]),
               "data_length": len(X), "train_start": train_start, "train_end": t1,
               "val_end": t2, "real_windows": n, "val_windows": len(val[0]),
               "test_windows": len(test[0]), "original_steps": baseline_steps,
               "matched_steps": matched_steps, "generator_seconds": gen_seconds,
               "pool_sha256": pool_sha, "generator_pool_cached": from_cache,
               "cadence_check": cadence,
               "normalization": "per_regime_train_segment_only",
               "val_checkpoint_selection": "minimum_validation_mse",
               "device": str(device), "pytorch_version": str(torch.__version__)}
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        done.add(k)
        print(f"  {common['condition']:<16} rep={common['rep']} pool={common['pool_seed']} "
              f"steps={steps} mse={result['test_mse']:.6f}", flush=True)

    for rep in reps:
        for c, pair, steps, interval in (
            ("epoch_baseline", real, baseline_steps, early_interval),
            ("step_baseline", real, matched_steps, equal_interval),
            ("duplicate_real", doubled, matched_steps, equal_interval)):
            save(meta(digest=digest, lookback=L, horizon=H, rep=rep,
                      condition=c, **kw), pair, steps, interval)

    params = pool_metadata(digest=digest, lookback=L, horizon=H,
                           train_start=train_start, train_end=t1,
                           vae_epochs=kw["vae_epochs"], latent=kw["latent"])
    for ps in pool_seeds:
        pending = [r for r in reps if rowkey(meta(digest=digest, lookback=L, horizon=H,
                            rep=r, pool_seed=ps, condition="vae_augmented", **kw)) not in done]
        if not pending:
            continue
        seeded_params = {**params, "pool_seed": ps}
        path_to_pool = pool_path(pool_dir, seeded_params, ps)
        from_cache = path_to_pool.exists()
        pair, stamp, seconds = load_or_generate_pool(path_to_pool,
            seeded_params, segment, window, ps, n, device)
        previous_pool = prior[(prior.lookback == L) & (prior.horizon == H) &
                              (prior.pool_seed == ps) & (prior.condition == "vae_augmented") &
                              (prior.vae_epochs == kw["vae_epochs"]) &
                              (prior.latent == kw["latent"])]
        if len(previous_pool) and not (previous_pool.pool_sha256 == stamp).all():
            raise ValueError("Cached VAE pool differs from previously trained rep; abort")
        synth = only_channels(pair, INPUT_MODE)
        augmented = (np.concatenate([real[0], synth[0]]),
                     np.concatenate([real[1], synth[1]]))
        for rep in pending:
            save(meta(digest=digest, lookback=L, horizon=H, rep=rep,
                      pool_seed=ps, condition="vae_augmented", **kw),
                 augmented, matched_steps, equal_interval, stamp, seconds, from_cache)
    return len(missing)


def paired_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only within-(L,H,rep,pool) comparisons, never cross-horizon raw MSE."""
    keys = [k for k in KEY if k not in ("condition", "pool_seed")]
    controls = df[df.condition != "vae_augmented"].set_index(keys + ["condition"], drop=False)
    paired, incomplete = [], []
    for _, aug in df[df.condition == "vae_augmented"].iterrows():
        base = tuple(aug[k] for k in keys)
        missing = [c for c in CONDITIONS[:3] if (*base, c) not in controls.index]
        if missing:
            incomplete.append(dict(lookback=int(aug.lookback), horizon=int(aug.horizon),
                                   rep=int(aug.rep), pool_seed=int(aug.pool_seed),
                                   missing=",".join(missing)))
            continue
        refs = {c: controls.loc[(*base, c)] for c in CONDITIONS[:3]}
        if any(not np.isclose(aug.target_sd, ref.target_sd, rtol=0, atol=1e-10)
               or int(aug.train_start) != int(ref.train_start)
               or int(aug.train_end) != int(ref.train_end)
               for ref in refs.values()):
            raise ValueError("Mismatched partitions or standardizer")
        if int(aug.optimizer_steps) != int(aug.matched_steps) or any(
                int(refs[c].optimizer_steps) != int(aug.matched_steps)
                for c in ("step_baseline", "duplicate_real")) or int(
                refs["epoch_baseline"].optimizer_steps) != int(aug.original_steps):
            raise ValueError("Optimizer-step budget mismatch")
        for row in (aug, *refs.values()):
            if not np.isfinite([row.test_mse, row.test_mae, row.mse_raw, row.mae_raw, row.target_sd]).all():
                raise ValueError("Nonfinite result")
            if (row.test_mse <= 0 or row.test_mae <= 0 or row.target_sd <= 0 or
                not np.isclose(row.mse_raw, row.test_mse * row.target_sd ** 2, rtol=1e-5) or
                not np.isclose(row.mae_raw, row.test_mae * row.target_sd, rtol=1e-5)):
                raise ValueError("Bad loss or original-scale conversion")
        item = {"domain": DOMAIN, "regime": REGIME, "lookback": int(aug.lookback),
                "horizon": int(aug.horizon), "rep": int(aug.rep),
                "pool_seed": int(aug.pool_seed), "source": "phase9_new",
                "matched_steps": int(aug.matched_steps),
                "vae_best_at_last": int(aug.best_step) == int(aug.optimizer_steps)}
        for metric in ("test_mse", "test_mae", "mse_raw", "mae_raw"):
            a = float(aug[metric]); item[f"vae_{metric}"] = a
            for name, ref in refs.items():
                b = float(ref[metric]); item[f"{name}_{metric}"] = b
                item[f"vae_vs_{name}_{metric}_pct"] = 100 * (a - b) / b
                item[f"vae_vs_{name}_{metric}_abs"] = a - b
        item["beats_both"] = (item["vae_vs_step_baseline_test_mse_pct"] < 0 and
                              item["vae_vs_duplicate_real_test_mse_pct"] < 0)
        paired.append(item)
    return pd.DataFrame(paired), pd.DataFrame(incomplete)


def phase7_reference(reference_path: Path, digest: str, reps: tuple[int, ...],
                     pool_seeds: tuple[int, ...], **kw) -> pd.DataFrame:
    """Read-only import of actual Phase 7 L=96,H=24 rows; strict protocol match."""
    if not reference_path.is_file():
        raise FileNotFoundError(f"Required read-only Phase 7 source absent: {reference_path}")
    from scripts.asoc_real_report import paired_table as phase7_pairs
    old = pd.read_csv(reference_path)
    match = ((old.experiment_version == "phase7_real_equal_step_v1") &
             (old.hidden == 128) & (np.isclose(old.horizon_weight, .5)) &
             (old.domain == DOMAIN) & (np.isclose(old.regime, REGIME)) &
             (old.arch == ARCH) & (old.input_mode == INPUT_MODE) &
             (old.lookback == REFERENCE[0]) & (old.horizon == REFERENCE[1]) &
             (old.rep.isin(reps)) &
             (((old.condition == "vae_augmented") & old.pool_seed.isin(pool_seeds)) |
              ((old.condition != "vae_augmented") & (old.pool_seed == -1))))
    for k, v in kw.items():
        match &= np.isclose(old[k], v) if k == "lr" else (old[k] == v)
    select = old[match].copy()
    if select.empty:
        raise ValueError("No compatible Phase 7 Weather reference rows")
    if not (select.data_sha256 == digest).all():
        raise ValueError("Phase 7 Weather reference data SHA differs")
    pair, missing = phase7_pairs(select)
    if not missing.empty:
        raise ValueError("Incomplete Phase 7 Weather reference pairs")
    actual = {(int(r.rep), int(r.pool_seed)) for _, r in pair.iterrows()}
    needed = {(r, s) for r in reps for s in pool_seeds}
    if needed - actual:
        raise ValueError(f"Phase 7 reference missing model/pool pairs: {sorted(needed - actual)}")
    result = []
    for _, r in pair.iterrows():
        entry = {"domain": DOMAIN, "regime": REGIME, "lookback": 96,
                 "horizon": 24, "rep": int(r.rep), "pool_seed": int(r.pool_seed),
                 "source": "phase7_read_only", "matched_steps": int(r.matched_steps),
                 "vae_best_at_last": bool(r.vae_best_at_last),
                 "beats_both": bool(r.vae_beats_both_matched)}
        for metric in ("test_mse", "test_mae", "mse_raw", "mae_raw"):
            entry[f"vae_{metric}"] = float(r[f"vae_{metric}"])
            for c in CONDITIONS[:3]:
                entry[f"{c}_{metric}"] = float(r[f"{c}_{metric}"])
                entry[f"vae_vs_{c}_{metric}_pct"] = float(r[f"vae_vs_{c}_{metric}_pct"])
                entry[f"vae_vs_{c}_{metric}_abs"] = float(r[f"vae_vs_{c}_{metric}_abs"])
        result.append(entry)
    return pd.DataFrame(result)


def make_reports(new: pd.DataFrame, old: pd.DataFrame,
                 configs: tuple[tuple[int, int], ...], reps: tuple[int, ...],
                 pool_seeds: tuple[int, ...]):
    df = pd.concat([old, new], ignore_index=True)
    if df.duplicated(["lookback", "horizon", "rep", "pool_seed"]).any():
        raise ValueError("Duplicated comparison between Phase 7 and Phase 9")
    pool_rows = []
    medcols = [c for c in df.columns if c.startswith("vae_vs_")]
    for (L, H, ps), block in df.groupby(["lookback", "horizon", "pool_seed"]):
        p = dict(lookback=int(L), horizon=int(H), pool_seed=int(ps),
                 n_model_seeds=block.rep.nunique(), n_pairs=len(block),
                 fraction_beats_both=float(block.beats_both.mean()),
                 source=block.source.iloc[0])
        for c in medcols:
            p["median_" + c] = float(block[c].median())
        pool_rows.append(p)
    pools = pd.DataFrame(pool_rows)
    summary = []
    for (L, H), block in pools.groupby(["lookback", "horizon"]):
        r = dict(domain=DOMAIN, regime=REGIME, lookback=int(L), horizon=int(H),
                 n_pools=block.pool_seed.nunique(),
                 n_model_seeds_min=int(block.n_model_seeds.min()),
                 n_model_seeds_max=int(block.n_model_seeds.max()),
                 n_pairs=int(block.n_pairs.sum()),
                 source=block.source.iloc[0])
        for c in medcols:
            vals = block["median_" + c]
            r["median_" + c] = float(vals.median())
            r["min_" + c] = float(vals.min())
            r["max_" + c] = float(vals.max())
        summary.append(r)
    expected_counts = {(L, H): len(reps)*len(pool_seeds) for L, H in (REFERENCE, *configs)}
    summ = pd.DataFrame(summary)
    summ["expected_pairs"] = summ.apply(
        lambda r: expected_counts[(int(r.lookback), int(r.horizon))], axis=1)
    summ["complete_for_requested_seeds"] = summ.n_pairs == summ.expected_pairs
    return df, pools, summ


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", default="48:24,96:48")
    p.add_argument("--reps", default="0")
    p.add_argument("--pool-seeds", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "real")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--pools", type=Path, default=POOLS)
    p.add_argument("--outdir", type=Path, default=REPORT)
    p.add_argument("--reference-csv", type=Path, default=REFERENCE_CSV)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Number of incomplete L/H configurations")
    p.add_argument("--report", action="store_true", help="Read-only comparison + report CSV")
    args = p.parse_args()
    try:
        configs = parse_configs(args.configs)
        reps, seeds = parse_ints(args.reps), parse_ints(args.pool_seeds)
        kw = settings(epochs=args.epochs, vae_epochs=args.vae_epochs,
                      batch_size=args.batch_size, lr=args.lr, latent=args.latent)
        if args.limit < 0 or args.gpu_index < 0:
            raise ValueError("Negative limit/GPU index")
        dest = lambda pth: pth if pth.is_absolute() else ROOT / pth
        data_dir, out, pools = dest(args.data_dir), dest(args.out), dest(args.pools)
        reference = dest(args.reference_csv)
        path = data_dir / "weather.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Weather prepared array unavailable: {path}")
        digest = sha256(path)
        if args.report:
            _, raw = read_runs(out, digest)
            config_mask = pd.Series([(int(L), int(H)) in configs
                                     for L, H in zip(raw.lookback, raw.horizon)],
                                    index=raw.index, dtype=bool)
            raw = raw[(raw.rep.isin(reps)) &
                      ((raw.pool_seed == -1) | raw.pool_seed.isin(seeds)) & config_mask]
            for k, v in kw.items():
                raw = raw[np.isclose(raw[k], v) if k == "lr" else (raw[k] == v)]
            new, incomplete = paired_table(raw)
            if not incomplete.empty:
                print("WARNING: incomplete Phase 9 comparisons", flush=True)
            old = phase7_reference(reference, digest, reps, seeds, **kw)
            merged, per_pool, summary = make_reports(new, old, configs, reps, seeds)
            directory = dest(args.outdir)
            directory.mkdir(parents=True, exist_ok=True)
            merged.to_csv(directory / "window_paired.csv", index=False)
            per_pool.to_csv(directory / "window_per_pool.csv", index=False)
            summary.to_csv(directory / "window_summary.csv", index=False)
            incomplete.to_csv(directory / "window_incomplete.csv", index=False)
            missing_cfgs = [cfg for cfg in configs
                            if not ((summary.lookback == cfg[0]) &
                                    (summary.horizon == cfg[1])).any()]
            if missing_cfgs:
                print(f"INCOMPLETE: no paired results yet for {missing_cfgs}", flush=True)
            print(summary[["lookback", "horizon", "n_pools", "n_model_seeds_min",
                           "n_pairs", "expected_pairs", "complete_for_requested_seeds",
                           "median_vae_vs_duplicate_real_test_mse_pct",
                           "median_vae_vs_step_baseline_test_mse_pct"]].to_string(index=False))
            print("All MSE comparisons are within each L/H; do NOT rank cross-horizon raw MSE.")
            return
        done, _ = read_runs(out, digest)
        pending = []
        for cfg in configs:
            missing = [r for r in expected(lookback=cfg[0], horizon=cfg[1],
                                          reps=reps, pool_seeds=seeds, digest=digest, **kw)
                       if rowkey(r) not in done]
            print(f"L={cfg[0]} H={cfg[1]}: {len(missing)} pending runs", flush=True)
            if missing:
                pending.append(cfg)
        if args.dry_run:
            print("Dry-run complete; no training or output files written.")
            return
        if not pending:
            print("No missing runs.")
            return
        for cfg in pending[:args.limit or None]:
            run_config(cfg=cfg, reps=reps, pool_seeds=seeds, digest=digest,
                       data_path=path, raw_csv=data_dir / "weather" / "weather.csv",
                       out=out, pool_dir=pools, gpu_index=args.gpu_index,
                       allow_cpu=args.allow_cpu, dry_run=False, **kw)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
