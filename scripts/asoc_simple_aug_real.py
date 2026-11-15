"""Phase 8: fixed-step jitter/scaling controls for Weather/Traffic 10%.

New rows are written ONLY to results/asoc_revision/simple_aug_v1.csv.
The existing Phase7 1x VAE, duplicated-real, step and epoch baselines are
read-only. This is an exploratory, deliberately narrow reviewer R1.3 control.

PowerShell examples:
  python -m scripts.asoc_simple_aug_real --domain weather --dry-run
  python -m scripts.asoc_simple_aug_real --domain weather --limit 2
  python -m scripts.asoc_simple_aug_real --domain weather --report
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "phase8_simple_aug_equal_step_v1"
REF_VERSION = "phase7_real_equal_step_v1"
DOMAIN_FREQ = {"weather": "10min", "traffic": "1h"}
REGIME, ARCH, INPUT_MODE = 0.1, "itransformer", "target"
LOOKBACK, HORIZON = 96, 24
EPOCHS, BATCH, LR = 10, 64, .001
VAE_EPOCHS, LATENT, HIDDEN, HORIZON_WEIGHT = 30, 16, 128, .5
METHODS = ("jitter", "scaling")
JITTER_SIGMA, SCALE_SIGMA = .03, .10
REF_PATH = ROOT / "results/asoc_revision/real_budget_controls.csv"
OUT_PATH = ROOT / "results/asoc_revision/simple_aug_v1.csv"
POOL_DIR = ROOT / "results/asoc_revision/simple_aug_pools_v1"
REPORT_DIR = ROOT / "results/asoc_revision/simple_aug_report_v1"
KEY = ("experiment_version", "domain", "data_sha256", "regime", "arch",
       "input_mode", "method", "rep", "aug_seed")
REF_KEY = ("experiment_version", "domain", "data_sha256", "regime", "arch",
           "input_mode", "rep", "epochs", "batch_size", "lr", "vae_epochs",
           "latent", "hidden", "horizon_weight", "condition", "pool_seed")
METRICS = ("test_mse", "test_mae", "mse_raw", "mae_raw")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(x.strip()) for x in text.split(","))
    except ValueError as e:
        raise ValueError("Invalid comma-separated nonnegative seeds") from e
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError("Seeds must be distinct nonnegative integers")
    return values


def parse_methods(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(","))
    if not values or len(set(values)) != len(values) or not set(values).issubset(METHODS):
        raise ValueError(f"Expected a unique subset of {METHODS}")
    return values


def augmented_training_pair(real: tuple[np.ndarray, np.ndarray], method: str, seed: int,
                            jitter_sigma: float = JITTER_SIGMA,
                            scale_sigma: float = SCALE_SIGMA):
    """Return *only the synthetic extra* (1x) using training windows only.

    For each complete lookback+horizon training window, perturb both its
    input and label. A single scale factor covers each window's past/future;
    jitter is independent point-wise Gaussian noise at the standardized scale.
    No validation/test observations, no augmentation of evaluation windows.
    """
    x, y = real
    if (x.ndim != 3 or x.shape[2] != 1 or y.ndim != 2 or len(x) != len(y)
            or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Invalid finite target-only training windows")
    if method not in METHODS or min(jitter_sigma, scale_sigma) <= 0 or seed < 0:
        raise ValueError("Unknown augmentation or invalid perturbation parameter")
    rng = np.random.default_rng(seed)
    z = np.concatenate([x[:, :, 0], y], axis=1).astype(np.float32, copy=True)
    if method == "jitter":
        z += rng.normal(0.0, jitter_sigma, size=z.shape).astype(np.float32)
    else:
        factors = (1.0 + rng.normal(0.0, scale_sigma, size=(len(z), 1))).astype(np.float32)
        factors = np.clip(factors, .7, 1.3)
        z *= factors
    L = x.shape[1]
    sx, sy = z[:, :L, None].copy(), z[:, L:].copy()
    return sx, sy


def audit_timestamps(raw_path: Path, length: int, active_start: int, domain: str):
    if not raw_path.is_file():
        raise FileNotFoundError(f"Source chronology CSV required: {raw_path}")
    df = pd.read_csv(raw_path, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce")
    if len(dates) != length:
        raise ValueError("Raw timestamps and prepared array have different lengths")
    sub = dates.iloc[active_start:]
    if (sub.isna().any() or not sub.is_unique or
            not (sub.diff().dropna() == pd.Timedelta(DOMAIN_FREQ[domain])).all()):
        raise ValueError(f"Irregular {domain} timestamps in active train/validation/test; STOP")
    return str(sub.iloc[0]), str(sub.iloc[-1])


def validate_reference(ref_file: Path, domain: str, digest: str, reps, aug_seeds,
                       n: int, train_start: int, train_end: int):
    if not ref_file.is_file():
        raise FileNotFoundError(f"Phase7 raw results required (no retraining): {ref_file}")
    raw = pd.read_csv(ref_file)
    required = set(REF_KEY) | set(METRICS) | {"optimizer_steps", "matched_steps",
        "original_steps", "target_sd", "train_start", "train_end", "best_step"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Phase7 results lack columns: {sorted(missing)}")
    if raw.duplicated(list(REF_KEY)).any():
        raise ValueError("Phase7 results have duplicate experiment keys")
    spec = dict(experiment_version=REF_VERSION, domain=domain, data_sha256=digest,
        regime=REGIME, arch=ARCH, input_mode=INPUT_MODE, epochs=EPOCHS,
        batch_size=BATCH, lr=LR, vae_epochs=VAE_EPOCHS, latent=LATENT,
        hidden=HIDDEN, horizon_weight=HORIZON_WEIGHT)
    df = raw
    for k, val in spec.items():
        if isinstance(val, float):
            df = df[np.isclose(df[k].astype(float), val, atol=1e-10, rtol=0)]
        else:
            df = df[df[k] == val]
    refs = {(int(r["rep"]), str(r["condition"]), int(r["pool_seed"])): r
            for r in df.to_dict("records")}
    old, matched = EPOCHS * math.ceil(n / BATCH), EPOCHS * math.ceil(2*n / BATCH)
    expected = [(rep, condition, -1) for rep in reps
                for condition in ("epoch_baseline", "step_baseline", "duplicate_real")]
    expected += [(rep, "vae_augmented", s) for rep in reps for s in aug_seeds]
    for key in expected:
        if key not in refs:
            raise ValueError(f"Missing Phase7 1x reference {domain} {key}; do not auto-train")
        row = refs[key]
        step = old if key[1] == "epoch_baseline" else matched
        if (int(row["optimizer_steps"]) != step or int(row["matched_steps"]) != matched
            or int(row["original_steps"]) != old or int(row["train_start"]) != train_start
            or int(row["train_end"]) != train_end):
            raise ValueError(f"Reference budget/split mismatch: {domain} {key}")
        if not all(np.isfinite(float(row[k])) and float(row[k]) > 0
                   for k in METRICS + ("target_sd",)):
            raise ValueError(f"Nonpositive/nonfinite Phase7 metric: {key}")
        if not np.isclose(float(row["mse_raw"]), float(row["test_mse"])*float(row["target_sd"])**2, rtol=1e-5, atol=1e-7):
            raise ValueError(f"Phase7 raw-scale MSE conversion mismatch: {key}")
    return refs, old, matched


def load_new(path: Path, domain: str, digest: str, matched: int) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=KEY)
    df = pd.read_csv(path)
    required = set(KEY) | set(METRICS) | {"optimizer_steps", "matched_steps",
        "target_sd", "pool_file_sha256", "best_step", "epochs", "batch_size", "lr",
        "vae_epochs", "latent", "lookback", "horizon", "jitter_sigma", "scale_sigma"}
    if missing := required - set(df.columns):
        raise ValueError(f"Previous simple-augmentation log lacks {sorted(missing)}")
    if df.duplicated(list(KEY)).any():
        raise ValueError("Duplicate simple-augmentation run keys")
    if not (df.experiment_version == VERSION).all() or not df.domain.isin(DOMAIN_FREQ).all():
        raise ValueError("Unexpected experiment version or domain in previous results")
    old = df[df.domain == domain]
    if len(old) and not (old.data_sha256 == digest).all():
        raise ValueError("Data hash changed since previous augmentation results")
    if len(old) and not (old.optimizer_steps == matched).all():
        raise ValueError("Previous simple augmentation used a different optimizer budget")
    frozen = {"regime": REGIME, "arch": ARCH, "input_mode": INPUT_MODE,
        "epochs": EPOCHS, "batch_size": BATCH, "lr": LR, "vae_epochs": VAE_EPOCHS,
        "latent": LATENT, "lookback": LOOKBACK, "horizon": HORIZON,
        "jitter_sigma": JITTER_SIGMA, "scale_sigma": SCALE_SIGMA}
    for col, val in frozen.items():
        if isinstance(val, float):
            valid = np.isclose(df[col].astype(float), val, atol=1e-10, rtol=0)
        else:
            valid = df[col] == val
        if not valid.all():
            raise ValueError(f"Previous rows use different {col}; bump VERSION")
    if (df.rep < 0).any() or (df.aug_seed < 0).any() or not df.method.isin(METHODS).all():
        raise ValueError("Invalid method or seed in previous rows")
    return df


def pending_runs(domain, digest, methods, reps, seeds, previous):
    done = set(previous.loc[:, list(KEY)].itertuples(index=False, name=None))
    return [(method, rep, seed) for method in methods for seed in seeds for rep in reps
            if (VERSION, domain, digest, REGIME, ARCH, INPUT_MODE, method, rep, seed) not in done]


def pool_path(pool_dir: Path, domain: str, digest: str, method: str, seed: int) -> Path:
    return pool_dir / f"{domain}_{digest[:16]}_{method}_seed{seed}_v1.npz"


def load_or_create_pool(pool_dir: Path, domain: str, digest: str, method: str,
                        seed: int, real, previous: pd.DataFrame):
    path = pool_path(pool_dir, domain, digest, method, seed)
    prior = previous[(previous.domain == domain) & (previous.method == method) &
                     (previous.aug_seed == seed)]
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if (str(z["data_sha256"]) != digest or str(z["method"]) != method or
                    int(z["aug_seed"]) != seed or str(z["version"]) != VERSION):
                raise ValueError(f"Unexpected pool metadata: {path}")
            sx, sy, seconds = z["x"], z["y"], float(z["seconds"])
        digest_pool = file_sha(path)
        if len(prior) and not (prior.pool_file_sha256 == digest_pool).all():
            raise ValueError(f"Cached augmentation pool changed: {path}")
    else:
        if len(prior):
            raise ValueError(f"Previously used augmentation pool missing; do not regenerate: {path}")
        start = time.perf_counter()
        sx, sy = augmented_training_pair(real, method, seed)
        seconds = time.perf_counter()-start
        pool_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part.npz")
        np.savez_compressed(tmp, x=sx, y=sy, seconds=seconds, data_sha256=digest,
                            method=method, aug_seed=seed, version=VERSION)
        os.replace(tmp, path)
        digest_pool = file_sha(path)
    if (sx.shape != real[0].shape or sy.shape != real[1].shape
        or not np.isfinite(sx).all() or not np.isfinite(sy).all()):
        raise ValueError("Augmentation pool must contain exactly 1x finite matching windows")
    return (sx, sy), digest_pool, seconds


def add_result(path: Path, previous: pd.DataFrame, row: dict):
    new = pd.DataFrame([row]) if previous.empty else pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    if new.duplicated(list(KEY)).any():
        raise ValueError("Duplicate result key while appending")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return new


def summarize(refs, new: pd.DataFrame, domain: str, methods, reps, seeds,
              matched: int, report_dir: Path):
    index = {(r.method, int(r.rep), int(r.aug_seed)): r for r in
             new[new.domain == domain].itertuples(index=False)}
    paired, incomplete = [], []
    for method in methods:
        for seed in seeds:
            for rep in reps:
                record = index.get((method, rep, seed))
                if record is None:
                    incomplete.append(dict(domain=domain, method=method, rep=rep, aug_seed=seed))
                    continue
                baselines = {"vae": refs[(rep, "vae_augmented", seed)],
                    "step": refs[(rep, "step_baseline", -1)],
                    "duplicate": refs[(rep, "duplicate_real", -1)],
                    "epoch": refs[(rep, "epoch_baseline", -1)]}
                if int(record.optimizer_steps) != matched:
                    raise ValueError("Simple augmentation optimizer step mismatch")
                if not all(np.isclose(float(record.target_sd), float(b["target_sd"]), rtol=1e-6)
                           for b in baselines.values()):
                    raise ValueError("Simple augmentation/reference normalization mismatch")
                row = dict(domain=domain, method=method, rep=rep, aug_seed=seed,
                           optimizer_steps=matched, simple_best_at_end=(int(record.best_step) == matched),
                           vae_best_at_end=(int(baselines["vae"]["best_step"]) == matched))
                for metric in METRICS:
                    a = float(getattr(record, metric))
                    row["simple_"+metric] = a
                    for label, b in baselines.items():
                        v = float(b[metric])
                        row[f"{label}_{metric}"] = v
                        row[f"simple_vs_{label}_{metric}_abs"] = a-v
                        row[f"simple_vs_{label}_{metric}_pct"] = 100*(a/v-1)
                row["simple_beats_both"] = (row["simple_vs_step_test_mse_abs"] < 0 and
                                            row["simple_vs_duplicate_test_mse_abs"] < 0)
                row["simple_beats_vae"] = row["simple_vs_vae_test_mse_abs"] < 0
                paired.append(row)
    paired_df = pd.DataFrame(paired)
    pool_rows, summary_rows = [], []
    if len(paired_df):
        metric_cols = [c for c in paired_df.columns if c.startswith("simple_vs_")]
        for (method, seed), grp in paired_df.groupby(["method", "aug_seed"]):
            row = dict(domain=domain, method=method, aug_seed=seed,
                n_training_reps=int(grp.rep.nunique()), n_pairs=len(grp),
                n_simple_beats_both=int(grp.simple_beats_both.sum()),
                n_simple_beats_vae=int(grp.simple_beats_vae.sum()),
                n_simple_best_at_end=int(grp.simple_best_at_end.sum()))
            for m in metric_cols:
                row["median_"+m] = float(grp[m].median())
            pool_rows.append(row)
        pools_df = pd.DataFrame(pool_rows)
        for method, grp in pools_df.groupby("method"):
            row = dict(domain=domain, method=method, n_pool_seeds=int(grp.aug_seed.nunique()),
                       n_pairs=int(grp.n_pairs.sum()),
                       n_training_reps_min=int(grp.n_training_reps.min()),
                       n_training_reps_max=int(grp.n_training_reps.max()),
                       n_simple_beats_both=int(grp.n_simple_beats_both.sum()),
                       n_simple_beats_vae=int(grp.n_simple_beats_vae.sum()))
            for c in [c for c in grp.columns if c.startswith("median_")]:
                row[c] = float(grp[c].median())
                row["min_"+c] = float(grp[c].min())
                row["max_"+c] = float(grp[c].max())
            summary_rows.append(row)
    else:
        pools_df = pd.DataFrame()
    report_dir.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(report_dir / f"{domain}_paired.csv", index=False)
    pools_df.to_csv(report_dir / f"{domain}_per_pool.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(report_dir / f"{domain}_summary.csv", index=False)
    pd.DataFrame(incomplete, columns=["domain", "method", "rep", "aug_seed"]).to_csv(
        report_dir / f"{domain}_incomplete.csv", index=False)
    if summary_rows:
        print(pd.DataFrame(summary_rows)[["domain", "method", "n_pairs", "n_pool_seeds",
            "n_simple_beats_both", "n_simple_beats_vae",
            "median_simple_vs_vae_test_mse_pct",
            "median_simple_vs_duplicate_test_mse_pct",
            "median_simple_vs_step_test_mse_pct"]].to_string(index=False))
    print(f"{domain}: {len(paired)} complete, {len(incomplete)} missing; no significance claims")
    return paired_df, pools_df, pd.DataFrame(summary_rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=tuple(DOMAIN_FREQ), default="weather")
    p.add_argument("--methods", default="jitter,scaling")
    p.add_argument("--reps", default="0")
    p.add_argument("--aug-seeds", default="0")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="maximum pending new fits (0=all)")
    p.add_argument("--reference", type=Path, default=REF_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/real")
    args = p.parse_args(argv)
    try:
        methods, reps, seeds = parse_methods(args.methods), parse_seeds(args.reps), parse_seeds(args.aug_seeds)
        if args.gpu_index < 0 or args.limit < 0:
            raise ValueError("Invalid GPU index or limit")
        source = args.data_dir / f"{args.domain}.npy"
        raw_csv = args.data_dir / args.domain / f"{args.domain}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        array = np.load(source, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
            raise ValueError("Invalid prepared array")
        t1 = int(len(array)*.7)
        start = int(t1-REGIME*t1)
        n = t1-start-LOOKBACK-HORIZON+1
        if n < 1:
            raise ValueError("Not enough training windows")
        digest = file_sha(source)
        first, last = audit_timestamps(raw_csv, len(array), start, args.domain)
        refs, old, matched = validate_reference(args.reference, args.domain, digest, reps, seeds, n, start, t1)
        previous = load_new(args.out, args.domain, digest, matched)
        pending = pending_runs(args.domain, digest, methods, reps, seeds, previous)
        print(f"{args.domain} 10%, active timestamps {first} to {last}; {n} windows")
        print(f"Phase7 1x refs READ-ONLY: original={old}, fixed={matched} steps")
        print(f"New simple augmentation rows: {len(pending)} pending; {len(methods)*len(reps)*len(seeds)} requested")
        for task in pending:
            print("  ", task)
        if args.dry_run:
            return
        if args.report:
            summarize(refs, previous, args.domain, methods, reps, seeds, matched, args.report_dir)
            return
        if not pending:
            return
        from src.data import WindowConfig, prepare
        from scripts.asoc_budget_controls import fit_condition, select_device
        device = select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
        prepared = prepare(np.asarray(array, dtype=np.float32), regime=REGIME,
                           cfg=WindowConfig(lookback=LOOKBACK, horizon=HORIZON))
        real = (prepared["train"][0][:,:,:1], prepared["train"][1])
        val = (prepared["val"][0][:,:,:1], prepared["val"][1])
        test = (prepared["test"][0][:,:,:1], prepared["test"][1])
        if len(real[0]) != n:
            raise ValueError("Unexpected real train window count")
        target_sd = float(prepared["sd"][0,0])
        if not np.isclose(target_sd, float(refs[(reps[0], "step_baseline", -1)]["target_sd"]), rtol=1e-5):
            raise ValueError("Different scaling from Phase7")
        for method, rep, seed in pending[:args.limit or len(pending)]:
            (sx, sy), pool_hash, seconds = load_or_create_pool(args.pool_dir, args.domain, digest,
                                                        method, seed, real, previous)
            fit = fit_condition(arch=ARCH,
                  train_pair=(np.concatenate([real[0], sx]), np.concatenate([real[1], sy])),
                  val_pair=val, test_pair=test, init_seed=rep, max_steps=matched,
                  validation_interval=math.ceil(2*n/BATCH), batch_size=BATCH, lr=LR,
                  device=device)
            row = dict(experiment_version=VERSION, domain=args.domain, data_sha256=digest,
                regime=REGIME, arch=ARCH, input_mode=INPUT_MODE,
                method=method, rep=rep, aug_seed=seed, epochs=EPOCHS,
                batch_size=BATCH, lr=LR, vae_epochs=VAE_EPOCHS, latent=LATENT,
                hidden=HIDDEN, horizon_weight=HORIZON_WEIGHT,
                jitter_sigma=JITTER_SIGMA, scale_sigma=SCALE_SIGMA,
                lookback=LOOKBACK, horizon=HORIZON,
                original_steps=old, matched_steps=matched, **fit,
                mse_raw=float(fit["test_mse"])*target_sd**2,
                mae_raw=float(fit["test_mae"])*target_sd,
                target_sd=target_sd, train_start=start, train_end=t1,
                real_windows=n, pool_file_sha256=pool_hash,
                augment_seconds=seconds,
                augmentation_scope="target_only_train_window_input_and_label")
            previous = add_result(args.out, previous, row)
            print(f"saved {args.domain} {method} rep={rep} aug_seed={seed} "
                  f"MSE={fit['test_mse']:.6f} steps={fit['optimizer_steps']}")
        print("Complete. --report will compare the new rows with frozen Phase7 1x baselines.")
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
