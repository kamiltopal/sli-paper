"""Phase 7 Weather 10% VAE augmentation-ratio sensitivity, fixed 1x step budget.

This is an ADD-ON: it never edits Phase 7 results, prepared arrays, or Traffic.
The existing 1x results (3 training seeds x 2 VAE pool seeds) are imported
from real_budget_controls.csv and are NOT retrained. New 0.5x/2x arms have
ratio-matched duplicated-real controls and a shared original step baseline.

Pilot (no GPU on --dry-run):
  python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0 --dry-run
  python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0
  python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0 --report

Full pre-declared grid (run after verifying the pilot's protocol, not its sign):
  python -m scripts.asoc_real_ratio_weather --reps 0,1,2 --pool-seeds 0,1
  python -m scripts.asoc_real_ratio_weather --reps 0,1,2 --pool-seeds 0,1 --report
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
EXPERIMENT_VERSION = "phase7_weather_ratio_fixed1120_v1"
REFERENCE_VERSION = "phase7_real_equal_step_v1"
DOMAIN, REGIME, ARCH, INPUT_MODE = "weather", 0.10, "itransformer", "target"
LOOKBACK, HORIZON = 96, 24
EPOCHS, BATCH, LR = 10, 64, 1e-3
VAE_EPOCHS, LATENT, HIDDEN, HORIZON_WEIGHT = 30, 16, 128, 0.5
RATIOS = (0.5, 2.0)
POOL_CONTROL_SEED = -1
REF_DEFAULT = ROOT / "results/asoc_revision/real_budget_controls.csv"
OUT_DEFAULT = ROOT / "results/asoc_revision/ratio_weather_v1.csv"
POOL_DIR_DEFAULT = ROOT / "results/asoc_revision/ratio_weather_pools_v1"
REPORT_DIR_DEFAULT = ROOT / "results/asoc_revision/ratio_weather_report_v1"
DATA_DEFAULT = ROOT / "data/real/weather.npy"
RAW_DEFAULT = ROOT / "data/real/weather/weather.csv"
KEY = ("experiment_version", "data_sha256", "ratio", "rep", "condition", "pool_seed")
REF_ID = ("experiment_version", "domain", "data_sha256", "regime", "arch", "input_mode",
          "rep", "epochs", "batch_size", "lr", "vae_epochs", "latent", "hidden",
          "horizon_weight", "condition", "pool_seed")
METRICS = ("test_mse", "test_mae", "mse_raw", "mae_raw")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        result = tuple(int(i.strip()) for i in raw.split(","))
    except ValueError as exc:
        raise ValueError("Seeds must be comma-separated integers") from exc
    if not result or any(v < 0 for v in result) or len(set(result)) != len(result):
        raise ValueError("Seeds must be distinct and nonnegative")
    return result


def ratio_values(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(x.strip()) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError("Ratios must be 0.5,2.0 (1x already exists)") from exc
    if not values or len(set(values)) != len(values) or not set(values).issubset(RATIOS):
        raise ValueError("Ratios must be a unique subset of 0.5,2.0")
    return values


def make_duplicate(real: tuple[np.ndarray, np.ndarray], ratio: float, seed: int = 2718):
    """Real + ratio*n extra real windows, without artificial target perturbations.

    Each full unit of ratio is one complete real copy; a fractional part is
    a fixed seeded subset sampled without replacement. Same extra pool across
    model seeds, independent of synthetic-pool seed.
    """
    x, y = real
    n = len(x)
    if n < 1 or len(y) != n or ratio <= 0 or not np.isfinite(ratio):
        raise ValueError("Invalid duplicate-real inputs")
    count = int(ratio * n)  # identical rounding convention as src.generators.gen_vae
    whole, remain = divmod(count, n)
    ids = [np.arange(n, dtype=np.int64) for _ in range(whole)]
    if remain:
        ids.append(np.random.default_rng(seed).permutation(n)[:remain])
    chosen = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
    return np.concatenate((x, x[chosen])), np.concatenate((y, y[chosen]))


def basic_schema(df: pd.DataFrame, required, keys, path):
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(set(required) - set(df.columns))}")
    if df.duplicated(list(keys)).any():
        raise ValueError(f"{path}: duplicate experiment IDs")


def load_reference(path: Path, digest: str, reps, pools, n: int, expected_start=None, expected_end=None):
    if not path.is_file():
        raise FileNotFoundError(f"Required existing 1x raw Phase7 log not found: {path}")
    raw = pd.read_csv(path)
    required = set(REF_ID) | set(METRICS) | {"optimizer_steps", "original_steps",
               "matched_steps", "target_sd", "train_start", "train_end", "best_step"}
    basic_schema(raw, required, REF_ID, path)
    expected = dict(experiment_version=REFERENCE_VERSION, domain=DOMAIN,
                    data_sha256=digest, regime=REGIME, arch=ARCH, input_mode=INPUT_MODE,
                    epochs=EPOCHS, batch_size=BATCH, lr=LR,
                    vae_epochs=VAE_EPOCHS, latent=LATENT, hidden=HIDDEN,
                    horizon_weight=HORIZON_WEIGHT)
    filtered = raw.copy()
    for col, val in expected.items():
        if isinstance(val, float):
            filtered = filtered[np.isclose(filtered[col].astype(float), val, rtol=0, atol=1e-10)]
        else:
            filtered = filtered[filtered[col] == val]
    index = {}
    for row in filtered.to_dict("records"):
        index[(int(row["rep"]), str(row["condition"]), int(row["pool_seed"]))] = row
    expected_old, expected_matched = EPOCHS * math.ceil(n / BATCH), EPOCHS * math.ceil(2 * n / BATCH)
    for rep in reps:
        for cond, pool in (("epoch_baseline", -1), ("step_baseline", -1), ("duplicate_real", -1)) + tuple(("vae_augmented", ps) for ps in pools):
            ref_key = (rep, cond, pool)
            if ref_key not in index:
                raise ValueError(f"Missing 1x Phase7 reference {ref_key}; do not rerun it implicitly")
            row = index[ref_key]
            target_steps = expected_old if cond == "epoch_baseline" else expected_matched
            if (int(row["optimizer_steps"]) != target_steps or
                    int(row["matched_steps"]) != expected_matched or
                    int(row["original_steps"]) != expected_old):
                raise ValueError(f"Reference budget mismatch for {ref_key}")
            if not all(np.isfinite(float(row[c])) and float(row[c]) > 0 for c in METRICS + ("target_sd",)):
                raise ValueError(f"Nonfinite or invalid reference metric: {ref_key}")
            if ((expected_start is not None and int(row["train_start"]) != expected_start) or
                    (expected_end is not None and int(row["train_end"]) != expected_end)):
                raise ValueError(f"Phase7 reference split mismatch: {ref_key}")
    return index, expected_old, expected_matched


def check_active_time(raw_csv: Path, length: int, train_start: int):
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Weather source timestamps required: {raw_csv}")
    dates = pd.to_datetime(pd.read_csv(raw_csv, usecols=["date"])["date"], errors="coerce")
    if len(dates) != length:
        raise ValueError("Weather raw CSV and prepared array have different row counts")
    active = dates.iloc[train_start:]
    if (active.isna().any() or not active.is_unique or
            not (active.diff().dropna() == pd.Timedelta(minutes=10)).all()):
        raise ValueError("Weather 10% active train/validation/test timestamps irregular: STOP")
    return str(active.iloc[0]), str(active.iloc[-1])


def load_existing(path: Path, digest: str):
    if not path.exists():
        return pd.DataFrame(columns=KEY)
    df = pd.read_csv(path)
    config = dict(domain=DOMAIN, regime=REGIME, arch=ARCH, input_mode=INPUT_MODE,
                  epochs=EPOCHS, batch_size=BATCH, lr=LR, vae_epochs=VAE_EPOCHS,
                  latent=LATENT, hidden=HIDDEN, horizon_weight=HORIZON_WEIGHT,
                  lookback=LOOKBACK, horizon=HORIZON)
    required = set(KEY) | set(config) | set(METRICS) | {"optimizer_steps", "matched_steps",
                                                  "fixed_steps", "pool_file_sha256"}
    basic_schema(df, required, KEY, path)
    for col, val in config.items():
        if isinstance(val, float):
            matches = np.isclose(df[col].astype(float), val, rtol=0, atol=1e-10)
        else:
            matches = df[col] == val
        if not matches.all():
            raise ValueError(f"Existing ratio log was produced with different {col}")
    if not (df["experiment_version"] == EXPERIMENT_VERSION).all():
        raise ValueError("Unexpected ratio experiment version")
    if not (df["data_sha256"] == digest).all():
        raise ValueError("Weather data digest changed; use a new output and audit provenance")
    if not df["ratio"].isin(RATIOS).all() or not df["condition"].isin(("duplicate_real", "vae_augmented")).all():
        raise ValueError("Unexpected ratio/condition in output CSV")
    if ((df.condition == "duplicate_real") & (df.pool_seed != -1)).any():
        raise ValueError("Duplicate-real control must have pool_seed=-1")
    if ((df.condition == "vae_augmented") & (df.pool_seed < 0)).any():
        raise ValueError("VAE must have nonnegative pool seed")
    if len(df) and (not (df.optimizer_steps == df.matched_steps).all() or
                    not (df.optimizer_steps == df.fixed_steps).all()):
        raise ValueError("Unexpected optimizer step budget in existing ratio output")
    return df


def result_key(ratio, rep, condition, pool_seed, digest):
    return (EXPERIMENT_VERSION, digest, float(ratio), int(rep), str(condition), int(pool_seed))


def planned(reps, pools, ratios, completed, digest):
    done = set(completed.loc[:, KEY].itertuples(index=False, name=None))
    pending = []
    for ratio in ratios:
        for rep in reps:
            key = result_key(ratio, rep, "duplicate_real", -1, digest)
            if key not in done:
                pending.append((ratio, rep, "duplicate_real", -1))
        for ps in pools:
            for rep in reps:
                key = result_key(ratio, rep, "vae_augmented", ps, digest)
                if key not in done:
                    pending.append((ratio, rep, "vae_augmented", ps))
    return pending


def pool_cache_path(pool_dir, ratio, ps, digest):
    return pool_dir / f"weather_{digest[:16]}_r{int(ratio*100):03d}_pool{ps}_v{VAE_EPOCHS}_z{LATENT}.npz"


def get_pool(pool_dir, ratio, ps, digest, segment, cfg, real, previous):
    path = pool_cache_path(pool_dir, ratio, ps, digest)
    done_pool = previous[(previous.ratio == ratio) & (previous.pool_seed == ps) &
                         (previous.condition == "vae_augmented")]
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if (str(z["data_sha256"]) != digest or float(z["ratio"]) != ratio or
                    int(z["pool_seed"]) != ps or int(z["vae_epochs"]) != VAE_EPOCHS or
                    int(z["latent"]) != LATENT):
                raise ValueError(f"Stale/corrupted synthetic pool metadata {path}")
            sx, sy, gen_seconds = z["x"], z["y"], float(z["seconds"])
        file_hash = sha256_file(path)
        if not done_pool.empty and (not (done_pool.pool_file_sha256 == file_hash).all()):
            raise ValueError(f"Pool file changed since earlier model run: {path}")
    else:
        if not done_pool.empty:
            raise ValueError(f"Previously used pool cache missing: {path}; cannot safely regenerate")
        from src.generators import gen_vae
        begin = time.perf_counter()
        synth = gen_vae(segment, cfg, ratio=ratio, seed=ps, epochs=VAE_EPOCHS, latent=LATENT)
        gen_seconds = time.perf_counter() - begin
        sx, sy = synth[0][:, :, :1], synth[1]
        pool_dir.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part.npz")
        np.savez_compressed(temp, x=sx, y=sy, seconds=gen_seconds,
                            data_sha256=digest, ratio=ratio, pool_seed=ps,
                            vae_epochs=VAE_EPOCHS, latent=LATENT)
        os.replace(temp, path)
        file_hash = sha256_file(path)
    if (len(sx) != int(len(real[0]) * ratio) or sx.shape[1:] != real[0].shape[1:] or
            sy.shape != (len(sx), HORIZON) or
            not np.isfinite(sx).all() or not np.isfinite(sy).all()):
        raise ValueError(f"Synthetic pool dimensions/values invalid: {path}")
    return (sx, sy), file_hash, gen_seconds


def atomic_append(path: Path, previous: pd.DataFrame, row: dict):
    new = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    basic_schema(new, KEY, KEY, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    new.to_csv(temp, index=False)
    os.replace(temp, path)
    return new


def report(reference, extra, reps, pools, ratios, steps, report_dir):
    index = {(float(r.ratio), int(r.rep), str(r.condition), int(r.pool_seed)): r
             for r in extra.itertuples(index=False)}
    paired, incomplete = [], []
    for ratio in sorted(set(ratios) | {1.0}):
        for ps in pools:
            for rep in reps:
                old_step = reference[(rep, "step_baseline", -1)]
                old_epoch = reference[(rep, "epoch_baseline", -1)]
                old_vae = reference[(rep, "vae_augmented", ps)]
                if ratio == 1.0:
                    old_dup = reference[(rep, "duplicate_real", -1)]
                    vae, dup = old_vae, old_dup
                else:
                    vae = index.get((ratio, rep, "vae_augmented", ps))
                    dup = index.get((ratio, rep, "duplicate_real", -1))
                if vae is None or dup is None:
                    incomplete.append(dict(ratio=ratio, rep=rep, pool_seed=ps,
                                           missing="vae or ratio-matched duplicate control"))
                    continue
                if int(vae["optimizer_steps"] if isinstance(vae, dict) else vae.optimizer_steps) != steps:
                    raise ValueError("VAE report row has wrong budget")
                if int(dup["optimizer_steps"] if isinstance(dup, dict) else dup.optimizer_steps) != steps:
                    raise ValueError("Duplicated-real report row has wrong budget")
                vals = lambda row, col: float(row[col] if isinstance(row, dict) else getattr(row, col))
                if any(not np.isclose(vals(vae, "target_sd"), vals(other, "target_sd"), rtol=1e-6)
                       for other in (dup, old_step)):
                    raise ValueError("Mismatched normalization in ratio comparison")
                entry = dict(ratio=ratio, rep=rep, pool_seed=ps,
                             optimizer_steps=steps, source=("phase7_existing_1x" if ratio == 1.0 else "ratio_new"),
                             vae_best_at_end=(int(vae["best_step"] if isinstance(vae, dict) else vae.best_step) == steps))
                for metric in METRICS:
                    v = vals(vae, metric)
                    entry[f"vae_{metric}"] = v
                    for name, row in (("duplicate_real", dup), ("step_baseline", old_step),
                                      ("epoch_baseline", old_epoch), ("one_x_vae", old_vae)):
                        b = vals(row, metric)
                        entry[f"{name}_{metric}"] = b
                        entry[f"vae_vs_{name}_{metric}_abs"] = v - b
                        entry[f"vae_vs_{name}_{metric}_pct"] = (v/b - 1)*100
                entry["beats_both_matched"] = (
                    entry["vae_vs_duplicate_real_test_mse_abs"] < 0 and
                    entry["vae_vs_step_baseline_test_mse_abs"] < 0)
                paired.append(entry)
    if not paired:
        raise ValueError("No complete ratio pairs")
    paired_frame = pd.DataFrame(paired)
    per_pool = []
    measure_cols = [c for c in paired_frame.columns if c.startswith("vae_vs_")]
    for (ratio, ps), group in paired_frame.groupby(["ratio", "pool_seed"], sort=True):
        row = dict(ratio=ratio, pool_seed=ps, n_training_reps=len(group),
                   beats_both_count=int(group.beats_both_matched.sum()),
                   vae_checkpoint_at_end_count=int(group.vae_best_at_end.sum()))
        for col in measure_cols:
            row["median_" + col] = group[col].median()
        per_pool.append(row)
    per_pool_frame = pd.DataFrame(per_pool)
    summaries = []
    for ratio, group in per_pool_frame.groupby("ratio", sort=True):
        entry = dict(ratio=ratio, n_pool_seeds=len(group),
                     training_reps_min=int(group.n_training_reps.min()),
                     training_reps_max=int(group.n_training_reps.max()),
                     n_pairs=int(group.n_training_reps.sum()),
                     beats_both_n_pairs=int(group.beats_both_count.sum()))
        for col in (c for c in per_pool_frame.columns if c.startswith("median_")):
            entry[col] = group[col].median()
            entry["min_" + col] = group[col].min()
            entry["max_" + col] = group[col].max()
        summaries.append(entry)
    summary_frame = pd.DataFrame(summaries)
    report_dir.mkdir(parents=True, exist_ok=True)
    paired_frame.to_csv(report_dir / "ratio_paired.csv", index=False)
    per_pool_frame.to_csv(report_dir / "ratio_per_pool.csv", index=False)
    summary_frame.to_csv(report_dir / "ratio_summary.csv", index=False)
    pd.DataFrame(incomplete, columns=["ratio", "rep", "pool_seed", "missing"]).to_csv(
        report_dir / "ratio_incomplete.csv", index=False)
    print(summary_frame[["ratio", "n_pairs", "n_pool_seeds", "beats_both_n_pairs",
                         "median_vae_vs_duplicate_real_test_mse_pct",
                         "median_vae_vs_step_baseline_test_mse_pct",
                         "median_vae_vs_one_x_vae_test_mse_pct"]].to_string(index=False))
    print(f"Incomplete ratio pairs: {len(incomplete)}; summaries are descriptive, NOT independent-domain CIs")
    return paired_frame, per_pool_frame, summary_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", default="0", help="model initialization seeds")
    parser.add_argument("--pool-seeds", default="0", help="independent VAE pool seeds")
    parser.add_argument("--ratios", default="0.5,2.0", help="new ratios; existing 1x is automatically imported")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--reference", type=Path, default=REF_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR_DEFAULT)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--raw-csv", type=Path, default=RAW_DEFAULT)
    args = parser.parse_args(argv)
    try:
        reps, pools, ratios = parse_seeds(args.reps), parse_seeds(args.pool_seeds), ratio_values(args.ratios)
        if args.gpu_index < 0:
            raise ValueError("GPU index must be nonnegative")
        if not args.data.is_file():
            raise FileNotFoundError(args.data)
        array = np.load(args.data, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
            raise ValueError("Weather prepared array must be finite and 2D")
        train_end = int(len(array) * 0.7)
        start = int(train_end - REGIME * train_end)
        n = train_end - start - LOOKBACK - HORIZON + 1
        if n < 1:
            raise ValueError("Insufficient scarce train windows")
        digest = sha256_file(args.data)
        timestamp_first, timestamp_last = check_active_time(args.raw_csv, len(array), start)
        reference, old_steps, fixed_steps = load_reference(
            args.reference, digest, reps, pools, n, expected_start=start, expected_end=train_end)
        previous = load_existing(args.out, digest)
        pending = planned(reps, pools, ratios, previous, digest)
        print(f"Weather 10% active time: {timestamp_first} to {timestamp_last}")
        print(f"1x existing: {len(reps)} reps x {len(pools)} pools; ",
              f"old steps={old_steps}, fixed steps={fixed_steps}, windows={n}")
        print(f"New ratio runs: {len(pending)} pending / "
              f"{len(ratios)*len(reps)*(1+len(pools))} planned; original 1x is read-only")
        for ratio in ratios:
            subset = [p for p in pending if p[0] == ratio]
            print(f" ratio {ratio:g}x: {len(subset)} pending; synthetic windows={int(ratio*n)}")
        if args.report:
            report(reference, previous, reps, pools, ratios, fixed_steps, args.report_dir)
            return
        if args.dry_run or not pending:
            return
        from scripts.asoc_budget_controls import fit_condition, select_device
        from scripts.asoc_real_controls import sha256 as _compatible_sha  # verifies original import compatibility
        from src.data import WindowConfig, prepare, chrono_split, apply_regime, standardize
        if _compatible_sha(args.data) != digest:
            raise ValueError("Inconsistent data digest")
        device = select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
        X = np.asarray(array, dtype=np.float32)
        cfg = WindowConfig(lookback=LOOKBACK, horizon=HORIZON)
        ready = prepare(X, regime=REGIME, cfg=cfg)
        real = (ready["train"][0][:, :, :1], ready["train"][1])
        val = (ready["val"][0][:, :, :1], ready["val"][1])
        test = (ready["test"][0][:, :, :1], ready["test"][1])
        if len(real[0]) != n:
            raise ValueError("Training window count differs from Phase7 reference")
        std = float(ready["sd"][0, 0])
        if not np.isclose(std, float(reference[(reps[0], "step_baseline", -1)]["target_sd"]), rtol=1e-5):
            raise ValueError("Training normalization differs from Phase7")
        norm, _, _ = standardize(X, slice(start, train_end))
        segment = np.asarray(norm[start:train_end], dtype=np.float32)
        validate_interval = math.ceil(2*n/BATCH)  # exactly the 1x Phase7 matched cadence

        def train_and_store(ratio, rep, condition, ps, train_pair, pool_sha="", seconds=float("nan")):
            nonlocal previous
            key = result_key(ratio, rep, condition, ps, digest)
            done = set(previous.loc[:, KEY].itertuples(index=False, name=None))
            if key in done:
                return
            metric = fit_condition(arch=ARCH, train_pair=train_pair, val_pair=val,
                                   test_pair=test, init_seed=rep, max_steps=fixed_steps,
                                   validation_interval=validate_interval, batch_size=BATCH,
                                   lr=LR, device=device)
            row = dict(experiment_version=EXPERIMENT_VERSION, domain=DOMAIN,
                       data_sha256=digest, regime=REGIME, arch=ARCH, input_mode=INPUT_MODE,
                       ratio=ratio, rep=rep, condition=condition, pool_seed=ps,
                       epochs=EPOCHS, batch_size=BATCH, lr=LR,
                       vae_epochs=VAE_EPOCHS, latent=LATENT, hidden=HIDDEN,
                       horizon_weight=HORIZON_WEIGHT, lookback=LOOKBACK, horizon=HORIZON,
                       fixed_steps=fixed_steps, matched_steps=fixed_steps,
                       validation_interval=validate_interval,
                       synthetic_windows=int(ratio*n), real_windows=n,
                       train_start=start, train_end=train_end,
                       target_sd=std, mse_raw=metric["test_mse"]*std**2,
                       mae_raw=metric["test_mae"]*std,
                       generator_seconds=seconds, pool_file_sha256=pool_sha,
                       **metric)
            if (not np.isfinite(metric["test_mse"]) or not np.isfinite(metric["test_mae"]) or
                    metric["test_mse"] <= 0 or metric["test_mae"] <= 0 or
                    int(metric["optimizer_steps"]) != fixed_steps):
                raise ValueError("Nonfinite/invalid metric or unexpected training budget; row not saved")
            previous = atomic_append(args.out, previous, row)
            print(f"{ratio:g}x rep={rep} pool={ps} {condition}: MSE={metric['test_mse']:.6f}, "
                  f"steps={metric['optimizer_steps']}", flush=True)

        for ratio in ratios:
            dup = make_duplicate(real, ratio)
            for rep in reps:
                train_and_store(ratio, rep, "duplicate_real", POOL_CONTROL_SEED, dup)
            for ps in pools:
                missing_reps = [rep for rep in reps if result_key(ratio, rep, "vae_augmented", ps, digest)
                                not in set(previous.loc[:, KEY].itertuples(index=False, name=None))]
                if not missing_reps:
                    continue
                synth, file_sha, seconds = get_pool(args.pool_dir, ratio, ps, digest,
                                                     segment, cfg, real, previous)
                x, y = np.concatenate((real[0], synth[0])), np.concatenate((real[1], synth[1]))
                for rep in missing_reps:
                    train_and_store(ratio, rep, "vae_augmented", ps, (x, y), file_sha, seconds)
        print("Done. Run again with --report; existing rows/pools are preserved.")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
