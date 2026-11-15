"""Phase 13 (R1.7): TRAIN-ONLY VAE window fidelity; no forecaster retraining.

Reconstructs VAE pools from logged seeds/configuration; original synthetic
windows were not persisted in Phases 7/12. Treat fidelity as a reconstructed
replicate, not a verified metric of the exact previously trained pool.

Usage:
  python -m scripts.asoc_fidelity --source nonfourier --dry-run
  python -m scripts.asoc_fidelity --source nonfourier --limit 1 --gpu-index 0
  python -m scripts.asoc_fidelity --source nonfourier --gpu-index 0
  python -m scripts.asoc_fidelity --source real --domains weather,traffic,electricity,epias --regimes 0.10 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.data import WindowConfig, apply_regime, chrono_split, prepare, standardize

ROOT = Path(__file__).resolve().parents[1]
VERSION = "phase13_train_fidelity_reconstructed_v1"
DEFAULT_OUT = Path("results/asoc_revision/phase13_fidelity")
SOURCES = ("nonfourier", "real")
DEFAULT_INPUT = {
    "nonfourier": Path("results/asoc_revision/nonfourier/nonfourier_runs.csv"),
    "real": Path("results/asoc_revision/real_budget_controls.csv"),
}
GROUP_FIELDS = {
    "nonfourier": ("family", "series_id", "regime", "arch", "pool_seed", "data_sha256",
                   "vae_epochs", "latent", "lookback", "horizon"),
    "real": ("domain", "regime", "arch", "input_mode", "pool_seed", "data_sha256",
             "vae_epochs", "latent", "lookback", "horizon"),
}
METRICS = ("marginal_w1", "sliced_w1", "spectral_js", "spectral_tv",
           "acf_mae", "tail_p2_absdiff")


def parse_csv(raw: str):
    vals = tuple(v.strip() for v in raw.split(","))
    if not vals or "" in vals or len(vals) != len(set(vals)):
        raise ValueError("Expected unique nonempty comma-separated values")
    return vals


def parse_regimes(raw: str):
    vals = tuple(float(x) for x in parse_csv(raw))
    if len(set(vals)) != len(vals) or any(v not in (0.1, 0.25, 1.0) for v in vals):
        raise ValueError("Use distinct regimes from 0.10,0.25,1.00")
    return vals


def digest_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def digest_pair(x: np.ndarray, y: np.ndarray):
    h = hashlib.sha256()
    for v in (x, y):
        a = np.ascontiguousarray(v, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def target_windows(pair):
    x, y = pair
    if x.ndim != 3 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Expected aligned (N,L,C) and (N,H) windows")
    a = np.concatenate((x[:, :, 0], y), axis=1).astype(np.float64, copy=False)
    if not np.isfinite(a).all() or a.shape[1] < 3:
        raise ValueError("Nonfinite or short target windows")
    return a


def _mean_spectrum(a):
    x = a - a.mean(axis=1, keepdims=True)
    psd = np.square(np.abs(np.fft.rfft(x, axis=1)))
    p = psd.mean(axis=0)
    p = np.maximum(p, 1e-15)
    return p / p.sum()


def _mean_acf(a, lags):
    x = a - a.mean(axis=1, keepdims=True)
    power = np.square(x).sum(axis=1).mean()
    if power < 1e-15:
        return np.zeros(len(lags), dtype=float)
    return np.array([(x[:, :-k] * x[:, k:]).sum(axis=1).mean() /
                     (len(x[0]) - k) * len(x[0]) / power for k in lags])


def compare_windows(a: np.ndarray, b: np.ndarray, directions: np.ndarray,
                    lags=(1, 2, 6, 12, 24)) -> dict[str, float]:
    """Fixed-sample distances on standardized target windows; no labels used."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 4 or a.shape[1] < 3:
        raise ValueError("Windows must have matching (N,W), N>=4")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite windows")
    if any(k <= 0 or k >= a.shape[1] for k in lags):
        raise ValueError("Autocorrelation lags must be in [1,W-1]")
    if directions.ndim != 2 or directions.shape[1] != a.shape[1]:
        raise ValueError("Directions incompatible with window length")
    norm = np.linalg.norm(directions, axis=1)
    if np.any(norm <= 0):
        raise ValueError("Zero projection direction")
    directions = directions / norm[:, None]
    # Identical sample sizes, so sorted one-dimensional projections yield W1.
    projected_a = np.sort(a @ directions.T, axis=0)
    projected_b = np.sort(b @ directions.T, axis=0)
    sliced_w1 = float(np.abs(projected_a - projected_b).mean())
    marginal_w1 = float(wasserstein_distance(a.ravel(), b.ravel()))
    pa, pb = _mean_spectrum(a), _mean_spectrum(b)
    tv = float(0.5 * np.abs(pa - pb).sum())
    mid = 0.5 * (pa + pb)
    js = float(max(0.0, 0.5 * (np.sum(pa * np.log2(pa / mid)) +
                              np.sum(pb * np.log2(pb / mid)))))  # JS divergence in bits
    acf = float(np.abs(_mean_acf(a, lags) - _mean_acf(b, lags)).mean())
    tails = float(abs(np.mean(np.abs(a) > 2) - np.mean(np.abs(b) > 2)))
    return dict(marginal_w1=marginal_w1, sliced_w1=sliced_w1,
                spectral_js=js, spectral_tv=tv,
                acf_mae=acf, tail_p2_absdiff=tails)


def diagnostic_metrics(real_pair, synthetic_pair, *, sample_windows=512,
                       projections=32, seed=13071, lags=(1, 2, 6, 12, 24)):
    """RS fidelity, RR finite-sampling reference, shuffled-order negative control.

    Two disjoint selections of train windows form the RR comparator, but these
    sliding windows overlap in time and must NEVER be called independent.
    """
    if sample_windows < 4 or projections < 1:
        raise ValueError("Sample size >=4 and projections >=1 required")
    a = target_windows(real_pair)
    s = target_windows(synthetic_pair)
    if a.shape != s.shape:
        raise ValueError("Expected 1:1 VAE synthetic pool, matching real windows")
    n = min(sample_windows, len(a) // 2, len(s))
    if n < 4:
        raise ValueError("Insufficient windows for two reference subsets")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(a))
    first, second = a[idx[:n]], a[idx[n:2*n]]
    synth = s[rng.choice(len(s), size=n, replace=False)]
    directions = rng.normal(size=(projections, a.shape[1]))
    # Negative control: destroy temporal order but preserve all marginal values.
    shuffled = first[np.arange(n)[:, None], np.argsort(rng.random(first.shape), axis=1)]
    out = {"sample_windows_per_group": n, "projections": projections,
           "diagnostic_seed": seed, "acf_lags": ",".join(map(str, lags))}
    for prefix, pair in (("vae", (first, synth)),
                         ("real_real", (first, second)),
                         ("shuffled", (first, shuffled))):
        out.update({f"{prefix}_{k}": v for k, v in
                    compare_windows(*pair, directions, lags).items()})
    for m in METRICS:
        out[f"vae_minus_real_real_{m}"] = out[f"vae_{m}"] - out[f"real_real_{m}"]
    return out


def select_rows(df, source: str, *, families, ids, domains, regimes):
    if source not in SOURCES:
        raise ValueError("Unrecognized source")
    required = set(GROUP_FIELDS[source]) | {"condition", "rep", "test_mse", "train_start",
             "train_end", "real_windows", "target_sd"}
    if missing := required - set(df.columns):
        raise ValueError(f"Run CSV lacks required columns: {sorted(missing)}")
    if not np.isfinite(df["test_mse"]).all():
        raise ValueError("Nonfinite test MSE")
    x = df[(df.condition == "vae_augmented") &
           (df.regime.map(lambda q: any(math.isclose(float(q), r, abs_tol=1e-8)
                                         for r in regimes)))].copy()
    if source == "nonfourier":
        x = x[x.family.isin(families) & x.series_id.isin(ids)]
    else:
        if "input_mode" not in x or (x.input_mode != "target").any():
            x = x[x.input_mode == "target"]
        x = x[x.domain.isin(domains)]
    if x.empty:
        raise ValueError("No matching VAE runs in source CSV")
    if (x.pool_seed < 0).any():
        raise ValueError("Invalid VAE pool seed")
    keys = list(GROUP_FIELDS[source])
    groups = []
    for _, part in x.groupby(keys, dropna=False, sort=True):
        row = part.iloc[0].to_dict()
        for k in ("train_start", "train_end", "real_windows", "target_sd"):
            if not np.allclose(part[k].to_numpy(dtype=float), float(row[k]), rtol=1e-7, atol=1e-6):
                raise ValueError(f"Conflicting {k} within pool group")
        groups.append(row)
    return groups


def load_training_group(row, source, data_dir):
    cfg = WindowConfig(int(row["lookback"]), int(row["horizon"]))
    if source == "nonfourier":
        from scripts.asoc_nonfourier import ensure_series
        arr, digest = ensure_series(data_dir, str(row["family"]),
                                    int(row["series_id"]), write=False)
    else:
        path = data_dir / f"{row['domain']}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared real domain: {path}")
        digest = digest_file(path)
        arr = np.load(path, allow_pickle=False)
    if str(digest) != str(row["data_sha256"]):
        raise ValueError("Data SHA mismatch; refuse analysis of nonidentical training data")
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Invalid input data")
    prepared = prepare(arr, regime=float(row["regime"]), cfg=cfg)
    end, _ = chrono_split(len(arr))
    start = apply_regime(end, float(row["regime"]), cfg.lookback)
    if int(row["train_start"]) != start or int(row["train_end"]) != end or int(row["real_windows"]) != len(prepared["train"][0]):
        raise ValueError("Training partition/window count changed")
    if not np.isclose(float(row["target_sd"]), float(prepared["sd"][0, 0]), rtol=2e-5, atol=1e-6):
        raise ValueError("Training normalization changed")
    standardized, _, _ = standardize(arr, slice(start, end))
    segment = np.asarray(standardized[start:end], dtype=np.float32)
    return prepared["train"], segment, cfg


def read_previous(path: Path, source: str):
    if not path.exists():
        return [], pd.DataFrame()
    df = pd.read_csv(path)
    fields = ["source", "experiment_version", *GROUP_FIELDS[source], "diagnostic_version",
              "sample_windows", "projections"]
    if missing := set(fields) - set(df.columns):
        raise ValueError(f"Existing diagnostic CSV incompatible: missing {sorted(missing)}")
    if df.duplicated(fields).any():
        raise ValueError("Duplicate diagnostic rows")
    if not (df.diagnostic_version == VERSION).all():
        raise ValueError("Mixed diagnostic versions")
    return fields, df


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=SOURCES, default="nonfourier")
    p.add_argument("--in", dest="input", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--families", default="nonlinear_ar,markov_ar")
    p.add_argument("--ids", default="0,1,2")
    p.add_argument("--domains", default="weather,traffic,electricity,epias")
    p.add_argument("--regimes", default="0.10,1.00")
    p.add_argument("--sample-windows", type=int, default=512)
    p.add_argument("--projections", type=int, default=32)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        families, domains = parse_csv(args.families), parse_csv(args.domains)
        ids = tuple(int(x) for x in parse_csv(args.ids))
        regimes = parse_regimes(args.regimes)
        if args.sample_windows < 4 or args.projections < 1 or args.limit < 0 or args.gpu_index < 0:
            raise ValueError("Invalid sampling, projection, limit or GPU argument")
        src = args.input or DEFAULT_INPUT[args.source]
        src = src if src.is_absolute() else ROOT / src
        if not src.is_file():
            raise FileNotFoundError(f"Missing experiment CSV: {src}")
        outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
        target = outdir / f"{args.source}_fidelity.csv"
        data_dir = args.data_dir or (Path("results/asoc_revision/nonfourier/data")
                                     if args.source == "nonfourier" else Path("data/real"))
        data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
        raw = pd.read_csv(src)
        groups = select_rows(raw, args.source, families=families, ids=ids,
                             domains=domains, regimes=regimes)
        fields, previous = read_previous(target, args.source)
        generator_sha = digest_file(ROOT / "src" / "generators.py")
        if len(previous):
            if not ((previous.sample_windows == args.sample_windows) &
                    (previous.projections == args.projections)).all():
                raise ValueError("Diagnostic sampling settings changed; use a NEW --outdir")
            if "generator_python_sha256" not in previous or not (
                    previous.generator_python_sha256 == generator_sha).all():
                raise ValueError("Generator code hash differs from earlier diagnostics; "
                                 "use a NEW --outdir and audit provenance")
        if not fields:
            fields = ["source", "experiment_version", *GROUP_FIELDS[args.source],
                      "diagnostic_version", "sample_windows", "projections"]
        existing = set(previous[fields].itertuples(index=False, name=None)) if len(previous) else set()
        def key(row):
            data = {"source": args.source, "experiment_version": row["experiment_version"],
                    **row, "diagnostic_version": VERSION,
                    "sample_windows": args.sample_windows, "projections": args.projections}
            return tuple(data[k] for k in fields)
        missing = [row for row in groups if key(row) not in existing]
        # One run/series per VAE pool, shared by every forecaster rep.
        print(f"PLAN source={args.source}: {len(groups)} pools, {len(missing)} to reconstruct; "
              f"forecasters to retrain=0; out={target}", flush=True)
        for row in missing[:15]:
            label = (f"{row['family']}/{int(row['series_id'])}" if args.source == "nonfourier"
                     else row["domain"])
            print(f"  {label} regime={float(row['regime']):.2f} pool={int(row['pool_seed'])} "
                  f"arch={row['arch']}", flush=True)
        if args.dry_run or not missing:
            return
        # Only now initialize Torch/CUDA: dry-run performs no generation or writes.
        from scripts.asoc_budget_controls import select_device
        from src.generators import gen_vae
        import torch
        device = select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
        print(f"Regenerating VAE pools on {device}; these are RECONSTRUCTED replicas, "
              "not verified exact original training pools.", flush=True)
        outdir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(missing[:args.limit or None], 1):
            real, segment, cfg = load_training_group(row, args.source, data_dir)
            synth = gen_vae(segment, cfg, ratio=1.0, seed=int(row["pool_seed"]),
                            epochs=int(row["vae_epochs"]), latent=int(row["latent"]))
            if target_windows(real).shape != target_windows(synth).shape:
                raise ValueError("Regenerated VAE pool has wrong size")
            # Deterministic diagnostic sampling, unrelated to model seeds.
            diag_seed = int(hashlib.sha256((str(row["data_sha256"]) + ":" +
                        str(row["pool_seed"])).encode()).hexdigest()[:8], 16)
            metrics = diagnostic_metrics(real, synth, sample_windows=args.sample_windows,
                    projections=args.projections, seed=diag_seed)
            record = {"source": args.source, "diagnostic_version": VERSION,
                      **{k: row[k] for k in GROUP_FIELDS[args.source]},
                      "experiment_version": row["experiment_version"],
                      "sample_windows": args.sample_windows, "projections": args.projections,
                      "original_vae_pool_persisted": False,
                      "generator_python_sha256": generator_sha,
                      "reconstructed_synthetic_sha256": digest_pair(*synth),
                      "torch_version_reconstruction": str(torch.__version__),
                      "device_reconstruction": str(device),
                      "n_real_windows": len(real[0]), "n_synth_windows": len(synth[0]),
                      **metrics}
            pd.DataFrame([record]).to_csv(target, index=False, mode="a",
                                          header=not target.exists())
            print(f"  [{i}/{len(missing)}] saved {record['reconstructed_synthetic_sha256'][:12]} "
                  f"W1={record['vae_marginal_w1']:.4g} SWD={record['vae_sliced_w1']:.4g} "
                  f"spectral_JS={record['vae_spectral_js']:.4g}", flush=True)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    run()
