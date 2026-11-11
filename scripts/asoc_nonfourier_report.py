"""Phase 12: strict pair audit and nested aggregation: rep -> pool -> series -> family.

The independent units at the family level are the three simulated time series,
NOT model seeds, VAE pool seeds, training windows or test timestamps.
No inferential p-values or confidence intervals are fabricated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_nonfourier import (ROOT, BASE, OUT, KEY, CONTROLS,
                                   VERSION, read_runs)

BASEKEY = tuple(k for k in KEY if k not in ("condition", "pool_seed"))
METRICS = ("test_mse", "test_mae", "mse_raw", "mae_raw")


def audit_and_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(KEY) | set(METRICS) | {"optimizer_steps", "matched_steps", "original_steps",
        "best_step", "real_windows", "train_start", "train_end", "target_sd",
        "persistence_mse", "trainmean_mse", "direct_ar2_mse"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if df.duplicated(list(KEY)).any():
        raise ValueError("Duplicate run keys")
    if not (df.experiment_version == VERSION).all():
        raise ValueError("Mixed experiment versions")
    if not np.isfinite(df[list(METRICS) + ["target_sd", "optimizer_steps",
                     "persistence_mse", "trainmean_mse", "direct_ar2_mse"]]
                    .to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite losses or diagnostic metrics")
    if (df[list(METRICS) + ["target_sd"]] <= 0).any().any():
        raise ValueError("Non-positive losses or scaling factor")
    for metric, base, exponent in (("mse_raw", "test_mse", 2),
                                   ("mae_raw", "test_mae", 1)):
        if not np.allclose(df[metric], df[base] * df.target_sd.pow(exponent),
                           rtol=1e-5, atol=1e-6):
            raise ValueError("Incorrect original-scale conversion")
    controls = {}
    for _, r in df[df.condition.isin(CONTROLS)].iterrows():
        controls[tuple(r[k] for k in BASEKEY) + (r.condition,)] = r
    rows, incomplete = [], []
    for _, vae in df[df.condition == "vae_augmented"].iterrows():
        common = tuple(vae[k] for k in BASEKEY)
        missing = [c for c in CONTROLS if common + (c,) not in controls]
        if missing:
            incomplete.append({**{k: vae[k] for k in BASEKEY}, "pool_seed": int(vae.pool_seed),
                               "missing": ",".join(missing)})
            continue
        refs = {c: controls[common + (c,)] for c in CONTROLS}
        for ref in refs.values():
            if any(not np.isclose(float(ref[k]), float(vae[k]), rtol=0, atol=1e-8)
                   for k in ("target_sd", "real_windows", "train_start", "train_end",
                             "persistence_mse", "trainmean_mse", "direct_ar2_mse")):
                raise ValueError("Inconsistent partition, reference or scale within pair")
        if not (int(vae.optimizer_steps) == int(vae.matched_steps) and
                all(int(refs[c].optimizer_steps) == int(vae.matched_steps)
                    for c in ("step_baseline", "duplicate_real")) and
                int(refs["epoch_baseline"].optimizer_steps) == int(vae.original_steps)):
            raise ValueError("Unmatched optimizer-step budgets")
        entry = {k: vae[k] for k in BASEKEY}
        entry.update(pool_seed=int(vae.pool_seed),
                     vae_best_at_last=bool(int(vae.best_step) == int(vae.optimizer_steps)),
                     n_real_windows=int(vae.real_windows),
                     persistence_mse=float(vae.persistence_mse),
                     trainmean_mse=float(vae.trainmean_mse),
                     direct_ar2_mse=float(vae.direct_ar2_mse),
                     gen_seconds=float(vae.generator_seconds),
                     vae_mse_vs_ar2_pct=100 * (float(vae.test_mse) / float(vae.direct_ar2_mse)-1))
        for metric in METRICS:
            entry[f"vae_{metric}"] = float(vae[metric])
            for c, ref in refs.items():
                entry[f"{c}_{metric}"] = float(ref[metric])
                entry[f"vae_vs_{c}_{metric}_pct"] = 100*(float(vae[metric])/float(ref[metric])-1)
                entry[f"vae_vs_{c}_{metric}_abs"] = float(vae[metric])-float(ref[metric])
        entry["vae_beats_both"] = (entry["vae_vs_step_baseline_test_mse_pct"] < 0 and
                                   entry["vae_vs_duplicate_real_test_mse_pct"] < 0)
        rows.append(entry)
    return pd.DataFrame(rows), pd.DataFrame(incomplete, columns=list(BASEKEY) + ["pool_seed", "missing"])


def aggregate(paired: pd.DataFrame):
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # First training rep within pool; then pool within one independent series.
    by_pool = [k for k in BASEKEY if k != "rep"] + ["pool_seed"]
    diff = [c for c in paired if c.startswith("vae_vs_") or c == "vae_mse_vs_ar2_pct"]
    pool_rows = []
    for values, sub in paired.groupby(by_pool, dropna=False):
        row = dict(zip(by_pool, values))
        row.update(n_model_reps=sub.rep.nunique(), n_pairs=len(sub),
                   frac_reps_vae_beats_both=float(sub.vae_beats_both.mean()),
                   frac_reps_last_checkpoint=float(sub.vae_best_at_last.mean()))
        row.update({"median_"+c: float(sub[c].median()) for c in diff})
        pool_rows.append(row)
    pools = pd.DataFrame(pool_rows)
    by_series = [k for k in by_pool if k != "pool_seed"]
    diff_pool = [c for c in pools if c.startswith("median_vae_")]
    series_rows = []
    for values, sub in pools.groupby(by_series, dropna=False):
        row = dict(zip(by_series, values))
        row.update(n_pool_seeds=sub.pool_seed.nunique(),
                   n_model_reps_min=int(sub.n_model_reps.min()),
                   n_pairs=int(sub.n_pairs.sum()),
                   frac_pools_median_beats_both=float((
                       (sub.median_vae_vs_step_baseline_test_mse_pct < 0) &
                       (sub.median_vae_vs_duplicate_real_test_mse_pct < 0)).mean()))
        row.update({c: float(sub[c].median()) for c in diff_pool})
        series_rows.append(row)
    series = pd.DataFrame(series_rows)
    by_family = [k for k in by_series if k not in ("series_id", "data_sha256")]
    group_rows = []
    for values, sub in series.groupby(by_family, dropna=False):
        row = dict(zip(by_family, values))
        row.update(n_independent_series=sub.series_id.nunique(),
                   n_pairs=int(sub.n_pairs.sum()),
                   n_pool_seeds_min=int(sub.n_pool_seeds.min()),
                   n_model_reps_min=int(sub.n_model_reps_min.min()),
                   frac_series_both_median_better=float((
                       (sub.median_vae_vs_step_baseline_test_mse_pct < 0) &
                       (sub.median_vae_vs_duplicate_real_test_mse_pct < 0)).mean()))
        row.update({c: float(sub[c].median()) for c in diff_pool})
        row.update({"min_"+c: float(sub[c].min()) for c in diff_pool})
        row.update({"max_"+c: float(sub[c].max()) for c in diff_pool})
        group_rows.append(row)
    return pools, series, pd.DataFrame(group_rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="source", type=Path, default=OUT)
    p.add_argument("--outdir", type=Path, default=BASE / "report")
    args = p.parse_args(argv)
    src = args.source if args.source.is_absolute() else ROOT / args.source
    dest = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    if not src.exists():
        p.error(f"No run CSV: {src}")
    _, data = read_runs(src)
    paired, incomplete = audit_and_pair(data)
    pools, series, summary = aggregate(paired)
    dest.mkdir(parents=True, exist_ok=True)
    for name, table in (("nonfourier_paired.csv", paired),
                        ("nonfourier_incomplete.csv", incomplete),
                        ("nonfourier_per_pool.csv", pools),
                        ("nonfourier_per_series.csv", series),
                        ("nonfourier_summary.csv", summary)):
        table.to_csv(dest / name, index=False)
    print(f"Complete VAE pairs: {len(paired)} | missing control pairs: {len(incomplete)}")
    if not summary.empty:
        cols = ["family", "regime", "arch", "n_independent_series", "n_pairs",
                "median_vae_vs_duplicate_real_test_mse_pct",
                "median_vae_vs_step_baseline_test_mse_pct",
                "frac_series_both_median_better"]
        print(summary[cols].to_string(index=False, float_format=lambda z: f"{z:.5g}"))
    print("Descriptive nested medians, not significance tests or confidence intervals.")


if __name__ == "__main__":
    main()
