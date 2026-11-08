"""Phase 7 paired MSE/MAE and pool-seed report, with no invented uncertainty.

 python -m scripts.asoc_real_report
 python -m scripts.asoc_real_report --in results/asoc_revision/real_budget_controls.csv

Baseline controls appear once per model seed; VAE samples from independent
pool seeds. Each VAE is compared to the SAME baseline repetition; reporting
first aggregates training repetitions within a pool, then across pool seeds.
The two levels are not treated as independent training observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_real_controls import (CONDITIONS, CONTROL_CONDITIONS,
                                       DEFAULT_OUT, KEY, ROOT, VERSION,
                                       read_completed)

BASEKEY = [x for x in KEY if x not in ("condition", "pool_seed")]
METRICS = ("test_mse", "test_mae", "mse_raw", "mae_raw")
COMPARATORS = ("epoch_baseline", "step_baseline", "duplicate_real")


def paired_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(KEY) | set(METRICS) | {"optimizer_steps", "matched_steps",
                                       "original_steps", "best_step", "target_sd",
                                       "real_windows", "train_start", "train_end"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if raw.duplicated(list(KEY)).any():
        raise ValueError("Duplicate experiment rows")
    if set(raw.condition) - set(CONDITIONS):
        raise ValueError("Unknown condition")
    if not np.isfinite(raw[[*METRICS, "target_sd", "optimizer_steps"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite losses, scale, or optimizer steps")
    if (raw[[*METRICS, "target_sd"]] <= 0).any().any():
        raise ValueError("Expected strictly positive losses and scale")
    if not (raw.experiment_version == VERSION).all():
        raise ValueError("Mixed experiment versions")
    if ((raw.condition == "vae_augmented") & (raw.pool_seed < 0)).any():
        raise ValueError("Invalid VAE pool seed")
    if ((raw.condition != "vae_augmented") & (raw.pool_seed != -1)).any():
        raise ValueError("Invalid control pool seed")
    for metric, original in (("mse_raw", "test_mse"), ("mae_raw", "test_mae")):
        scale = raw.target_sd ** (2 if metric == "mse_raw" else 1)
        if not np.allclose(raw[metric], raw[original] * scale, rtol=1e-5, atol=1e-7):
            raise ValueError(f"Incorrect original-scale conversion: {metric}")
    controls = raw[raw.condition.isin(CONTROL_CONDITIONS)]
    synthetics = raw[raw.condition == "vae_augmented"]
    table = controls.set_index(BASEKEY + ["condition"], drop=False)
    pairs, incomplete = [], []
    for _, aug in synthetics.iterrows():
        common = tuple(aug[k] for k in BASEKEY)
        missing = [c for c in COMPARATORS if (*common, c) not in table.index]
        if missing:
            incomplete.append({**{k: aug[k] for k in BASEKEY},
                               "pool_seed": aug.pool_seed,
                               "missing": ",".join(missing)})
            continue
        ref = {c: table.loc[(*common, c)] for c in COMPARATORS}
        if any(not np.isclose(float(aug[k]), float(c[k]), rtol=0, atol=1e-10)
               for c in ref.values()
               for k in ("target_sd", "train_start", "train_end", "real_windows")):
            raise ValueError("Mismatched dataset normalization or training partition")
        if (int(aug.optimizer_steps) != int(aug.matched_steps) or
                any(int(ref[c].optimizer_steps) != int(aug.matched_steps)
                    for c in ("step_baseline", "duplicate_real")) or
                int(ref["epoch_baseline"].optimizer_steps) != int(aug.original_steps)):
            raise ValueError("Unmatched optimizer-step budgets")
        entry = {k: aug[k] for k in BASEKEY}
        entry.update(pool_seed=int(aug.pool_seed),
                     matched_steps=int(aug.matched_steps),
                     original_steps=int(aug.original_steps),
                     vae_best_at_last=int(aug.best_step) == int(aug.optimizer_steps))
        for metric in METRICS:
            entry[f"vae_{metric}"] = float(aug[metric])
            for name, reference in ref.items():
                b = float(reference[metric])
                a = float(aug[metric])
                entry[f"{name}_{metric}"] = b
                entry[f"vae_vs_{name}_{metric}_abs"] = a - b
                entry[f"vae_vs_{name}_{metric}_pct"] = 100 * (a - b) / b
        entry["vae_beats_both_matched"] = (
            entry["vae_vs_step_baseline_test_mse_abs"] < 0 and
            entry["vae_vs_duplicate_real_test_mse_abs"] < 0)
        pairs.append(entry)
    if not pairs:
        raise ValueError("No complete paired real-domain VAE comparisons")
    return pd.DataFrame(pairs), pd.DataFrame(incomplete,
                columns=BASEKEY + ["pool_seed", "missing"])


def pool_table(paired: pd.DataFrame) -> pd.DataFrame:
    # First collapse *model seeds* within each independently generated pool.
    keys = [k for k in BASEKEY if k != "rep"] + ["pool_seed"]
    contrast_cols = [col for col in paired.columns if col.startswith("vae_vs_")]
    rows = []
    for group_key, part in paired.groupby(keys, dropna=False):
        row = dict(zip(keys, group_key))
        row.update(n_training_reps=part.rep.nunique(), n_pairs=len(part),
                   vae_beats_both_fraction_reps=float(part.vae_beats_both_matched.mean()),
                   vae_best_at_last_fraction_reps=float(part.vae_best_at_last.mean()))
        for col in contrast_cols:
            row[f"median_{col}"] = float(part[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(pools: pd.DataFrame) -> pd.DataFrame:
    if pools.empty:
        return pd.DataFrame()
    keys = [x for x in BASEKEY if x != "rep"]
    contrasts = [x for x in pools.columns if x.startswith("median_vae_vs_")]
    out = []
    for values, part in pools.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        row.update(n_pool_seeds=part.pool_seed.nunique(),
                   n_training_reps_min=int(part.n_training_reps.min()),
                   n_training_reps_max=int(part.n_training_reps.max()),
                   n_pairs=int(part.n_pairs.sum()),
                   fraction_pools_vae_beats_both=float((
                       (part["median_vae_vs_step_baseline_test_mse_abs"] < 0) &
                       (part["median_vae_vs_duplicate_real_test_mse_abs"] < 0)).mean()),
                   fraction_pools_vae_checkpoint_at_end=float(
                       (part.vae_best_at_last_fraction_reps > 0.5).mean()))
        for col in contrasts:
            row[col] = float(part[col].median())
            row[f"min_{col}"] = float(part[col].min())
            row[f"max_{col}"] = float(part[col].max())
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="source", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--outdir", type=Path,
                    default=Path("results/asoc_revision/real_report"))
    args = ap.parse_args()
    src = args.source if args.source.is_absolute() else ROOT / args.source
    target = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    if not src.exists():
        ap.error(f"Missing input CSV: {src}")
    # read_completed checks duplicate IDs and provenance-oriented schema.
    _, frame = read_completed(src)
    paired, incomplete = paired_table(frame)
    pools = pool_table(paired)
    summary = summary_table(pools)
    target.mkdir(parents=True, exist_ok=True)
    paired.to_csv(target / "real_paired.csv", index=False)
    incomplete.to_csv(target / "real_incomplete.csv", index=False)
    pools.to_csv(target / "real_per_pool.csv", index=False)
    summary.to_csv(target / "real_summary.csv", index=False)
    print(f"Complete paired VAE rows: {len(paired)} | incomplete: {len(incomplete)}",
          flush=True)
    print(summary[["domain", "regime", "arch", "input_mode",
                   "n_pool_seeds", "n_training_reps_min",
                   "median_vae_vs_duplicate_real_test_mse_pct",
                   "median_vae_vs_step_baseline_test_mse_pct",
                   "median_vae_vs_duplicate_real_mse_raw_abs"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("Descriptive pool-seed medians/ranges only; not confidence intervals.")


if __name__ == "__main__":
    main()
