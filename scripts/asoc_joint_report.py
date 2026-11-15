"""ASOC Phase 3B: paired independent-seed descriptive summaries.

Run:
  python -m scripts.asoc_joint_report

Each training repetition is paired. The same backbone generator seed is
reused across Omega bands, so the configurations are NOT independent draws.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_joint_controls import CONDITIONS, KEY_COLS, ROOT, DEFAULT_OUT

COMPARE = {
    "joint_vs_original_pct": ("itr_all_vae_joint", "itr_all_vae_original"),
    "joint_vs_duplicate_pct": ("itr_all_vae_joint", "itr_all_duplicate"),
    "original_vs_duplicate_pct": ("itr_all_vae_original", "itr_all_duplicate"),
    "joint_shuffled_vs_joint_pct": ("itr_all_vae_joint_shuffled", "itr_all_vae_joint"),
    "joint_shuffled_vs_duplicate_pct": ("itr_all_vae_joint_shuffled", "itr_all_duplicate"),
    "duplicate_vs_step_pct": ("itr_all_duplicate", "itr_all_step"),
}


def paired_table(raw: pd.DataFrame):
    required = set(KEY_COLS) | {"test_mse", "val_mse", "optimizer_steps", "best_step"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    if raw.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate keys in raw output")
    k = [c for c in KEY_COLS if c != "condition"]
    complete, incomplete = [], []
    for values, group in raw.groupby(k, dropna=False):
        present = set(group["condition"].astype(str))
        if present != set(CONDITIONS):
            incomplete.append({**dict(zip(k, values)), "present": ",".join(sorted(present))})
            continue
        if group["optimizer_steps"].nunique() != 1:
            raise ValueError(f"Unequal optimizer steps in {values}")
        if not np.isfinite(group[["test_mse", "val_mse"]].to_numpy(dtype=float)).all() or (group["test_mse"] <= 0).any():
            raise ValueError(f"Nonfinite or nonpositive test MSE in {values}")
        records = group.set_index("condition")
        entry = dict(zip(k, values))
        entry["matched_steps"] = int(group["optimizer_steps"].iloc[0])
        for title, (first, second) in COMPARE.items():
            a = float(records.loc[first, "test_mse"])
            b = float(records.loc[second, "test_mse"])
            entry[title] = 100.0 * (a - b) / b
        for name, condition in [("original", "itr_all_vae_original"),
                                ("joint", "itr_all_vae_joint"),
                                ("shuffled", "itr_all_vae_joint_shuffled")]:
            for diag in ("lag_corr_gap", "future_assoc_gap", "zero_lag_corr_gap"):
                if diag in records.columns:
                    entry[f"{name}_{diag}"] = float(records.loc[condition, diag])
        for cond in CONDITIONS:
            entry[f"{cond}_last_checkpoint"] = bool(
                int(records.loc[cond, "best_step"]) == int(records.loc[cond, "optimizer_steps"]))
        complete.append(entry)
    return pd.DataFrame(complete), pd.DataFrame(incomplete, columns=k + ["present"])


def summary_table(paired: pd.DataFrame):
    if paired.empty:
        return pd.DataFrame()
    metrics = list(COMPARE)
    optional = [c for c in paired.columns if c.endswith(("_lag_corr_gap", "_future_assoc_gap", "_zero_lag_corr_gap"))]
    group_cols = ["regime", "epochs", "vae_epochs", "latent", "hidden", "horizon_weight"]
    result = []
    for vals, part in paired.groupby(group_cols):
        # Training repetitions are NOT independent: collapse per-series first.
        series = part.groupby("file", as_index=False)[metrics + optional].median()
        row = dict(zip(group_cols, vals))
        row["n_series_configurations"] = len(series)
        row["n_backbone_seed_blocks"] = part["file"].str.extract(r"_s(\d+)\.npy$", expand=False).nunique()
        row["n_training_cells"] = len(part)
        for name in metrics + optional:
            row[f"median_{name}"] = float(series[name].median())
        row["joint_beats_original_fraction_series"] = float((series["joint_vs_original_pct"] < 0).mean())
        row["joint_beats_duplicate_fraction_series"] = float((series["joint_vs_duplicate_pct"] < 0).mean())
        row["shuffle_hurts_joint_fraction_series"] = float((series["joint_shuffled_vs_joint_pct"] > 0).mean())
        for condition in CONDITIONS:
            row[f"{condition}_best_at_last_fraction_cells"] = float(part[f"{condition}_last_checkpoint"].mean())
        result.append(row)
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="source", type=Path, default=DEFAULT_OUT)
    p.add_argument("--outdir", type=Path, default=Path("results/asoc_revision/joint_report"))
    args = p.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    paired, incomplete = paired_table(pd.read_csv(source))
    summary = summary_table(paired)
    outdir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(outdir / "joint_paired.csv", index=False)
    incomplete.to_csv(outdir / "joint_incomplete.csv", index=False)
    summary.to_csv(outdir / "joint_summary.csv", index=False)
    print(f"Phase 3B complete paired cells: {len(paired)}; incomplete: {len(incomplete)}", flush=True)
    if not summary.empty:
        cols = ["regime", "n_series_configurations", "n_training_cells",
                "median_joint_vs_original_pct", "median_joint_vs_duplicate_pct",
                "median_joint_shuffled_vs_joint_pct",
                "median_joint_lag_corr_gap", "median_joint_future_assoc_gap"]
        print(summary[[c for c in cols if c in summary.columns]].round(3).to_string(index=False))
    print("Negative ΔMSE% favors numerator; shuffled-vs-joint positive supports alignment relevance.")
    print("Caution: descriptive, correlated backbone seeds; NOT confidence intervals or a causal proof.")


if __name__ == "__main__":
    main()
