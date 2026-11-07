"""Report matched-step phase-3 channel/pairing ablations from CSV results.

  python -m scripts.asoc_channel_report
  python -m scripts.asoc_channel_report --in results/asoc_revision/channel_controls.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_channel_controls import CONDITIONS, KEY_COLS, ROOT

COMPARE = {
    "target_vae_vs_duplicate_pct": ("itr_target_vae", "itr_target_duplicate"),
    "target_vae_vs_step_pct": ("itr_target_vae", "itr_target_step"),
    "all_vae_random_vs_duplicate_pct": ("itr_all_vae_random", "itr_all_duplicate"),
    "all_vae_context_vs_duplicate_pct": ("itr_all_vae_context", "itr_all_duplicate"),
    "all_vae_context_vs_random_pct": ("itr_all_vae_context", "itr_all_vae_random"),
    "all_duplicate_permuted_vs_duplicate_pct": ("itr_all_duplicate_permuted", "itr_all_duplicate"),
    "all_duplicate_vs_target_duplicate_pct": ("itr_all_duplicate", "itr_target_duplicate"),
}


def paired_table(df: pd.DataFrame):
    missing = (set(KEY_COLS) | {"test_mse", "optimizer_steps", "val_mse"}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate result key(s), check CSV")
    key = [k for k in KEY_COLS if k != "condition"]
    complete, incomplete = [], []
    for fields, group in df.groupby(key, dropna=False):
        conditions = set(group["condition"])
        if conditions != set(CONDITIONS):
            incomplete.append({**dict(zip(key, fields)), "present": ",".join(sorted(conditions))})
            continue
        if group["optimizer_steps"].nunique() != 1:
            raise ValueError(f"Optimizer steps not matched: {fields}")
        if not np.isfinite(group["test_mse"]).all() or (group["test_mse"] <= 0).any():
            raise ValueError(f"Nonpositive/nonfinite test MSE: {fields}")
        records = group.set_index("condition")
        result = dict(zip(key, fields))
        result["matched_steps"] = int(group["optimizer_steps"].iloc[0])
        for label, (a, b) in COMPARE.items():
            x, y = float(records.loc[a, "test_mse"]), float(records.loc[b, "test_mse"])
            result[label] = 100 * (x - y) / y
        result["channel_interaction_pp"] = (
            result["all_vae_random_vs_duplicate_pct"]
            - result["target_vae_vs_duplicate_pct"]
        )
        for condition, suffix in [
            ("itr_all_vae_random", "random"),
            ("itr_all_vae_context", "context"),
            ("itr_all_duplicate_permuted", "permuted"),
        ]:
            if "mean_abs_corr_gap" in records.columns:
                result[f"aux_corr_gap_{suffix}"] = records.loc[condition, "mean_abs_corr_gap"]
        complete.append(result)
    return pd.DataFrame(complete), pd.DataFrame(incomplete, columns=key + ["present"])


def summary_table(paired: pd.DataFrame):
    if paired.empty:
        return pd.DataFrame()
    df = paired.copy()
    df["series_seed"] = pd.to_numeric(
        df["file"].str.extract(r"_s(\d+)\.npy$", expand=False),
        errors="coerce",
    ).fillna(-1).astype(int)
    df["series_config"] = df["file"]
    metrics = list(COMPARE) + ["channel_interaction_pp"]
    group_cols = ["regime", "epochs", "vae_epochs", "context_bins"]
    rows = []
    for fields, sub in df.groupby(group_cols):
        # training seeds are repeated measures within a single series configuration.
        by_series = sub.groupby("series_config", as_index=False)[metrics].median()
        row = dict(zip(group_cols, fields))
        row.update({
            "n_series_configurations": len(by_series),
            "n_backbone_seed_blocks": sub["series_seed"].nunique(),
            "n_training_cells": len(sub),
        })
        for metric in metrics:
            row["median_" + metric] = float(by_series[metric].median())
        row["context_better_than_random_fraction_series"] = float(
            (by_series["all_vae_context_vs_random_pct"] < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="source", type=Path,
                   default=Path("results/asoc_revision/channel_controls.csv"))
    p.add_argument("--outdir", type=Path,
                   default=Path("results/asoc_revision/channel_report"))
    args = p.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    raw = pd.read_csv(source)
    paired, incomplete = paired_table(raw)
    summary = summary_table(paired)
    outdir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(outdir / "channel_paired.csv", index=False)
    incomplete.to_csv(outdir / "channel_incomplete.csv", index=False)
    summary.to_csv(outdir / "channel_summary.csv", index=False)
    print(f"Complete cells: {len(paired)}; incomplete cells: {len(incomplete)}")
    if not summary.empty:
        print(summary.round(3).to_string(index=False))
    print("Interpretation: negative contrasts mean the first condition has lower MSE.")
    print("Nearest-context pairing is only an exploratory pairing heuristic, not true alignment.")
    print("These descriptive medians are not confidence intervals or causal proof.")


if __name__ == "__main__":
    main()
