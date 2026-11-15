"""Phase 4 paired, descriptive comparisons of bootstrap boundary controls.

Usage: python -m scripts.asoc_bootstrap_report
Repeated training seeds collapse within each series before summarizing.
Correlated Omega levels / generator seeds do NOT give independent observations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_bootstrap_controls import CONDITIONS, KEY_COLS, ROOT, DEFAULT_OUT

COMPARE = {
    "original_vs_duplicate_pct": ("bootstrap_original", "duplicate_real"),
    "safe_vs_original_pct": ("bootstrap_safe", "bootstrap_original"),
    "residual_vs_original_pct": ("bootstrap_residual", "bootstrap_original"),
    "safe_vs_duplicate_pct": ("bootstrap_safe", "duplicate_real"),
    "residual_vs_duplicate_pct": ("bootstrap_residual", "duplicate_real"),
    "safe_vs_step_pct": ("bootstrap_safe", "step_real"),
    "residual_vs_step_pct": ("bootstrap_residual", "step_real"),
}
DIAG = ("original_crossing_fraction", "safe_crossing_fraction",
        "candidate_safe_fraction", "boundary_jump_p95", "regular_jump_p95",
        "safe_unique_source_fraction", "residual_training_std")


def paired_table(raw: pd.DataFrame):
    required = set(KEY_COLS) | {"test_mse", "val_mse", "optimizer_steps", "best_step"} | set(DIAG)
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    if raw.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate Phase 4 keys in raw results")
    key = [x for x in KEY_COLS if x != "condition"]
    completed, incomplete = [], []
    for vals, group in raw.groupby(key, dropna=False):
        present = set(group.condition.astype(str))
        if present != set(CONDITIONS):
            incomplete.append({**dict(zip(key, vals)), "present": ",".join(sorted(present))})
            continue
        if group.optimizer_steps.nunique() != 1:
            raise ValueError(f"Optimizer steps not matched: {vals}")
        nums = group[["test_mse", "val_mse", "optimizer_steps", *DIAG]].to_numpy(dtype=float)
        if not np.isfinite(nums).all() or (group.test_mse <= 0).any():
            raise ValueError(f"Invalid metrics: {vals}")
        if (group.safe_crossing_fraction != 0).any():
            raise ValueError(f"Boundary-safe method crossed a splice: {vals}")
        recs = group.set_index("condition")
        entry = dict(zip(key, vals))
        entry["matched_steps"] = int(group.optimizer_steps.iloc[0])
        for name, (a, b) in COMPARE.items():
            first = float(recs.loc[a, "test_mse"])
            second = float(recs.loc[b, "test_mse"])
            entry[name] = 100 * (first - second) / second
        for col in DIAG:
            entry[col] = float(group[col].iloc[0])
        for cond in CONDITIONS:
            entry[f"{cond}_best_at_last"] = bool(
                int(recs.loc[cond, "best_step"]) == int(recs.loc[cond, "optimizer_steps"]))
        completed.append(entry)
    return pd.DataFrame(completed), pd.DataFrame(incomplete, columns=key + ["present"])


def summary_table(paired: pd.DataFrame):
    if paired.empty:
        return pd.DataFrame()
    group_fields = ("regime", "arch", "epochs", "block", "trend_window", "periods", "version")
    outputs = []
    for vals, group in paired.groupby(list(group_fields), dropna=False):
        # Median within repeated TRAINING seeds before summing across series.
        metrics = list(COMPARE) + list(DIAG)
        series = group.groupby("file", as_index=False)[metrics].median()
        entry = dict(zip(group_fields, vals))
        entry["n_training_cells"] = len(group)
        entry["n_series_configurations"] = len(series)
        entry["n_backbone_seed_blocks"] = group.file.str.extract(r"_s(\d+)\.npy$", expand=False).nunique()
        for m in metrics:
            entry[f"median_{m}"] = float(series[m].median())
        entry["safe_beats_original_fraction_series"] = float((series.safe_vs_original_pct < 0).mean())
        entry["residual_beats_original_fraction_series"] = float((series.residual_vs_original_pct < 0).mean())
        entry["safe_beats_duplicate_fraction_series"] = float((series.safe_vs_duplicate_pct < 0).mean())
        entry["residual_beats_duplicate_fraction_series"] = float((series.residual_vs_duplicate_pct < 0).mean())
        for cond in CONDITIONS:
            entry[f"{cond}_last_checkpoint_fraction_cells"] = float(group[f"{cond}_best_at_last"].mean())
        outputs.append(entry)
    return pd.DataFrame(outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="source", type=Path, default=DEFAULT_OUT)
    p.add_argument("--outdir", type=Path, default=Path("results/asoc_revision/bootstrap_report"))
    args = p.parse_args()
    src = args.source if args.source.is_absolute() else ROOT / args.source
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    paired, incomplete = paired_table(pd.read_csv(src))
    summary = summary_table(paired)
    outdir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(outdir / "bootstrap_paired.csv", index=False)
    incomplete.to_csv(outdir / "bootstrap_incomplete.csv", index=False)
    summary.to_csv(outdir / "bootstrap_summary.csv", index=False)
    print(f"Phase 4: complete paired cells={len(paired)}, incomplete={len(incomplete)}")
    if not summary.empty:
        cols = ["regime", "arch", "n_series_configurations", "n_training_cells",
                "median_original_crossing_fraction", "median_safe_vs_original_pct",
                "median_residual_vs_original_pct", "median_safe_vs_duplicate_pct",
                "median_residual_vs_duplicate_pct"]
        print(summary[cols].round(3).to_string(index=False))
    print("Negative ΔMSE% favors numerator. Series seeds are correlated across Omega bands; no CIs.")


if __name__ == "__main__":
    main()
