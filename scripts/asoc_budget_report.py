"""Audit ASOC step-budget control results without changing archived results.

Reads phase-1 CSV (GPU and non-GPU variants supported), checks paired design,
then writes per-cell contrasts and series-level summaries. Negative percent
means that the numerator model has lower MSE than its comparator.

    python -m scripts.asoc_budget_report
    python -m scripts.asoc_budget_report --in results/asoc_revision/budget_controls.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONDITIONS = ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
KEY = ("file", "regime", "arch", "rep", "pool_seed", "input_mode",
       "epochs", "batch_size", "lr", "vae_epochs")
CONTRASTS = {
    "vae_vs_epoch_pct": ("vae_augmented", "epoch_baseline"),
    "vae_vs_step_pct": ("vae_augmented", "step_baseline"),
    "vae_vs_duplicate_pct": ("vae_augmented", "duplicate_real"),
    "duplicate_vs_step_pct": ("duplicate_real", "step_baseline"),
    "step_vs_epoch_pct": ("step_baseline", "epoch_baseline"),
}


def paired_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(KEY) | {"condition", "test_mse", "val_mse", "best_step", "optimizer_steps"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.duplicated(list(KEY) + ["condition"]).any():
        raise ValueError("Duplicate run keys: preserve records and audit before reporting")
    invalid = set(df.condition) - set(REQUIRED_CONDITIONS)
    if invalid:
        raise ValueError(f"Unrecognized conditions {invalid}")
    groups = df.groupby(list(KEY), dropna=False, sort=True)
    complete_keys = []
    incomplete_keys = []
    for key, g in groups:
        if set(g.condition) == set(REQUIRED_CONDITIONS):
            complete_keys.append(key)
        else:
            incomplete_keys.append(key)
    if not complete_keys:
        raise ValueError("No complete 4-condition experiment cells")
    indexed = df.set_index(list(KEY))
    sub = indexed.loc[complete_keys].reset_index()
    pivot = sub.pivot(index=list(KEY), columns="condition", values="test_mse")
    out = pivot.reset_index()
    for name, (num, den) in CONTRASTS.items():
        out[name] = 100.0 * (out[num] - out[den]) / out[den]
    out["vae_beats_both_matched"] = (
        (out["vae_augmented"] < out["step_baseline"]) &
        (out["vae_augmented"] < out["duplicate_real"]))

    checkpoint_info = sub.assign(
        selected_last=lambda t: t["best_step"] == t["optimizer_steps"])
    best_last = checkpoint_info.pivot(
        index=list(KEY), columns="condition", values="selected_last").reset_index()
    for c in REQUIRED_CONDITIONS:
        out[f"best_at_last_{c}"] = best_last[c].to_numpy(dtype=bool)
    remaining = pd.DataFrame(incomplete_keys, columns=KEY)
    return out, remaining


def series_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    # First aggregate training repetitions within each series configuration.
    # The same generator seed is reused across Omega bands, so file entries
    # with the same seed may be correlated; they are NOT independent draws.
    series_key = ["file", "regime", "arch", "input_mode", "epochs"]
    cols = list(CONTRASTS)
    per_series = pairs.groupby(series_key, as_index=False)[cols].median()
    rows = []
    for key, group in per_series.groupby(
            ["regime", "arch", "input_mode", "epochs"], sort=True):
        regime, arch, mode, epochs = key
        subset = pairs[(pairs.regime == regime) & (pairs.arch == arch) &
                       (pairs.input_mode == mode) & (pairs.epochs == epochs)]
        seed_blocks = set()
        for filename in group["file"]:
            found = re.search(r"_s(\d+)\.npy$", str(filename))
            seed_blocks.add(found.group(1) if found else str(filename))
        row = {
            "regime": regime, "arch": arch, "input_mode": mode, "epochs": epochs,
            "n_series_configurations": len(group),
            "n_backbone_seed_blocks": len(seed_blocks),
            "n_training_cells": len(subset),
            "vae_beats_both_fraction_cells": float(subset.vae_beats_both_matched.mean()),
            "vae_best_checkpoint_at_last_fraction": float(subset.best_at_last_vae_augmented.mean()),
            "step_best_checkpoint_at_last_fraction": float(subset.best_at_last_step_baseline.mean()),
            "duplicate_best_checkpoint_at_last_fraction": float(subset.best_at_last_duplicate_real.mean()),
        }
        for col in cols:
            row[f"median_{col}"] = float(group[col].median())
            row[f"q25_{col}"] = float(group[col].quantile(.25))
            row[f"q75_{col}"] = float(group[col].quantile(.75))
            row[f"fraction_series_{col}_negative"] = float((group[col] < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", type=Path,
                   default=Path("results/asoc_revision/budget_matrix_v2.csv"))
    p.add_argument("--outdir", type=Path,
                   default=Path("results/asoc_revision"))
    args = p.parse_args()
    infile = args.infile if args.infile.is_absolute() else ROOT / args.infile
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    if not infile.exists():
        p.error(f"Missing results file {infile}")
    pairs, incomplete = paired_cells(pd.read_csv(infile))
    summary = series_summary(pairs)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(outdir / "budget_paired_cells.csv", index=False)
    summary.to_csv(outdir / "budget_series_summary.csv", index=False)
    incomplete.to_csv(outdir / "budget_incomplete_cells.csv", index=False)
    pd.set_option("display.max_columns", None)
    print(f"Complete paired cells: {len(pairs)}; incomplete: {len(incomplete)}")
    show = ["regime", "arch", "epochs", "n_series_configurations",
            "n_backbone_seed_blocks", "n_training_cells",
            "median_vae_vs_epoch_pct", "median_vae_vs_step_pct",
            "median_vae_vs_duplicate_pct", "vae_beats_both_fraction_cells",
            "vae_best_checkpoint_at_last_fraction"]
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("NOTE: negative ΔMSE% means improvement; the summary uses medians of"
          " per-configuration medians, NOT confidence intervals.")
    print("The generator reuses RNG seeds across Omega bands; configurations"
          " sharing a seed are not statistically independent.")
    print("Pilot results are descriptive and must not be treated as a new general finding.")
    print(f"Wrote 3 CSV reports under {outdir}")


if __name__ == "__main__":
    main()
