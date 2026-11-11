"""Phase 13 R1.7 reporting: join TRAIN-only reconstructed VAE diagnostics
with previously recorded matched-control test outcomes. Descriptive only.

  python -m scripts.asoc_fidelity_report --source nonfourier
  python -m scripts.asoc_fidelity_report --source real
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.asoc_fidelity import (ROOT, DEFAULT_INPUT, DEFAULT_OUT,
                                  GROUP_FIELDS, METRICS, VERSION)

CONTROL_NAMES = ("step_baseline", "duplicate_real")


def _select(df: pd.DataFrame, row, fields):
    mask = np.ones(len(df), dtype=bool)
    for k in fields:
        if k not in df:
            raise ValueError(f"Missing field: {k}")
        if pd.api.types.is_numeric_dtype(df[k]):
            mask &= np.isclose(df[k].to_numpy(dtype=float), float(row[k]),
                               atol=1e-9, rtol=1e-7)
        else:
            mask &= df[k].astype(str).to_numpy() == str(row[k])
    return df.loc[mask]


def combine(run_df: pd.DataFrame, diag_df: pd.DataFrame, source: str) -> pd.DataFrame:
    if source not in GROUP_FIELDS:
        raise ValueError("Unknown dataset source")
    fields = GROUP_FIELDS[source]
    if diag_df.empty:
        raise ValueError("No diagnostics available")
    if diag_df.duplicated(list(fields)).any():
        raise ValueError("Repeated reconstructed pool rows")
    if not (diag_df.diagnostic_version == VERSION).all():
        raise ValueError("Incorrect diagnostic version")
    rows = []
    control_context = (["family", "series_id", "regime", "arch", "data_sha256"]
                       if source == "nonfourier" else
                       ["domain", "regime", "arch", "input_mode", "data_sha256"])
    matching_params = ["epochs", "batch_size", "lr", "vae_epochs", "latent"]
    for _, d in diag_df.iterrows():
        vaes = _select(run_df[run_df.condition == "vae_augmented"], d, fields)
        if vaes.empty:
            raise ValueError("Diagnostics lack VAE outcome pairing")
        diffs = []
        for _, vae in vaes.iterrows():
            ref_context = {**vae.to_dict()}
            match_fields = control_context + matching_params + ["rep"]
            candidates = _select(run_df[run_df.condition.isin(CONTROL_NAMES)],
                                 ref_context, match_fields)
            refs = {}
            for name in CONTROL_NAMES:
                subset = candidates[candidates.condition == name]
                if len(subset) != 1:
                    raise ValueError(f"Expected one {name} control per model seed")
                ref = subset.iloc[0]
                if int(ref.optimizer_steps) != int(vae.optimizer_steps):
                    raise ValueError("Optimizer steps mismatch")
                for part in ("train_start", "train_end", "real_windows", "target_sd"):
                    if not np.isclose(float(ref[part]), float(vae[part]), rtol=1e-6, atol=1e-6):
                        raise ValueError(f"Unmatched training information: {part}")
                refs[name] = ref
            a = float(vae.test_mse)
            if not a > 0:
                raise ValueError("Non-positive VAE test loss")
            diffs.append({f"vae_vs_{name}_mse_pct": 100 * (a / float(ref.test_mse) - 1)
                          for name, ref in refs.items()})
        res = {**d.to_dict(), "n_model_seeds": vaes.rep.nunique(),
               "n_paired_runs": len(diffs)}
        for name in CONTROL_NAMES:
            col = f"vae_vs_{name}_mse_pct"
            res[col] = float(pd.DataFrame(diffs)[col].median())
        res["vae_beats_both"] = (res["vae_vs_step_baseline_mse_pct"] < 0 and
                                 res["vae_vs_duplicate_real_mse_pct"] < 0)
        rows.append(res)
    return pd.DataFrame(rows)


def _series_level(joined: pd.DataFrame, source: str):
    """Collapse VAE pool seeds *within each independent non-Fourier series*."""
    if source != "nonfourier":
        return joined
    keys = ["family", "series_id", "regime"]
    cols = ["vae_vs_duplicate_real_mse_pct", "vae_vs_step_baseline_mse_pct"] + [
        "vae_" + name for name in METRICS]
    return joined.groupby(keys, as_index=False)[cols].median()


def summarize(joined: pd.DataFrame, source: str):
    independent_keys = (["family", "regime"] if source == "nonfourier"
                        else ["domain", "regime"])
    xall = _series_level(joined, source)
    rows = []
    for groups, x in xall.groupby(independent_keys, sort=True):
        row = dict(zip(independent_keys, groups))
        row.update(n_pool_rows=len(joined.loc[
                    (joined[independent_keys[0]] == groups[0]) &
                    (joined.regime == groups[1])]),
                   n_independent_series=(x.series_id.nunique()
                    if source == "nonfourier" else np.nan),
                   fraction_pool_rows_beating_both=float(joined.loc[
                    (joined[independent_keys[0]] == groups[0]) &
                    (joined.regime == groups[1]), "vae_beats_both"].mean()))
        if source == "nonfourier":
            row["fraction_series_beating_both"] = float((
                (x.vae_vs_step_baseline_mse_pct < 0) &
                (x.vae_vs_duplicate_real_mse_pct < 0)).mean())
        for c in ("vae_vs_duplicate_real_mse_pct", "vae_vs_step_baseline_mse_pct"):
            row["median_" + c] = float(x[c].median())
        for c in ("vae_" + m for m in METRICS):
            row["median_" + c] = float(x[c].median())
        rows.append(row)
    return pd.DataFrame(rows)


def exploratory_correlations(joined: pd.DataFrame, source: str):
    """Series-level rankings; descriptive, without p-values/CIs or causality."""
    if source != "nonfourier":
        return pd.DataFrame(columns=["scope", "n", "distance", "target", "spearman_rho"])
    series = _series_level(joined, source)
    rows = []
    for scope, x in [("all", series)] + [
        (f"regime={r:.2f}", part) for r, part in series.groupby("regime")]:
        for metric in ("vae_marginal_w1", "vae_sliced_w1", "vae_spectral_js",
                       "vae_acf_mae"):
            for target in ("vae_vs_duplicate_real_mse_pct", "vae_vs_step_baseline_mse_pct"):
                if len(x) >= 5 and x[metric].nunique() > 1 and x[target].nunique() > 1:
                    rho = float(spearmanr(x[metric], x[target]).statistic)
                else:
                    rho = np.nan
                rows.append(dict(scope=scope, n=len(x), distance=metric,
                                 target=target, spearman_rho=rho))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=tuple(GROUP_FIELDS), default="nonfourier")
    p.add_argument("--runs", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    runs_path = args.runs or DEFAULT_INPUT[args.source]
    runs_path = runs_path if runs_path.is_absolute() else ROOT / runs_path
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    diag_path = outdir / f"{args.source}_fidelity.csv"
    if not runs_path.is_file() or not diag_path.is_file():
        p.error(f"Missing run/diagnostic CSV: {runs_path}, {diag_path}")
    try:
        joined = combine(pd.read_csv(runs_path), pd.read_csv(diag_path), args.source)
        summary = summarize(joined, args.source)
        corrs = exploratory_correlations(joined, args.source)
    except ValueError as exc:
        p.error(str(exc))
    outdir.mkdir(parents=True, exist_ok=True)
    for name, table in ((f"{args.source}_fidelity_joined.csv", joined),
                        (f"{args.source}_fidelity_summary.csv", summary),
                        (f"{args.source}_fidelity_correlations.csv", corrs)):
        table.to_csv(outdir / name, index=False)
    cols = (["family", "regime"] if args.source == "nonfourier" else
            ["domain", "regime"])
    cols += ["n_pool_rows", "median_vae_marginal_w1", "median_vae_sliced_w1",
             "median_vae_spectral_js", "median_vae_acf_mae",
             "median_vae_vs_duplicate_real_mse_pct", "median_vae_vs_step_baseline_mse_pct"]
    print(f"Joined {len(joined)} reconstructed VAE pool diagnostic rows ",
          "with previously recorded outcomes. No new forecaster training.")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("NOTE: Original synthetic windows were not archived. Fidelity is reconstructed, "
          "not verified bitwise against original. Correlations are exploratory and not causal.")


if __name__ == "__main__":
    main()
