"""Read-only Weather Phase-7 integrity and cost preflight.

Run at repository root:
    python -m scripts.asoc_weather_preflight

Does not train, regenerate data, or write outputs. Reconstructs the benchmark
preprocessing used by scripts/prepare_real.py as inspected in revision branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_weather(raw_csv: Path, prepared_npy: Path, previous_results: Path | None = None) -> dict:
    """Return a JSON-ready report. Empty `errors` is the gate for running the pilot."""
    result = {"domain": "weather", "errors": [], "warnings": [], "regimes": []}
    errors, warnings = result["errors"], result["warnings"]
    if not raw_csv.is_file() or not prepared_npy.is_file():
        errors.append(f"Missing required file(s): {raw_csv} or {prepared_npy}")
        return result

    raw = pd.read_csv(raw_csv)
    n = len(raw)
    date_cols = [c for c in raw.columns if c.lower() == "date"]
    if len(date_cols) != 1:
        errors.append(f"Expected exactly one date column; found {date_cols}")
    else:
        dates = pd.to_datetime(raw[date_cols[0]], errors="coerce")
        bad_dates = int(dates.isna().sum())
        duplicates = int(dates.duplicated().sum())
        backward = int((dates.diff().dropna() < pd.Timedelta(0)).sum())
        bad_cadence = int((dates.diff().dropna() != pd.Timedelta(minutes=10)).sum())
        result["timestamps"] = {
            "first": str(dates.iloc[0]) if n else None,
            "last": str(dates.iloc[-1]) if n else None,
            "null": bad_dates, "duplicates": duplicates,
            "backwards": backward, "non_10min_intervals": bad_cadence,
        }
        if any((bad_dates, duplicates, backward, bad_cadence)):
            errors.append("Weather timestamps have nulls, duplicates, reordering or irregular 10-minute gaps; audit before training")

    numeric = raw.drop(columns=date_cols).apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 3:
        errors.append("Need target plus at least two numeric covariates")
        return result
    missing = numeric.isna().sum()
    result["raw_missing_total"] = int(missing.sum())
    result["missing_by_column"] = {str(k): int(v) for k, v in missing.items() if v > 0}
    if result["raw_missing_total"]:
        errors.append("CSV has missing/non-numeric entries; original ffill/bfill can cross temporal boundaries. Audit imputations before training")

    # Mirror exactly scripts/prepare_real.py, without writing any .npy.
    imputed = numeric.ffill().bfill()
    target = "OT" if "OT" in imputed.columns else imputed.columns[-1]
    tr = imputed.iloc[:int(n * .7)]
    corr = (tr.drop(columns=[target]).apply(lambda c: tr[target].corr(c))
            .abs().sort_values(ascending=False))
    aux = corr.index[:2].tolist()
    result.update(target=str(target), auxiliary=[str(x) for x in aux])
    rebuilt = imputed[[target] + aux].to_numpy(dtype=np.float32)
    saved = np.load(prepared_npy, allow_pickle=False)
    result.update(csv_rows=n, prepared_shape=list(saved.shape), prepared_dtype=str(saved.dtype),
                  prepared_sha256=_sha256(prepared_npy))
    if saved.shape != rebuilt.shape or saved.dtype != np.float32:
        errors.append(f"Prepared array shape/dtype {saved.shape}/{saved.dtype} does not match reconstructed {rebuilt.shape}/float32")
    elif not np.array_equal(saved, rebuilt):
        neq = int(np.count_nonzero(saved != rebuilt))
        errors.append(f"Prepared array differs from current CSV preprocessing in {neq} entries; do NOT regenerate over existing results")
    if not np.isfinite(saved).all():
        errors.append("Prepared array contains NaN/inf")
    if len(saved) < 500:
        errors.append("Weather array has fewer than 500 rows")

    if previous_results is not None and previous_results.exists():
        prior = pd.read_csv(previous_results)
        if {"domain", "data_sha256"}.issubset(prior.columns):
            records = prior[prior.domain == "weather"]
            result["existing_weather_rows"] = len(records)
            if len(records) and not (records.data_sha256 == result["prepared_sha256"]).all():
                errors.append("Weather sha256 differs from existing Phase 7 results; use a NEW output and investigate provenance")
        else:
            errors.append("Existing Phase 7 output lacks domain/data_sha256 columns")

    # Same protocol as src/data.py & scripts/asoc_real_controls.py.
    t1 = int(n * .7)
    t2 = int(n * (.7 + .1))
    result["split"] = {"train_end_exclusive": t1, "val_end_exclusive": t2,
                       "test_end_exclusive": n, "lookback": 96, "horizon": 24}
    for regime in (.10, 1.0):
        start = max(0, min(int(t1 - regime * t1), t1 - 96 - 1))
        train_window_count = t1 - start - 96 - 24 + 1
        if train_window_count <= 0:
            errors.append(f"Not enough training windows for regime {regime}")
            continue
        epoch_steps = 10 * math.ceil(train_window_count / 64)
        matched_steps = 10 * math.ceil(2 * train_window_count / 64)
        result["regimes"].append({"regime": regime, "train_start": start,
                                  "real_train_windows": train_window_count,
                                  "epoch_baseline_steps": epoch_steps,
                                  "matched_steps_per_arm": matched_steps,
                                  "forecast_training_runs_if_rep0_pool0_1": 5,
                                  "total_forecaster_steps_if_new": epoch_steps + 4 * matched_steps})
    if len(saved) == n and saved.ndim == 2 and np.isfinite(saved).all():
        if float(np.std(saved[:, 0])) < 1e-8:
            errors.append("Target appears nearly constant")
        result["target_std_full"] = float(np.std(saved[:, 0]))
    if not result["errors"]:
        result["pass"] = True
    else:
        result["pass"] = False
    if not previous_results or not previous_results.exists():
        warnings.append("No previous Phase 7 CSV was available for SHA cross-check (normal on a fresh environment)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/real/weather/weather.csv")
    parser.add_argument("--npy", type=Path, default=ROOT / "data/real/weather.npy")
    parser.add_argument("--runs", type=Path, default=ROOT / "results/asoc_revision/real_budget_controls.csv")
    args = parser.parse_args()
    report = audit_weather(args.csv, args.npy, args.runs)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.get("pass") else 2)


if __name__ == "__main__":
    main()
