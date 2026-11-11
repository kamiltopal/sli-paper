"""Read-only EPİAŞ Phase-7 gate: raw PTF CSV, prepared .npy, chronology, SHA.

Never writes or repairs source datasets. Run BEFORE EPİAŞ model training:

  python -m scripts.asoc_epias_preflight --regime 0.10
  python -m scripts.asoc_epias_preflight --regime 1.00

The reconstruction follows scripts/prepare_real.py: load_epias, including its
first-occurrence duplicate removal and *lack* of missing-hour imputation.
A pass is scoped to the selected regime and original univariate target-only
Phase-7 protocol; it is not a global endorsement of all data characteristics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import WindowConfig, apply_regime, chrono_split

ROOT = Path(__file__).resolve().parents[1]
CSV_DEFAULT = ROOT / "data" / "real" / "epias" / "ptf.csv"
NPY_DEFAULT = ROOT / "data" / "real" / "epias.npy"
RUNS_DEFAULT = ROOT / "results" / "asoc_revision" / "real_budget_controls.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat(sep=" ")


def reconstruct(raw: pd.DataFrame) -> tuple[np.ndarray, pd.Series, dict]:
    """Mirror load_epias() in memory, exposing dropped/ambiguous time records."""
    needed = {"Tarih", "Saat"}
    if not needed.issubset(raw.columns):
        raise ValueError(f"Missing source columns: {sorted(needed - set(raw.columns))}")
    tl_cols = [c for c in raw.columns if "TL" in c]
    if not tl_cols:
        raise ValueError("No column containing 'TL'; current prepare_real.load_epias would fail")
    # Same source selection and price parsing as the current preparation code.
    tl = tl_cols[0]
    dt = pd.to_datetime(raw["Tarih"] + " " + raw["Saat"],
                        format="%d.%m.%Y %H:%M", errors="raise")
    prices = (raw[tl].str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False).astype(float))
    selected = pd.DataFrame({"dt": dt, "ptf": prices}).dropna()
    repeated = selected.duplicated("dt", keep="first")
    conflicting = 0
    first_price = {}
    for ts, price in zip(selected["dt"], selected["ptf"]):
        if ts in first_price and price != first_price[ts]:
            conflicting += 1
        first_price.setdefault(ts, price)
    clean = selected.drop_duplicates("dt").reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid EPİAŞ rows")
    hour = clean["dt"].dt.hour.to_numpy()
    dow = clean["dt"].dt.dayofweek.to_numpy()
    rebuilt = np.stack([clean["ptf"].to_numpy(),
                        np.sin(2 * np.pi * hour / 24),
                        dow / 6.0], axis=1).astype(np.float32)
    details = {
        "raw_rows": int(len(raw)), "source_tl_column": tl,
        "valid_rows_before_dedup": int(len(selected)),
        "raw_dropped_null_date_or_price": int(len(raw) - len(selected)),
        "removed_duplicate_timestamps": int(repeated.sum()),
        "duplicate_price_conflicts": int(conflicting),
        "reconstructed_rows": int(len(clean)),
    }
    return rebuilt, clean["dt"], details


def time_audit(dates: pd.Series, train_start: int, train_end: int,
               val_end: int, max_examples: int = 8) -> dict:
    """Check naive local hourly cadence AFTER prepare_real duplicate removal."""
    if len(dates) <= 1:
        raise ValueError("Not enough timestamps")
    dates = dates.reset_index(drop=True)
    gaps = dates.diff()
    irregular = gaps.notna() & (gaps != pd.Timedelta(hours=1))
    where = np.flatnonzero(irregular.to_numpy())

    def region(i: int) -> str:
        if i < train_start:
            return "excluded_before_selected_training"
        if i < train_end:
            return "train"
        if i < val_end:
            return "validation"
        return "test"

    events = [{"index": int(i), "region": region(int(i)),
               "previous": _iso(dates.iloc[i - 1]),
               "current": _iso(dates.iloc[i]),
               "delta_minutes": int(gaps.iloc[i].total_seconds() // 60)}
              for i in where[:max_examples]]
    active = where[where > train_start]  # first active observation has no prior active neighbor
    # A missing instant may be an artifact of source time-zone/DST conventions;
    # we report it, never invent a replacement observation.
    missing = pd.date_range(dates.min(), dates.max(), freq="h").difference(
        pd.DatetimeIndex(dates))
    return {
        "first": _iso(dates.iloc[0]), "last": _iso(dates.iloc[-1]),
        "global_irregular_intervals": int(len(where)),
        "selected_regime_irregular_intervals": int(len(active)),
        "global_nonmonotone_intervals": int((gaps.dropna() < pd.Timedelta(0)).sum()),
        "missing_calendar_hours_naive": int(len(missing)),
        "missing_hour_examples": [_iso(x) for x in missing[:max_examples]],
        "irregular_examples": events,
        "note": "Naive local clock timestamps; DST/source policy must be reviewed if anomalies exist.",
    }


def audit(*, csv_path: Path, npy_path: Path, runs_path: Path,
          regime: float = 0.10, epochs: int = 10,
          batch_size: int = 64) -> dict:
    if regime not in (0.10, 0.25, 1.0):
        raise ValueError("Regime must be 0.10, 0.25, or 1.0")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    report = {
        "domain": "epias", "regime": regime, "read_only": True,
        "csv_path": str(csv_path), "npy_path": str(npy_path),
        "phase7_results_path": str(runs_path),
        "pass": False, "errors": [], "warnings": [],
        "protocol": "phase7_real_equal_step_v1 / itransformer / target-only / lookback96_horizon24",
    }
    errors = report["errors"]
    warnings = report["warnings"]
    if not npy_path.is_file():
        errors.append(f"Prepared array missing: {npy_path}; do not run prepare_real.py blindly")
        return report
    try:
        arr = np.load(npy_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot load prepared array: {exc}")
        return report
    report["prepared_sha256"] = sha256(npy_path)
    report["prepared_shape"] = list(arr.shape)
    report["prepared_dtype"] = str(arr.dtype)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 500:
        errors.append("Expected (T,3) prepared array with at least 500 rows")
        return report
    if arr.dtype != np.float32:
        errors.append("Expected float32 .npy, as constructed by load_epias")
    if not np.isfinite(arr).all():
        errors.append("Prepared array has nonfinite values")
    if not csv_path.is_file():
        errors.append(f"Raw EPİAŞ CSV missing: {csv_path}; cannot verify prepared-array provenance")
        return report
    try:
        raw = pd.read_csv(csv_path, sep=";", dtype=str)
        rebuilt, dates, details = reconstruct(raw)
    except (KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        errors.append(f"Cannot reproduce load_epias conversion from raw CSV: {exc}")
        return report
    report["source_sha256"] = sha256(csv_path)
    report["raw_preparation"] = details
    report["raw_matches_prepared"] = bool(
        arr.shape == rebuilt.shape and np.array_equal(arr, rebuilt))
    if not report["raw_matches_prepared"]:
        errors.append("Raw CSV reconstruction differs from epias.npy; inspect provenance, DO NOT overwrite")
    if details["removed_duplicate_timestamps"]:
        warnings.append("load_epias keeps the FIRST timestamp occurrence; duplicate handling must be disclosed")
    if details["duplicate_price_conflicts"]:
        warnings.append("Duplicate timestamps disagree on price; investigate source policy")
    if details["raw_dropped_null_date_or_price"]:
        warnings.append("Raw rows with missing date/price were dropped by load_epias")

    cfg = WindowConfig(lookback=96, horizon=24)
    t1, t2 = chrono_split(len(arr))
    tstart = apply_regime(t1, regime, cfg.lookback)
    n_train = t1 - tstart - cfg.lookback - cfg.horizon + 1
    n_val = t2 - t1 - cfg.horizon + 1
    n_test = len(arr) - t2 - cfg.horizon + 1
    if min(n_train, n_val, n_test) < 1:
        errors.append("Insufficient windows under selected split/configuration")
        return report
    report["split"] = {
        "train_start": tstart, "train_end_exclusive": t1,
        "validation_end_exclusive": t2, "test_end_exclusive": len(arr),
        "train_start_time": _iso(dates.iloc[tstart]),
        "train_last_time": _iso(dates.iloc[t1 - 1]),
        "validation_start_time": _iso(dates.iloc[t1]),
        "validation_last_time": _iso(dates.iloc[t2 - 1]),
        "test_start_time": _iso(dates.iloc[t2]),
        "test_end_time": _iso(dates.iloc[-1]),
        "real_train_windows": n_train, "val_windows": n_val,
        "test_windows": n_test,
        "epoch_baseline_steps": epochs * math.ceil(n_train / batch_size),
        "matched_steps_per_arm": epochs * math.ceil(2 * n_train / batch_size),
        "normalization": "training_slice_only (data.prepare)",
    }
    report["time_audit"] = time_audit(dates, tstart, t1, t2)
    if report["time_audit"]["global_irregular_intervals"]:
        warnings.append("Irregular hourly timestamps exist globally; inspect examples and DST conventions")
    if report["time_audit"]["selected_regime_irregular_intervals"]:
        errors.append("Selected regime training/validation/test has irregular timestamps; audit before training")
    if report["time_audit"]["global_nonmonotone_intervals"]:
        errors.append("Timestamp ordering is not chronological in prepared sequence")
    target_sd = float(np.std(arr[tstart:t1, 0], dtype=np.float64))
    report["train_target_std"] = target_sd
    if target_sd <= 0 or not np.isfinite(target_sd):
        errors.append("Selected train target has zero/nonfinite variance")

    if runs_path.is_file():
        try:
            runs = pd.read_csv(runs_path)
            if not {"domain", "data_sha256"}.issubset(runs.columns):
                errors.append("Existing Phase-7 results CSV has no domain/data_sha256 fields")
            else:
                epias = runs[runs.domain == "epias"]
                report["existing_phase7_epias_rows"] = int(len(epias))
                bad_sha = (epias.data_sha256.astype(str) != report["prepared_sha256"]).sum()
                if bad_sha:
                    errors.append(f"Existing EPİAŞ Phase-7 records use a different data SHA: {bad_sha} rows")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"Cannot inspect prior Phase-7 results: {exc}")
    else:
        report["existing_phase7_epias_rows"] = 0
        warnings.append("No Phase-7 result CSV found at this path; verify shared results before resume")

    report["pass"] = not errors
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--npy", type=Path, default=NPY_DEFAULT)
    p.add_argument("--runs", type=Path, default=RUNS_DEFAULT)
    p.add_argument("--regime", type=float, choices=(0.10, 0.25, 1.0), default=0.10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    result = audit(csv_path=args.csv, npy_path=args.npy, runs_path=args.runs,
                   regime=args.regime, epochs=args.epochs,
                   batch_size=args.batch_size)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
