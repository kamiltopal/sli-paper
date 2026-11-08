"""Phase 4 boundary-bootstrap resumable grid; separate process per cell.

Pilot: python -m scripts.asoc_bootstrap_matrix --levels 0.5 --limit 1 --gpu-index 0
Inspect: python -m scripts.asoc_bootstrap_matrix --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.asoc_bootstrap_controls import CONDITIONS, KEY_COLS, ROOT, DEFAULT_OUT, VERSION


def _ints(spec):
    values = tuple(int(x.strip()) for x in spec.split(","))
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise ValueError("Expected distinct nonnegative integers")
    return values


def _floats(spec, allowed):
    values = tuple(float(x.strip()) for x in spec.split(","))
    if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
        raise ValueError(f"Expected distinct values in {sorted(allowed)}")
    return values


def planned_cells(*, levels, series_seeds, variants, regimes, reps,
                  arch, epochs, batch_size, lr, block, trend_window, periods):
    for level, variant, seed, regime, rep in itertools.product(
            levels, variants, series_seeds, regimes, reps):
        yield dict(file=f"om{level:.1f}_{variant}_s{seed}.npy",
                   regime=regime, arch=arch, rep=rep, pool_seed=seed,
                   epochs=epochs, batch_size=batch_size, lr=lr, block=block,
                   trend_window=trend_window, periods=periods, version=VERSION)


def completed(path: Path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not set(KEY_COLS).issubset(df):
        raise ValueError(f"Incompatible output columns in {path}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate experiment keys; audit file before resuming")
    k = [c for c in KEY_COLS if c != "condition"]
    return {key: set(group.condition.astype(str)) for key, group in df.groupby(k, dropna=False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", default="0.2,0.5,0.8")
    p.add_argument("--series-seeds", default="0")
    p.add_argument("--variants", default="plain")
    p.add_argument("--regimes", default="0.10,1.00")
    p.add_argument("--reps", default="0")
    p.add_argument("--arch", choices=("dlinear", "patchtst"), default="dlinear")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--block", type=int, default=200)
    p.add_argument("--trend-window", type=int, default=169)
    p.add_argument("--periods", default="24,168")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    try:
        levels = _floats(args.levels, {0.2, 0.5, 0.8})
        series_seeds = _ints(args.series_seeds)
        regimes = _floats(args.regimes, {0.1, 0.25, 1.0})
        reps = _ints(args.reps)
        variants = tuple(x.strip() for x in args.variants.split(","))
        if not variants or len(set(variants)) != len(variants) or not set(variants).issubset({"plain", "seasonal_ar", "regime"}):
            raise ValueError("Invalid or repeated variant")
        periods = tuple(int(x.strip()) for x in args.periods.split(","))
        if (not periods or min(periods) < 2 or len(set(periods)) != len(periods)
                or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
                or args.block < 120 or args.trend_window < 3 or args.trend_window % 2 != 1
                or args.gpu_index < 0 or args.limit < 0):
            raise ValueError("Invalid grid parameters")
    except ValueError as exc:
        p.error(str(exc))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    cells = list(planned_cells(levels=levels, series_seeds=series_seeds,
        variants=variants, regimes=regimes, reps=reps, arch=args.arch,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        block=args.block, trend_window=args.trend_window,
        periods=",".join(map(str, periods))))
    k = [c for c in KEY_COLS if c != "condition"]
    known = completed(out)
    incomplete = [cell for cell in cells
                  if known.get(tuple(cell[f] for f in k), set()) != set(CONDITIONS)]
    print(f"Planned {len(cells)} cells × {len(CONDITIONS)} forecasters; "
          f"incomplete={len(incomplete)}; out={out}", flush=True)
    if args.dry_run:
        for cell in incomplete:
            print(cell)
        return
    if not incomplete:
        return
    from scripts.asoc_budget_controls import select_device
    select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
    selected = incomplete[:args.limit] if args.limit else incomplete
    for i, cell in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {cell}", flush=True)
        fname = cell["file"]
        if not (ROOT / "data" / "synthetic" / fname).is_file():
            subprocess.run([sys.executable, "-m", "scripts.asoc_prepare_backbone",
                            "--file", fname], cwd=ROOT, check=True)
        cmd = [sys.executable, "-m", "scripts.asoc_bootstrap_controls",
               "--file", fname, "--regime", str(cell["regime"]),
               "--rep", str(cell["rep"]), "--pool-seed", str(cell["pool_seed"]),
               "--arch", args.arch, "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--block", str(args.block), "--trend-window", str(args.trend_window),
               "--periods", args.periods, "--gpu-index", str(args.gpu_index),
               "--out", str(out)]
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Selected cells complete. Rerunning skips completed rows.", flush=True)


if __name__ == "__main__":
    main()
