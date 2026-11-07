"""Resumable Phase 3B grid runner (one subprocess per five-condition cell).

Examples:
  python -m scripts.asoc_joint_matrix --dry-run
  python -m scripts.asoc_joint_matrix --levels 0.5 --limit 1 --gpu-index 0
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.asoc_joint_controls import CONDITIONS, KEY_COLS, ROOT, DEFAULT_OUT, EXPERIMENT_VERSION


def _ints(spec):
    vals = tuple(int(s.strip()) for s in spec.split(","))
    if not vals or min(vals) < 0 or len(set(vals)) != len(vals):
        raise ValueError("Expected unique nonnegative integers")
    return vals


def _floats(spec, allowed):
    vals = tuple(float(s.strip()) for s in spec.split(","))
    if not vals or not set(vals).issubset(allowed) or len(set(vals)) != len(vals):
        raise ValueError(f"Expected unique values from {sorted(allowed)}")
    return vals


def planned_cells(*, levels, series_seeds, variants, regimes, reps,
                  epochs, vae_epochs, batch_size, lr, latent, hidden,
                  horizon_weight):
    for level, variant, seed, regime, rep in itertools.product(
            levels, variants, series_seeds, regimes, reps):
        yield dict(file=f"om{level:.1f}_{variant}_s{seed}.npy",
                   regime=regime, rep=rep, pool_seed=seed, epochs=epochs,
                   vae_epochs=vae_epochs, batch_size=batch_size, lr=lr,
                   latent=latent, hidden=hidden, horizon_weight=horizon_weight,
                   experiment_version=EXPERIMENT_VERSION)


def existing_conditions(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    missing = set(KEY_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Existing output incompatible with Phase 3B: {sorted(missing)}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate rows exist in Phase 3B output; fix before resuming")
    group_columns = [c for c in KEY_COLS if c != "condition"]
    return {key: set(group["condition"].astype(str))
            for key, group in df.groupby(group_columns, dropna=False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", default="0.2,0.5,0.8")
    p.add_argument("--series-seeds", default="0")
    p.add_argument("--variants", default="plain")
    p.add_argument("--regimes", default="0.10")
    p.add_argument("--reps", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon-weight", type=float, default=0.5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Max number of incomplete cells, 0=all")
    args = p.parse_args()
    try:
        levels = _floats(args.levels, {0.2, 0.5, 0.8})
        regimes = _floats(args.regimes, {0.1, 0.25, 1.0})
        series_seeds = _ints(args.series_seeds)
        reps = _ints(args.reps)
        variants = tuple(s.strip() for s in args.variants.split(","))
        if not variants or len(set(variants)) != len(variants) or not set(variants).issubset({"plain", "seasonal_ar", "regime"}):
            raise ValueError("Unknown or repeated series variant")
        if (args.epochs < 1 or args.vae_epochs < 1 or args.batch_size < 1 or args.lr <= 0
                or args.latent < 1 or args.hidden < 1 or not 0 < args.horizon_weight < 1
                or args.gpu_index < 0 or args.limit < 0):
            raise ValueError("Invalid training, GPU, or limit setting")
    except ValueError as err:
        p.error(str(err))

    out = args.out if args.out.is_absolute() else ROOT / args.out
    cells = list(planned_cells(
        levels=levels, series_seeds=series_seeds, variants=variants,
        regimes=regimes, reps=reps, epochs=args.epochs,
        vae_epochs=args.vae_epochs, batch_size=args.batch_size,
        lr=args.lr, latent=args.latent, hidden=args.hidden,
        horizon_weight=args.horizon_weight))
    done = existing_conditions(out)
    keys_without_condition = [key for key in KEY_COLS if key != "condition"]
    required = set(CONDITIONS)
    incomplete = [cell for cell in cells if done.get(
        tuple(cell[key] for key in keys_without_condition), set()) != required]
    print(f"Planned {len(cells)} cells × {len(CONDITIONS)} forecasters; "
          f"incomplete={len(incomplete)}; out={out}", flush=True)
    if args.dry_run:
        for cell in incomplete:
            print(cell)
        return
    if not incomplete:
        return
    # Preflight GPU check BEFORE preparing files or starting any costly work.
    from scripts.asoc_budget_controls import select_device
    select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
    selected = incomplete[:args.limit] if args.limit else incomplete
    for number, cell in enumerate(selected, start=1):
        print(f"[{number}/{len(selected)}] {cell}", flush=True)
        filename = cell["file"]
        if not (ROOT / "data" / "synthetic" / filename).is_file():
            subprocess.run([sys.executable, "-m", "scripts.asoc_prepare_backbone",
                            "--file", filename], cwd=ROOT, check=True)
        cmd = [sys.executable, "-m", "scripts.asoc_joint_controls",
               "--file", filename, "--regime", str(cell["regime"]),
               "--rep", str(cell["rep"]), "--pool-seed", str(cell["pool_seed"]),
               "--epochs", str(args.epochs), "--vae-epochs", str(args.vae_epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--latent", str(args.latent), "--hidden", str(args.hidden),
               "--horizon-weight", str(args.horizon_weight),
               "--gpu-index", str(args.gpu_index), "--out", str(out)]
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Finished selected cells; rerun to skip completed rows.", flush=True)


if __name__ == "__main__":
    main()
