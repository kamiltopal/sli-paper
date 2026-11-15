"""Launch phase-3 channel ablations in isolated subprocesses; resumable.

For an inexpensive first GPU smoke run:
  python -m scripts.asoc_channel_matrix --dry-run
  python -m scripts.asoc_channel_matrix --limit 1

The default *pilot* has 3 series x 1 training rep x 1 regime x 8 conditions.
No old results/ CSV is overwritten. Values are descriptive until independently
replicated backbone seeds, variants and model seeds are added.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.asoc_channel_controls import CONDITIONS, KEY_COLS, ROOT


def _ints(spec: str):
    items = tuple(int(x.strip()) for x in spec.split(","))
    if not items or any(x < 0 for x in items) or len(set(items)) != len(items):
        raise ValueError("Unique nonnegative integer values required")
    return items


def _floats(spec: str, allowed):
    items = tuple(float(x.strip()) for x in spec.split(","))
    if not items or not set(items).issubset(allowed) or len(set(items)) != len(items):
        raise ValueError(f"Only unique values from {allowed} are allowed")
    return items


def planned_cells(*, levels, series_seeds, variants, regimes, reps,
                  epochs, vae_epochs, batch_size, lr, context_bins):
    for level, variant, seed, regime, rep in itertools.product(
            levels, variants, series_seeds, regimes, reps):
        yield dict(file=f"om{level:.1f}_{variant}_s{seed}.npy",
                   regime=regime, rep=rep, pool_seed=seed,
                   epochs=epochs, vae_epochs=vae_epochs, batch_size=batch_size,
                   lr=lr, context_bins=context_bins)


def existing_conditions(out: Path):
    if not out.exists():
        return {}
    df = pd.read_csv(out)
    missing = set(KEY_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Existing CSV missing {sorted(missing)}")
    if df.duplicated(list(KEY_COLS)).any():
        raise ValueError("Duplicate result keys found; inspect CSV before rerunning")
    key_no_cond = [x for x in KEY_COLS if x != "condition"]
    return {
        key: set(g["condition"].astype(str))
        for key, g in df.groupby(key_no_cond, dropna=False)
    }


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
    p.add_argument("--context-bins", type=int, default=12)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/asoc_revision/channel_controls.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="maximum number of incomplete cells to run, 0 means all")
    args = p.parse_args()
    try:
        levels = _floats(args.levels, (0.2, 0.5, 0.8))
        regimes = _floats(args.regimes, (0.10, 0.25, 1.00))
        series_seeds, reps = _ints(args.series_seeds), _ints(args.reps)
        variants = tuple(v.strip() for v in args.variants.split(","))
        if not variants or len(set(variants)) != len(variants) or not set(variants).issubset({"plain", "seasonal_ar", "regime"}):
            raise ValueError("Unknown or duplicate series variants")
        if args.epochs < 1 or args.vae_epochs < 1 or args.batch_size < 1 or args.lr <= 0:
            raise ValueError("Positive training settings required")
        if args.context_bins < 1 or 96 % args.context_bins or args.gpu_index < 0 or args.limit < 0:
            raise ValueError("Invalid bins, gpu-index or limit")
    except ValueError as exc:
        p.error(str(exc))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    cells = list(planned_cells(
        levels=levels, series_seeds=series_seeds, variants=variants,
        regimes=regimes, reps=reps, epochs=args.epochs,
        vae_epochs=args.vae_epochs, batch_size=args.batch_size,
        lr=args.lr, context_bins=args.context_bins,
    ))
    done = existing_conditions(out)
    required = set(CONDITIONS)
    incomplete = [cell for cell in cells if done.get(tuple(cell[x] for x in KEY_COLS if x != "condition"), set()) != required]
    print(f"Plan: {len(cells)} paired cells x 8 conditions; "
          f"{len(incomplete)} incomplete; out={out}", flush=True)
    if args.dry_run:
        for cell in incomplete:
            print(cell)
        return
    if not incomplete:
        return
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA unavailable; pass --allow-cpu for CPU-only smoke test")
    selected = incomplete[:args.limit] if args.limit else incomplete
    for i, cell in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {cell}", flush=True)
        fname = cell["file"]
        if not (ROOT / "data" / "synthetic" / fname).exists():
            subprocess.run([sys.executable, "-m", "scripts.asoc_prepare_backbone",
                            "--file", fname], cwd=ROOT, check=True)
        cmd = [sys.executable, "-m", "scripts.asoc_channel_controls",
               "--file", fname, "--regime", str(cell["regime"]),
               "--rep", str(cell["rep"]), "--pool-seed", str(cell["pool_seed"]),
               "--epochs", str(args.epochs), "--vae-epochs", str(args.vae_epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--context-bins", str(args.context_bins),
               "--gpu-index", str(args.gpu_index), "--out", str(out)]
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Verified {len(selected)} paired cells; rerun this command to resume remaining work", flush=True)


if __name__ == "__main__":
    main()
