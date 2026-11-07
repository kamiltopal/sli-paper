"""Run a predeclared, resumable ASOC step-budget control matrix on CUDA.

Requires phase-1 scripts.asoc_budget_controls and scripts.asoc_prepare_backbone.
Archives are never overwritten; outputs go to results/asoc_revision/.

Quick dry run:
    python -m scripts.asoc_budget_matrix --dry-run

Primary *screening* matrix (3 Fourier series x 3 training seeds x 2 regimes):
    python -m scripts.asoc_budget_matrix --gpu-index 0

Independently replicated extension (9 series x 3 reps x 3 regimes):
    python -m scripts.asoc_budget_matrix --series-seeds 0,1,2 --regimes 0.10,0.25,1.00

Convergence check (same controls, 20 epochs):
    python -m scripts.asoc_budget_matrix --levels 0.5 --series-seeds 0 \\
        --reps 0,1,2 --regimes 0.10 --epochs 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONDITIONS = frozenset(
    ("epoch_baseline", "step_baseline", "duplicate_real", "vae_augmented")
)
DEFAULT_OUT = Path("results/asoc_revision/budget_matrix_v2.csv")
KEY_FIELDS = ("file", "regime", "arch", "rep", "pool_seed", "input_mode",
              "epochs", "batch_size", "lr", "vae_epochs")


def _ints(spec: str) -> tuple[int, ...]:
    items = tuple(int(s.strip()) for s in spec.split(","))
    if not items or any(s < 0 for s in items) or len(set(items)) != len(items):
        raise ValueError(f"Expected unique nonnegative integers: {spec}")
    return items


def _floats(spec: str, *, allowed: tuple[float, ...]) -> tuple[float, ...]:
    items = tuple(float(s.strip()) for s in spec.split(","))
    if not items or len(set(items)) != len(items) or not set(items).issubset(allowed):
        raise ValueError(f"Allowed unique values are {allowed}: {spec}")
    return items


def planned_cells(*, levels, series_seeds, variants, regimes, reps, arch, epochs,
                  vae_epochs, batch_size, lr, input_mode):
    """Deterministic enumeration; model-seed and series-seed are independent axes."""
    for level, variant, s, regime, rep in itertools.product(
            levels, variants, series_seeds, regimes, reps):
        fname = f"om{level:.1f}_{variant}_s{s}.npy"
        yield dict(file=fname, regime=regime, arch=arch, rep=rep,
                   pool_seed=s, input_mode=input_mode, epochs=epochs,
                   batch_size=batch_size, lr=lr, vae_epochs=vae_epochs)


def completed_conditions(path: Path) -> dict[tuple, set[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = set(KEY_FIELDS) | {"condition"}
    if not required.issubset(df.columns):
        raise ValueError(f"Output {path} missing columns {sorted(required - set(df.columns))}")
    if df.duplicated(list(KEY_FIELDS) + ["condition"]).any():
        raise ValueError(f"Output {path} contains duplicate result keys; audit before resuming")
    groups = {}
    for key, group in df.groupby(list(KEY_FIELDS), dropna=False):
        groups[tuple(key)] = set(group["condition"].astype(str))
    return groups


def _key(cell: dict) -> tuple:
    return tuple(cell[field] for field in KEY_FIELDS)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", default="0.2,0.5,0.8")
    p.add_argument("--series-seeds", default="0")
    p.add_argument("--variants", default="plain",
                   help="comma-separated: plain,seasonal_ar,regime")
    p.add_argument("--regimes", default="0.10,1.00")
    p.add_argument("--reps", default="0,1,2")
    p.add_argument("--arch", choices=("dlinear", "patchtst", "itransformer"), default="dlinear")
    p.add_argument("--input-mode", choices=("target", "all"), default="target")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true", help="only for smoke tests")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0,
                   help="run at most this many incomplete cells (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        levels = _floats(args.levels, allowed=(0.2, 0.5, 0.8))
        regimes = _floats(args.regimes, allowed=(0.10, 0.25, 1.0))
        seeds, reps = _ints(args.series_seeds), _ints(args.reps)
        variants = tuple(x.strip() for x in args.variants.split(","))
        if not variants or any(x not in {"plain", "seasonal_ar", "regime"} for x in variants):
            raise ValueError("--variants must select plain,seasonal_ar,regime")
        if len(set(variants)) != len(variants):
            raise ValueError("Duplicate --variants")
        if args.epochs < 1 or args.vae_epochs < 1 or args.batch_size < 1 or args.lr <= 0:
            raise ValueError("epochs, vae-epochs, batch-size and lr must be positive")
        if args.limit < 0 or args.gpu_index < 0:
            raise ValueError("limit and gpu-index must be nonnegative")
    except ValueError as exc:
        p.error(str(exc))

    out = args.out if args.out.is_absolute() else ROOT / args.out
    cells = list(planned_cells(
        levels=levels, series_seeds=seeds, variants=variants, regimes=regimes,
        reps=reps, arch=args.arch, epochs=args.epochs,
        vae_epochs=args.vae_epochs, batch_size=args.batch_size,
        lr=args.lr, input_mode=args.input_mode))
    done = completed_conditions(out)
    incomplete = [cell for cell in cells if done.get(_key(cell), set()) != REQUIRED_CONDITIONS]
    print(f"PLAN: {len(cells)} cells x 4 conditions; {len(incomplete)} incomplete; "
          f"{len(cells) - len(incomplete)} complete; out={out}", flush=True)
    for i, cell in enumerate(incomplete, 1):
        if args.dry_run:
            print(f"  {i:03d} {cell['file']} regime={cell['regime']:.2f} "
                  f"rep={cell['rep']} epochs={cell['epochs']}")
    if args.dry_run or not incomplete:
        return
    if args.input_mode != "target":
        p.error("For confirmatory budget runs, use --input-mode target; "
                "multivariate VAE auxiliary pairing is currently confounded")

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA unavailable; require GPU for scientific matrix (or --allow-cpu for smoke only)")
    if torch.cuda.is_available() and args.gpu_index >= torch.cuda.device_count():
        p.error(f"Invalid GPU index {args.gpu_index}; {torch.cuda.device_count()} GPU(s) visible")
    helper_help = subprocess.run(
        [sys.executable, "-m", "scripts.asoc_budget_controls", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    supports_gpu_flag = "--gpu-index" in helper_help
    supports_cpu_flag = "--allow-cpu" in helper_help
    if args.gpu_index and not supports_gpu_flag:
        p.error("Phase-1 runner does not support --gpu-index; apply its GPU update patch")
    if args.allow_cpu and not supports_cpu_flag:
        p.error("Phase-1 runner does not support --allow-cpu; apply its GPU update patch")

    selected = incomplete[:args.limit] if args.limit else incomplete
    for i, cell in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {json.dumps(cell, sort_keys=True)}", flush=True)
        fname = cell["file"]
        if not (ROOT / "data" / "synthetic" / fname).is_file():
            _run([sys.executable, "-m", "scripts.asoc_prepare_backbone", "--file", fname])
        cmd = [sys.executable, "-m", "scripts.asoc_budget_controls",
               "--file", fname, "--regime", str(cell["regime"]),
               "--arch", cell["arch"], "--rep", str(cell["rep"]),
               "--pool-seed", str(cell["pool_seed"]),
               "--input-mode", cell["input_mode"],
               "--epochs", str(cell["epochs"]),
               "--vae-epochs", str(cell["vae_epochs"]),
               "--batch-size", str(cell["batch_size"]),
               "--lr", str(cell["lr"]), "--out", str(out)]
        if supports_gpu_flag:
            cmd += ["--gpu-index", str(args.gpu_index)]
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        _run(cmd)
    print(f"Completed this invocation: {len(selected)} cell(s); report with "
          "python -m scripts.asoc_budget_report", flush=True)


if __name__ == "__main__":
    main()
