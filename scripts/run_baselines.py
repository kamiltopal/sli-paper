"""Baseline (no augmentation) grid over the synthetic backbone.

Grid: 54 series x 3 regimes x 3 architectures x 5 repetition seeds
    = 2,430 runs, minutes-scale each on the smallest models.

This produces the E_base landscape (protocol §3.4): per-cell baseline
errors from which headroom is computed. Augmentation runs (stage 2b)
will diff against these numbers.

RESUMABLE: results are appended to results/baselines.csv; completed
(file, regime, arch, rep) rows are skipped on restart. Safe to Ctrl-C.

Run:  python -m scripts.run_baselines            # full grid
      python -m scripts.run_baselines --quick    # 1 seed-series, 1 rep
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import WindowConfig, prepare  # noqa: E402
from src.models import build  # noqa: E402
from src.train import TrainConfig, train_model  # noqa: E402

REGIMES = (0.10, 0.25, 1.00)
ARCHS = ("dlinear", "patchtst", "itransformer")
REPS = range(5)
WCFG = WindowConfig(lookback=96, horizon=24)
OUT = ROOT / "results" / "baselines.csv"


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["file"], df["regime"], df["arch"], df["rep"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "results" / "backbone_validation.csv")
    if args.quick:
        meta = meta[meta["seed"] == 0]
    reps = [0] if args.quick else list(REPS)

    done = done_keys(OUT)
    todo = [(r, rg, a, rp)
            for _, r in meta.iterrows()
            for rg in REGIMES for a in ARCHS for rp in reps
            if (r["file"], rg, a, rp) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {OUT.name})")

    t0 = time.time()
    for i, (row, regime, arch, rep) in enumerate(todo, 1):
        X = np.load(ROOT / "data/synthetic" / row["file"]).astype(np.float32)
        data = prepare(X, regime=regime, cfg=WCFG)
        res = train_model(build(arch, WCFG.lookback, WCFG.horizon),
                          data, TrainConfig(seed=rep))
        rec = {
            "file": row["file"], "target_omega": row["target_omega"],
            "achieved_omega": row["achieved_omega"],
            "pred_perm": row["pred_perm"], "variant": row["variant"],
            "series_seed": row["seed"], "regime": regime, "arch": arch,
            "rep": rep, **res,
        }
        pd.DataFrame([rec]).to_csv(OUT, mode="a", index=False,
                                   header=not OUT.exists())
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {row['file']:26s} reg={regime:.2f} "
              f"{arch:12s} rep={rep}  mse={res['test_mse']:.4f}  "
              f"({el/i:.1f}s/run, ETA {(len(todo)-i)*el/i/3600:.1f}h)")


if __name__ == "__main__":
    main()
