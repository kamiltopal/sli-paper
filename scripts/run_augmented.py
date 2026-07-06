"""Augmented runs over the synthetic backbone (protocol §3.2).

For each (series, regime, generator): synthesize a window pool ONCE
(cached), then train each (arch, rep) with real+synthetic windows via
the train_aug hook. Diffing against baselines.csv gives ΔMSE%.

Staging (protocol §3.6): use --arch itransformer for the stage-1 pass;
add remaining architectures afterwards.

RESUMABLE like run_baselines. Run:
  python -m scripts.run_augmented --arch itransformer     # stage-1 pass
  python -m scripts.run_augmented                          # full
  python -m scripts.run_augmented --quick                  # sanity tour
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

from src.data import WindowConfig, prepare, chrono_split, standardize  # noqa: E402
from src.generators import GENERATORS, OPENNESS  # noqa: E402
from src.models import build  # noqa: E402
from src.train import TrainConfig, train_model  # noqa: E402

REGIMES = (0.10, 0.25, 1.00)
ARCHS = ("dlinear", "patchtst", "itransformer")
GENS = ("bootstrap", "vae", "seasonal", "exog")
REPS = range(5)
WCFG = WindowConfig(lookback=96, horizon=24)
AUG_RATIO = 1.0          # synthetic windows = 1x real window count (frozen)
OUT = ROOT / "results" / "augmented.csv"


def training_segment(X, regime):
    """Standardized training segment + its start index (mirrors prepare)."""
    from src.data import apply_regime
    T = len(X)
    t1, _ = chrono_split(T)
    ts = apply_regime(t1, regime, WCFG.lookback)
    Xs, _, _ = standardize(X, slice(ts, t1))
    return Xs[ts:t1], ts


def done_keys(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["file"], df["regime"], df["generator"],
                   df["arch"], df["rep"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arch", choices=ARCHS, default=None,
                    help="restrict to one architecture (staging)")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "results" / "backbone_validation.csv")
    if args.quick:
        meta = meta[meta["seed"] == 0]
    archs = [args.arch] if args.arch else list(ARCHS)
    reps = [0] if args.quick else list(REPS)

    done = done_keys(OUT)
    cache: dict = {}
    todo = [(r, rg, g, a, rp)
            for _, r in meta.iterrows() for rg in REGIMES
            for g in GENS for a in archs for rp in reps
            if (r["file"], rg, g, a, rp) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {OUT.name})")

    t0 = time.time()
    for i, (row, regime, gen, arch, rep) in enumerate(todo, 1):
        X = np.load(ROOT / "data/synthetic" / row["file"]).astype(np.float32)
        data = prepare(X, regime=regime, cfg=WCFG)

        ckey = (row["file"], regime, gen)
        if ckey not in cache:
            Xtr, ts = training_segment(X, regime)
            cache.clear()            # keep memory bounded: one pool at a time
            cache[ckey] = GENERATORS[gen](
                Xtr, WCFG, ratio=AUG_RATIO, seed=int(row["seed"]),
                spec=row, train_start=ts)
        data["train_aug"] = cache[ckey]

        res = train_model(build(arch, WCFG.lookback, WCFG.horizon),
                          data, TrainConfig(seed=rep))
        rec = {"file": row["file"], "target_omega": row["target_omega"],
               "achieved_omega": row["achieved_omega"],
               "pred_perm": row["pred_perm"], "variant": row["variant"],
               "series_seed": row["seed"], "regime": regime,
               "generator": gen, "openness": OPENNESS[gen],
               "arch": arch, "rep": rep, **res}
        pd.DataFrame([rec]).to_csv(OUT, mode="a", index=False,
                                   header=not OUT.exists())
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {row['file']:24s} reg={regime:.2f} "
              f"{gen:9s} O={OPENNESS[gen]} {arch:12s} rep={rep} "
              f"mse={res['test_mse']:.4f} ({el/i:.1f}s/run, "
              f"ETA {(len(todo)-i)*el/i/3600:.1f}h)")


if __name__ == "__main__":
    main()
