"""Targeted rerun of the covariate-selection fix (27 runs) + merge.

Reruns ONLY the scarce-regime (10%) iTransformer cells of the three
benchmark domains, using the *_scarce.npy arrays produced by
scripts.prepare_real_scarce (slice-internal covariate selection).
DLinear cells and all full-data cells are unaffected by construction;
EPİAŞ is excluded (a-priori calendar covariates).

Step 1:  python -m scripts.prepare_real_scarce
Step 2:  python -m scripts.rerun_scarce_itransformer
Step 3:  python -m scripts.rerun_scarce_itransformer --merge
         (backs up results/real_runs.csv, replaces the 27 affected rows
          with the fixed ones, keyed identically so downstream analysis
          scripts run unchanged)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import WindowConfig, prepare  # noqa: E402
from src.generators import GENERATORS, OPENNESS  # noqa: E402
from src.models import build  # noqa: E402
from src.train import TrainConfig, train_model  # noqa: E402
from scripts.run_augmented import training_segment  # noqa: E402

DOMAINS = {"weather": (144, 1008), "traffic": (24, 168),
           "electricity": (24, 168)}
REGIME = 0.10
GENS = ("none", "vae", "stl")
ARCH = "itransformer"
REPS = range(3)
WCFG = WindowConfig(lookback=96, horizon=24)
FIX = ROOT / "results" / "real_runs_scarce_fix.csv"
MAIN = ROOT / "results" / "real_runs.csv"


def run():
    done = set()
    if FIX.exists():
        d = pd.read_csv(FIX)
        done = set(zip(d["domain"], d["generator"], d["rep"]))
    todo = [(dom, g, r) for dom in DOMAINS for g in GENS for r in REPS
            if (dom, g, r) not in done]
    print(f"{len(todo)} runs to do ({len(done)} done)")
    cache = {}
    t0 = time.time()
    for i, (dom, gen, rep) in enumerate(todo, 1):
        X = np.load(ROOT / "data/real" / f"{dom}_scarce.npy").astype(np.float32)
        data = prepare(X, regime=REGIME, cfg=WCFG)
        if gen != "none":
            key = (dom, gen)
            if key not in cache:
                Xtr, _ = training_segment(X, REGIME)
                cache.clear()
                cache[key] = GENERATORS[gen](Xtr, WCFG, ratio=1.0, seed=0,
                                             periods=DOMAINS[dom])
            data["train_aug"] = cache[key]
        res = train_model(build(ARCH, WCFG.lookback, WCFG.horizon),
                          data, TrainConfig(seed=rep))
        rec = {"domain": dom, "regime": REGIME, "generator": gen,
               "openness": (None if gen == "none" else OPENNESS[gen]),
               "arch": ARCH, "rep": rep, **res}
        pd.DataFrame([rec]).to_csv(FIX, mode="a", index=False,
                                   header=not FIX.exists())
        print(f"[{i}/{len(todo)}] {dom:12s} {gen:5s} rep={rep} "
              f"mse={res['test_mse']:.4f} ({(time.time()-t0)/i:.1f}s/run)")


def merge():
    main_df = pd.read_csv(MAIN)
    fix_df = pd.read_csv(FIX)
    assert len(fix_df) == 27, f"fix file has {len(fix_df)} rows, expected 27"
    backup = MAIN.with_suffix(".pre_covfix.csv")
    main_df.to_csv(backup, index=False)
    mask = (main_df["domain"].isin(DOMAINS) & (main_df["regime"] == REGIME)
            & (main_df["arch"] == ARCH))
    print(f"replacing {mask.sum()} rows (expected 27); backup -> {backup.name}")
    merged = pd.concat([main_df[~mask], fix_df], ignore_index=True)
    merged.to_csv(MAIN, index=False)
    print(f"real_runs.csv updated ({len(merged)} rows). "
          "Now rerun: python -m scripts.paper_stats  and  "
          "python -m scripts.make_figures")


if __name__ == "__main__":
    merge() if "--merge" in sys.argv else run()
