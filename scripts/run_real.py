"""Real-domain runs (protocol §3.6 / methodology §3.6).

Reduced, hypothesis-targeted grid. Generator choices are fixed by the
backbone analysis: best closed = window-VAE, best open = seasonal-trend
(STL-style with a priori periods). Baseline included as generator "none".

Grid: 4 domains x 2 regimes (0.10, 1.00) x 3 gens (none, vae, stl)
      x 2 archs (dlinear, itransformer) x 3 reps = 144 runs.

A priori period sets (from sampling interval, never from outcomes):
  hourly domains  -> (24, 168)
  weather (10-min)-> (144, 1008)

Run:  python -m scripts.run_real
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
           "electricity": (24, 168), "epias": (24, 168)}
REGIMES = (0.10, 1.00)
GENS = ("none", "vae", "stl")
ARCHS = ("dlinear", "itransformer")
REPS = range(3)
WCFG = WindowConfig(lookback=96, horizon=24)
OUT = ROOT / "results" / "real_runs.csv"


def done_keys(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["domain"], df["regime"], df["generator"],
                   df["arch"], df["rep"]))


def main():
    done = done_keys(OUT)
    todo = [(d, rg, g, a, rp) for d in DOMAINS for rg in REGIMES
            for g in GENS for a in ARCHS for rp in REPS
            if (d, rg, g, a, rp) not in done]
    print(f"{len(todo)} runs to do ({len(done)} done)")

    cache: dict = {}
    t0 = time.time()
    for i, (dom, regime, gen, arch, rep) in enumerate(todo, 1):
        X = np.load(ROOT / "data" / "real" / f"{dom}.npy").astype(np.float32)
        data = prepare(X, regime=regime, cfg=WCFG)
        if gen != "none":
            ckey = (dom, regime, gen)
            if ckey not in cache:
                Xtr, _ = training_segment(X, regime)
                cache.clear()
                cache[ckey] = GENERATORS[gen](
                    Xtr, WCFG, ratio=1.0, seed=0, periods=DOMAINS[dom])
            data["train_aug"] = cache[ckey]

        res = train_model(build(arch, WCFG.lookback, WCFG.horizon),
                          data, TrainConfig(seed=rep))
        rec = {"domain": dom, "regime": regime, "generator": gen,
               "openness": (None if gen == "none" else OPENNESS[gen]),
               "arch": arch, "rep": rep, **res}
        pd.DataFrame([rec]).to_csv(OUT, mode="a", index=False,
                                   header=not OUT.exists())
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {dom:12s} reg={regime:.2f} {gen:5s} "
              f"{arch:12s} rep={rep} mse={res['test_mse']:.4f} "
              f"({el/i:.1f}s/run)")


if __name__ == "__main__":
    main()
