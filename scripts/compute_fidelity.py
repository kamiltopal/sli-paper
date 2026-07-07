"""Compute pre-training fidelity (F_probe, F_spec) for every backbone
(series, regime, generator) pool -> results/fidelity.csv.

EXPLORATORY INSTRUMENT NOTE (paper §Results): F was formulated AFTER the
pre-registered rule failed; it uses no outcome data (only training
segments and pools) but its selection was informed by the failure
analysis. It is therefore reported as exploratory, evaluated with the
same frozen-half protocol.

Run:  python -m scripts.compute_fidelity          (VAE cells need torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import WindowConfig, make_windows  # noqa: E402
from src.generators import (GENERATORS, OPENNESS, probe_fidelity,  # noqa: E402
                            spectral_fidelity)
from scripts.run_augmented import training_segment, REGIMES  # noqa: E402

GENS = ("bootstrap", "vae", "seasonal", "exog")
WCFG = WindowConfig(lookback=96, horizon=24)
OUT = ROOT / "results" / "fidelity.csv"


def main():
    meta = pd.read_csv(ROOT / "results" / "backbone_validation.csv")
    rows = []
    for _, r in meta.iterrows():
        X = np.load(ROOT / "data/synthetic" / r["file"]).astype(np.float32)
        for regime in REGIMES:
            Xtr, ts = training_segment(X, regime)
            xr, yr = make_windows(Xtr, 0, len(Xtr), WCFG)
            rf = np.concatenate([xr[:, :, 0], yr], axis=1)
            for gen in GENS:
                xi, yi = GENERATORS[gen](Xtr, WCFG, ratio=1.0,
                                         seed=int(r["seed"]),
                                         spec=r, train_start=ts)
                sf = np.concatenate([xi[:, :, 0], yi], axis=1)
                rows.append({
                    "file": r["file"], "regime": regime, "generator": gen,
                    "openness": OPENNESS[gen],
                    "f_probe": probe_fidelity(xr, yr, xi, yi),
                    "f_spec": spectral_fidelity(rf, sf),
                })
        print(f"{r['file']} done")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
