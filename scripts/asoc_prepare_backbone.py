"""Regenerate one controlled series without changing archived result CSV files.

Uses results/backbone_validation.csv specifications. Does not overwrite an
existing .npy; neither calls validate_backbone nor modifies its output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="om0.5_plain_s0.npy")
    args = p.parse_args()
    from src.signals import SignalSpec, generate_multichannel

    table = ROOT / "results" / "backbone_validation.csv"
    if not table.exists():
        p.error(f"Missing frozen metadata: {table}")
    df = pd.read_csv(table)
    found = df.loc[df["file"] == args.file]
    if len(found) != 1:
        p.error(f"Expected exactly one matching metadata row for {args.file}; found {len(found)}")
    row = found.iloc[0]
    out = ROOT / "data" / "synthetic" / args.file
    if out.exists():
        print(f"Already exists, leaving unchanged: {out}")
        return
    spec = SignalSpec(
        target_omega=float(row["target_omega"]),
        seed=int(row["seed"]),
        variant=str(row["variant"]),
        n_points=int(row["n_points"]),
        n_components=int(row["n_components"]),
        ar_coef=float(row.get("ar_coef", 0.7)),
        regime_period=int(row.get("regime_period", 4000)),
    )
    X, computed = generate_multichannel(spec)
    for key in ("achieved_omega", "mix_weight", "concentration"):
        if not np.isclose(getattr(computed, key), float(row[key]), atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Regenerated {key} differs from archived metadata; "
                               "do not use this series until the code/provenance is checked")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, X.astype(np.float32))
    print(f"Wrote {out} from frozen metadata; achieved Ω={computed.achieved_omega:.4f}")


if __name__ == "__main__":
    main()
