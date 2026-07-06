"""Stage-1 validation (protocol §3.6, stage 1 entry gate).

Generates the frozen v1 backbone grid across all variants, measures
achieved Omega and the permutation proxy, checks calibration error and
rank-consistency between the two proxies, writes:

  results/backbone_validation.csv
  results/backbone_validation.png
  data/synthetic/*.npy          (the actual series, reproducible)

Run:  python -m scripts.validate_backbone
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signals import SignalSpec, generate_multichannel, spec_to_row  # noqa: E402
from src.measures import permutation_predictability  # noqa: E402

LEVELS = (0.2, 0.5, 0.8)
SEEDS = range(6)
VARIANTS = ("plain", "seasonal_ar", "regime")
OUT_DATA = ROOT / "data" / "synthetic"
OUT_RES = ROOT / "results"


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    rows = []
    for om in LEVELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                spec = SignalSpec(target_omega=om, seed=seed, variant=variant)
                X, spec = generate_multichannel(spec); x = X[:, 0]
                fname = f"om{om:.1f}_{variant}_s{seed}.npy"
                np.save(OUT_DATA / fname, X.astype(np.float32))
                row = spec_to_row(spec)
                row["pred_perm"] = permutation_predictability(x)
                row["calib_err"] = abs(spec.achieved_omega - om)
                row["file"] = fname
                rows.append(row)
                print(f"  Ω*={om:.1f} {variant:11s} seed={seed}  "
                      f"Ω={spec.achieved_omega:.3f}  "
                      f"perm={row['pred_perm']:.3f}  w={spec.mix_weight:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "backbone_validation.csv", index=False)

    # --- acceptance checks -------------------------------------------------
    max_err = df["calib_err"].max()
    # rank consistency: Spearman rho between the two proxies
    rho = df[["achieved_omega", "pred_perm"]].corr(method="spearman").iloc[0, 1]
    print(f"\nmax calibration error : {max_err:.4f} (gate: < 0.02)")
    print(f"Spearman rho (Ω vs perm): {rho:.3f} (gate: > 0.8)")
    ok = (max_err < 0.02) and (rho > 0.8)
    print("STAGE-1 GATE:", "PASS" if ok else "FAIL")

    # --- diagnostic plot ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, variant in zip(axes, VARIANTS):
        sub = df[df["variant"] == variant]
        ax.scatter(sub["target_omega"], sub["achieved_omega"],
                   alpha=0.7, label="achieved Ω")
        ax.scatter(sub["target_omega"], sub["pred_perm"],
                   alpha=0.7, marker="x", label="perm proxy")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(variant)
        ax.set_xlabel("target Ω")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("measured value")
    fig.suptitle("Backbone calibration: target vs. achieved predictability")
    fig.tight_layout()
    fig.savefig(OUT_RES / "backbone_validation.png", dpi=120)
    print(f"\nwrote {OUT_RES/'backbone_validation.csv'}")
    print(f"wrote {OUT_RES/'backbone_validation.png'}")


if __name__ == "__main__":
    main()
