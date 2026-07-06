"""SLI computation and hypothesis analysis (protocol §3.4–§3.5).

Usage after both grids have data:
  python -m src.sli            # prints H1/H2 summaries + SLI rule scores

Headroom is ORDINAL: E_base normalized against the empirical best error
within the same Ω band — never an absolute floor claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(root=ROOT):
    base = pd.read_csv(root / "results" / "baselines.csv")
    aug = pd.read_csv(root / "results" / "augmented.csv")
    return base, aug


def headroom_table(base: pd.DataFrame) -> pd.DataFrame:
    """Per (file, regime, arch): median baseline MSE, normalized to the
    empirical best within the same target-Ω band -> ordinal headroom."""
    g = (base.groupby(["file", "target_omega", "achieved_omega",
                       "pred_perm", "regime", "arch"])["test_mse"]
         .median().reset_index().rename(columns={"test_mse": "e_base"}))
    best = g.groupby("target_omega")["e_base"].transform("min")
    g["headroom"] = (g["e_base"] - best) / best          # 0 = at band best
    return g


def effects_table(base: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    """ΔMSE% per cell (median over reps), joined with headroom."""
    b = (base.groupby(["file", "regime", "arch"])["test_mse"]
         .median().rename("mse_base"))
    a = (aug.groupby(["file", "regime", "arch", "generator",
                      "openness", "target_omega"])["test_mse"]
         .median().rename("mse_aug").reset_index())
    df = a.join(b, on=["file", "regime", "arch"])
    df["delta_pct"] = 100 * (df["mse_aug"] - df["mse_base"]) / df["mse_base"]
    hr = headroom_table(base)[["file", "regime", "arch", "headroom",
                               "pred_perm"]]
    return df.merge(hr, on=["file", "regime", "arch"])


def sli_rule(df: pd.DataFrame, h_thresh: float) -> pd.Series:
    """Pre-registered rule: augment ⇔ [H high ∧ (O=1 ∨ data-scarce)]."""
    scarce = df["regime"] < 1.0
    return (df["headroom"] > h_thresh) & ((df["openness"] == 1) | scarce)


def evaluate_rule(df: pd.DataFrame, seed: int = 0):
    """Calibrate h_thresh on half the seed-series, test frozen on the rest
    + report vs naive rules (protocol §3.4)."""
    rng = np.random.default_rng(seed)
    files = df["file"].unique()
    cal_files = set(rng.choice(files, size=len(files) // 2, replace=False))
    cal, test = df[df["file"].isin(cal_files)], df[~df["file"].isin(cal_files)]

    helped = lambda d: d["delta_pct"] < 0
    # calibrate threshold by sign-accuracy on the calibration half
    grid = np.quantile(cal["headroom"], np.linspace(0.1, 0.9, 17))
    accs = [(np.mean(sli_rule(cal, t) == helped(cal)), t) for t in grid]
    best_acc, h_star = max(accs)

    y = helped(test)
    rules = {
        "SLI (frozen)": sli_rule(test, h_star),
        "always": pd.Series(True, index=test.index),
        "never": pd.Series(False, index=test.index),
        "if-scarce": test["regime"] < 1.0,
    }
    print(f"calibrated h* = {h_star:.3f} (cal acc {best_acc:.3f})")
    for name, pred in rules.items():
        acc = np.mean(pred == y)
        tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        print(f"  {name:14s} acc={acc:.3f}  F1={f1:.3f}")
    return h_star


def main():
    base, aug = load()
    df = effects_table(base, aug)
    df.to_csv(ROOT / "results" / "effects.csv", index=False)
    print(f"effects table: {len(df)} cells -> results/effects.csv\n")

    print("— H1 (ceiling): mean Δ% by Ω band, open generators, full data —")
    h1 = df[(df.openness == 1) & (df.regime == 1.0)]
    print(h1.groupby("target_omega")["delta_pct"].mean().round(2), "\n")

    print("— H2 (closed = regularizer): mean Δ% by regime, closed gens —")
    h2 = df[df.openness == 0]
    print(h2.groupby(["target_omega", "regime"])["delta_pct"]
          .mean().round(2).unstack(), "\n")

    print("— H3 (mixing advantage): mean Δ% open gens, by arch —")
    h3 = df[df.openness == 1]
    print(h3.groupby("arch")["delta_pct"].mean().round(2), "\n")

    print("— H4 (SLI rule vs naive) —")
    evaluate_rule(df)


if __name__ == "__main__":
    main()
