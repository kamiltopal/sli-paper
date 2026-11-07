"""Publication figures (final, post-review terminology).

Reads results/effects.csv (produced by `python -m src.sli`),
results/real_runs.csv and results/fidelity.csv.
Writes results/fig1..fig5 as .png (300 dpi) and .pdf.

Run:  python -m scripts.make_figures

NOTE: fig5 (and the architecture panels generally) require the
FULL-GRID effects.csv (1,944 cells, three architectures). If your
effects.csv has 648 cells it was built from the iTransformer-only
augmented.csv; rebuild it from the merged augmented.csv first.
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
RES = ROOT / "results"

# Colorblind-safe palette; generator identity consistent across figures.
# Labels use the manuscript's final terminology:
#   structure-informed (uses information beyond the training sample)
#   data-derived       (learned only from the training sample)
GEN_STYLE = {
    "seasonal": dict(color="#0072B2", label="Structural prior (oracle)"),
    "stl": dict(color="#0072B2", label="Seasonal-trend (STL-style)"),
    "exog": dict(color="#D55E00", label="Exogenous-conditioned"),
    "vae": dict(color="#009E73", label="Window-VAE"),
    "bootstrap": dict(color="#CC79A7", label="Block-bootstrap"),
}
REGIMES = [0.10, 0.25, 1.00]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1_vanishing(eff: pd.DataFrame):
    """Regime profiles per generator (median, IQR band)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for gen in ("seasonal", "vae", "bootstrap", "exog"):
        d = eff[eff.generator == gen]
        med = d.groupby("regime")["delta_pct"].median().reindex(REGIMES)
        q1 = d.groupby("regime")["delta_pct"].quantile(.25).reindex(REGIMES)
        q3 = d.groupby("regime")["delta_pct"].quantile(.75).reindex(REGIMES)
        st = GEN_STYLE[gen]
        ax.plot(REGIMES, med, "o-", color=st["color"], label=st["label"],
                lw=1.6)
        ax.fill_between(REGIMES, q1, q3, color=st["color"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(REGIMES)
    ax.set_xticklabels(["10%", "25%", "100%"])
    ax.set_xlabel("Real-data regime (fraction of training span)")
    ax.set_ylabel("ΔMSE% (augmented vs. baseline)")
    ax.set_title("Augmentation effects across data-availability regimes")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig1_vanishing_benefit")


def fig2_omega(eff: pd.DataFrame):
    """Predictability-band interaction."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bands = sorted(eff.target_omega.unique())
    width = 0.2
    for i, gen in enumerate(("seasonal", "vae", "bootstrap", "exog")):
        d = eff[eff.generator == gen]
        med = d.groupby("target_omega")["delta_pct"].median().reindex(bands)
        q1 = d.groupby("target_omega")["delta_pct"].quantile(.25).reindex(bands)
        q3 = d.groupby("target_omega")["delta_pct"].quantile(.75).reindex(bands)
        x = np.arange(len(bands)) + (i - 1.5) * width
        st = GEN_STYLE[gen]
        ax.bar(x, med, width, color=st["color"], label=st["label"])
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none",
                    ecolor="k", elinewidth=0.7, capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels([f"Ω = {b}" for b in bands])
    ax.set_ylabel("ΔMSE% (median, IQR)")
    ax.set_title("Augmentation effects by predictability band")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig2_omega_interaction")


def fig3_real(real: pd.DataFrame):
    """Real domains: scarce vs full data; EPİAŞ shaded."""
    b = (real[real.generator == "none"]
         .groupby(["domain", "regime", "arch"])["test_mse"].median()
         .rename("mse_base"))
    a = (real[real.generator != "none"]
         .groupby(["domain", "regime", "arch", "generator"])["test_mse"]
         .median().rename("mse_aug").reset_index().join(
             b, on=["domain", "regime", "arch"]))
    a["delta_pct"] = 100 * (a.mse_aug - a.mse_base) / a.mse_base
    med = a.groupby(["domain", "generator", "regime"])["delta_pct"].median()

    domains = ["weather", "electricity", "traffic", "epias"]
    labels = ["Weather\n(Ω=.37)", "Electricity\n(Ω=.44)",
              "Traffic\n(Ω=.71)", "EPİAŞ price\n(Ω=.31)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, regime, title in zip(axes, (0.10, 1.00),
                                 ("Scarce data (10%)", "Full data (100%)")):
        x = np.arange(len(domains))
        for i, gen in enumerate(("stl", "vae")):
            v = [med.get((d, gen, regime), np.nan) for d in domains]
            st = GEN_STYLE[gen]
            ax.bar(x + (i - .5) * .35, v, .35, color=st["color"],
                   label=st["label"] if regime == 0.10 else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.axvspan(2.55, 3.45, color="0.92", zorder=0)  # EPİAŞ shading
    axes[0].set_ylabel("ΔMSE% (median)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_ylim(-20, 60)
    axes[1].annotate("all cells > 0", xy=(3, 8), fontsize=7, ha="center",
                     style="italic")
    fig.suptitle("Real-domain augmentation effects across data-availability "
                 "regimes", fontsize=10)
    _save(fig, "fig3_real_domains")


def fig4_fidelity_null(eff: pd.DataFrame, fid: pd.DataFrame):
    """Exploratory null: similarity diagnostics vs realized effect."""
    m = eff.merge(fid.drop(columns=["openness"], errors="ignore"),
                  on=["file", "regime", "generator"])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for gen in ("seasonal", "vae", "bootstrap", "exog"):
        d = m[m.generator == gen]
        st = GEN_STYLE[gen]
        ax.scatter(d["f_probe"], d["delta_pct"], s=8, alpha=0.45,
                   color=st["color"], label=st["label"], edgecolors="none")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Forecasting-probe score")
    ax.set_ylabel("ΔMSE%")
    ax.set_title("Exploratory similarity diagnostics versus realized effect")
    rho_dd = (m[m.openness == 0][["delta_pct", "f_probe"]]
              .corr(method="spearman").iloc[0, 1])
    ax.annotate(f"data-derived class: Spearman ρ = +{rho_dd:.2f}",
                xy=(0.02, 0.96), xycoords="axes fraction", va="top",
                fontsize=7, style="italic")
    ax.set_ylim(-40, 80)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    _save(fig, "fig4_fidelity_null")


def fig5_architecture(eff: pd.DataFrame):
    """Architecture sensitivity to injected structure (both directions)."""
    d = eff.copy()
    d["aclass"] = np.where(d.arch.isin(["dlinear", "patchtst"]),
                           "Channel-independent", "Channel-mixing")
    CLS = {"Channel-independent": dict(color="#888888", ls="--", marker="s"),
           "Channel-mixing": dict(color="#0072B2", ls="-", marker="o")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    for ax, gen, title in zip(
            axes, ("seasonal", "exog"),
            ("Oracle structural prior", "Exogenous-conditioned")):
        sub_g = d[d.generator == gen]
        for cls, st in CLS.items():
            sub = sub_g[sub_g.aclass == cls]
            med = sub.groupby("regime")["delta_pct"].median().reindex(REGIMES)
            q1 = sub.groupby("regime")["delta_pct"].quantile(.25).reindex(REGIMES)
            q3 = sub.groupby("regime")["delta_pct"].quantile(.75).reindex(REGIMES)
            ax.plot(REGIMES, med, marker=st["marker"], ls=st["ls"],
                    color=st["color"], label=cls, lw=1.6)
            ax.fill_between(REGIMES, q1, q3, color=st["color"], alpha=0.12)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(REGIMES)
        ax.set_xticklabels(["10%", "25%", "100%"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Real-data regime")
    axes[0].set_ylabel("ΔMSE% (median, IQR)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Augmentation effects by model configuration", fontsize=10)
    _save(fig, "fig5_architecture")


def main():
    eff = pd.read_csv(RES / "effects.csv")
    real = pd.read_csv(RES / "real_runs.csv")
    fid = pd.read_csv(RES / "fidelity.csv")
    n_arch = eff["arch"].nunique()
    if n_arch < 3:
        print(f"WARNING: effects.csv contains {n_arch} architecture(s); "
              "fig5 and architecture panels need the full 3-architecture "
              "grid. Rebuild effects.csv from the merged augmented.csv "
              "via `python -m src.sli`.")
    fig1_vanishing(eff)
    fig2_omega(eff)
    fig3_real(real)
    fig4_fidelity_null(eff, fid)
    fig5_architecture(eff)


if __name__ == "__main__":
    main()
