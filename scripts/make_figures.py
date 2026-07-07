"""Publication figures (Results section). Reproducible from results/*.csv.

Run:  python -m scripts.make_figures
Writes results/fig1..fig4 as .png (300 dpi) and .pdf.
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

# colorblind-safe, consistent generator identity across all figures
GEN_STYLE = {
    "seasonal": dict(color="#0072B2", label="Structural prior (open)"),
    "stl": dict(color="#0072B2", label="Seasonal-trend STL (open)"),
    "exog": dict(color="#D55E00", label="Exogenous-cond. (open, low-fid.)"),
    "vae": dict(color="#009E73", label="Window-VAE (closed)"),
    "bootstrap": dict(color="#CC79A7", label="Block-bootstrap (closed)"),
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
    """Headline: ΔMSE% vs data regime per generator (backbone, iTransformer).
    Median with IQR band over 216 cells/generator."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for gen in ("seasonal", "vae", "bootstrap", "exog"):
        d = eff[eff.generator == gen]
        med = d.groupby("regime")["delta_pct"].median().reindex(REGIMES)
        q1 = d.groupby("regime")["delta_pct"].quantile(.25).reindex(REGIMES)
        q3 = d.groupby("regime")["delta_pct"].quantile(.75).reindex(REGIMES)
        st = GEN_STYLE[gen]
        ax.plot(REGIMES, med, "o-", color=st["color"], label=st["label"], lw=1.6)
        ax.fill_between(REGIMES, q1, q3, color=st["color"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(REGIMES)
    ax.set_xticklabels(["10%", "25%", "100%"])
    ax.set_xlabel("Real-data regime (fraction of training span)")
    ax.set_ylabel("ΔMSE% (augmented vs. baseline)")
    ax.set_title("Augmentation effect vanishes or reverses with data abundance")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig1_vanishing_benefit")


def fig2_omega(eff: pd.DataFrame):
    """Ω-band × generator interaction (backbone)."""
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
    ax.set_title("Effect by intrinsic predictability band")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig2_omega_interaction")


def fig3_real(real: pd.DataFrame):
    """Real domains: scarce vs full, STL and VAE; EPİAŞ highlighted."""
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
    axes[1].set_ylim(-20, 15)
    axes[1].annotate("all cells > 0", xy=(3, 8), fontsize=7, ha="center",
                     style="italic")
    fig.suptitle("Real domains: benefits under scarcity, harm at abundance",
                 fontsize=10)
    _save(fig, "fig3_real_domains")


def fig4_fidelity_null(eff: pd.DataFrame, fid: pd.DataFrame):
    """Honest negative: F_probe does not gate benefit; wrong sign in
    closed class."""
    m = eff.merge(fid, on=["file", "regime", "generator"],
                  suffixes=("", "_f"))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for gen in ("seasonal", "vae", "bootstrap", "exog"):
        d = m[m.generator == gen]
        st = GEN_STYLE[gen]
        ax.scatter(d["f_probe"], d["delta_pct"], s=8, alpha=0.45,
                   color=st["color"], label=st["label"], edgecolors="none")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Pre-training probe fidelity  F_probe")
    ax.set_ylabel("ΔMSE%")
    ax.set_title("Fidelity does not gate benefit (exploratory, null)")
    rho_closed = (m[m.openness == 0][["delta_pct", "f_probe"]]
                  .corr(method="spearman").iloc[0, 1])
    ax.annotate(f"closed class: Spearman ρ = +{rho_closed:.2f}\n"
                "(opposite of the fidelity hypothesis)",
                xy=(0.02, 0.96), xycoords="axes fraction", va="top",
                fontsize=7, style="italic")
    ax.set_ylim(-40, 80)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    _save(fig, "fig4_fidelity_null")


def main():
    eff = pd.read_csv(RES / "effects.csv")
    real = pd.read_csv(RES / "real_runs.csv")
    fid = pd.read_csv(RES / "fidelity.csv")
    fig1_vanishing(eff)
    fig2_omega(eff)
    fig3_real(real)
    fig4_fidelity_null(eff, fid)


if __name__ == "__main__":
    main()
