"""Reproduce every statistic reported in the manuscript, tagged by section.

Reads results/baselines.csv, results/augmented.csv, results/real_runs.csv,
results/fidelity.csv. Writes nothing; prints the numbers with the section
of the paper in which each appears.

Run:  python -m scripts.paper_stats            (full bootstraps, ~2-3 min)
      python -m scripts.paper_stats --fast     (500-rep bootstraps, quick check)

Reproducibility notes:
- The calibration/held-out split of the 54 controlled series is fixed by
  numpy default_rng(0), as in the pre-specified analysis.
- H4 is evaluated with the CLEAN reference: band-minimum baseline errors
  computed from the calibration half only (manuscript §3.2, §4.6). The two
  robustness variants reported in §4.6 (all-series reference; permutation-
  based bands) are also computed here.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 500 if "--fast" in sys.argv else 5000

ARCH_INDEP = ("dlinear", "patchtst")


def effects(base, aug, metric):
    b = base.groupby(["file", "regime", "arch"])[metric].median().rename("m_base")
    a = (aug.groupby(["file", "regime", "arch", "generator", "openness",
                      "target_omega"])[metric]
         .median().rename("m_aug").reset_index())
    d = a.join(b, on=["file", "regime", "arch"])
    d["delta"] = 100 * (d.m_aug - d.m_base) / d.m_base
    return d


def sign_test_geq(k, n):
    """One-sided P(X >= k), X ~ Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def acc_f1(pred, y):
    acc = float(np.mean(pred == y))
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    f1 = float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    return acc, f1


def cluster_boot(test, preds, y, rng, n=N_BOOT):
    """Series-clustered bootstrap of acc/F1 per policy + paired diffs."""
    tf = test["file"].unique()
    idx_by_file = {f: test.index[test.file == f].values for f in tf}
    out = {k: {"acc": [], "f1": []} for k in preds}
    d_acc_rn, d_f1_ri = [], []
    for _ in range(n):
        pick = rng.choice(tf, size=len(tf), replace=True)
        idx = np.concatenate([idx_by_file[f] for f in pick])
        yy = y.loc[idx]
        sc = {k: acc_f1(p.loc[idx], yy) for k, p in preds.items()}
        for k, (a, f) in sc.items():
            out[k]["acc"].append(a); out[k]["f1"].append(f)
        d_acc_rn.append(sc["rule"][0] - sc["never"][0])
        d_f1_ri.append(sc["rule"][1] - sc["if-scarce"][1])
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return out, ci(d_acc_rn), ci(d_f1_ri), ci


def main():
    base = pd.read_csv(RES / "baselines.csv")
    aug = pd.read_csv(RES / "augmented.csv")
    real = pd.read_csv(RES / "real_runs.csv")
    fid = pd.read_csv(RES / "fidelity.csv")
    eff = effects(base, aug, "test_mse")
    rng = np.random.default_rng(42)

    P = print
    P("=" * 72)
    P("§4.1 Baseline landscape")
    P("=" * 72)
    bmed = base.groupby(["target_omega", "regime"])["test_mse"].median().unstack()
    P("baseline median MSE by band (rows) x regime:\n", bmed.round(3))
    impr = 100 * (bmed[0.10] - bmed[1.00]) / bmed[0.10]
    P("improvement 10%->100% per band (%):", impr.round(1).to_dict())
    per = base.groupby("file").agg(om=("achieved_omega", "first"),
                                   pm=("pred_perm", "first"))
    from scipy.stats import spearmanr
    P("rank consistency Spearman rho(Omega, perm):",
      round(spearmanr(per["om"], per["pm"]).statistic, 3))

    P("\n" + "=" * 72)
    P("§4.2-4.3 Generator effects (MSE)")
    P("=" * 72)
    for g in ("vae", "bootstrap", "seasonal", "exog"):
        d = eff[eff.generator == g]
        med = d.groupby("regime")["delta"].median().round(1).to_dict()
        share = d.groupby("regime")["delta"].apply(lambda s: (s < 0).mean()).round(2).to_dict()
        P(f"{g:9s} median Δ%: {med}   helping share: {share}")
    s = eff[eff.generator == "seasonal"]
    P("seasonal median Δ% by Ω band:",
      s.groupby("target_omega")["delta"].median().round(1).to_dict())
    # H2 slope (VAE): series-clustered bootstrap + sign test
    v = eff[eff.generator == "vae"]
    slope = np.polyfit(v["regime"], v["delta"], 1)[0]
    files = v["file"].unique()
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(files, size=len(files), replace=True)
        sub = pd.concat([v[v.file == f] for f in pick])
        bs.append(np.polyfit(sub["regime"], sub["delta"], 1)[0])
    ps = v.groupby("file").apply(lambda x: np.polyfit(x["regime"], x["delta"], 1)[0],
                                 include_groups=False)
    k = int((ps > 0).sum())
    P(f"H2 (VAE): slope=+{slope:.1f} pp/regime-unit, "
      f"95% CI [{np.percentile(bs,2.5):.1f},{np.percentile(bs,97.5):.1f}]; "
      f"per-series slope>0: {k}/{len(ps)}, one-sided sign p={sign_test_geq(k,len(ps)):.2e}")

    P("\n" + "=" * 72)
    P("§4.3 MAE robustness")
    P("=" * 72)
    em = effects(base, aug, "test_mae")
    for g in ("vae", "seasonal", "bootstrap", "exog"):
        P(f"{g:9s} MAE median Δ%:",
          em[em.generator == g].groupby("regime")["delta"].median().round(1).to_dict())
    vm = em[em.generator == "vae"].groupby("file").apply(
        lambda x: np.polyfit(x["regime"], x["delta"], 1)[0], include_groups=False)
    P(f"MAE: VAE per-series slope>0 in {(vm>0).sum()}/{len(vm)} series")

    P("\n" + "=" * 72)
    P("§4.4 Configuration contrasts (H3)")
    P("=" * 72)
    e2 = eff.copy()
    e2["aclass"] = np.where(e2.arch.isin(ARCH_INDEP), "indep", "mix")
    for g in ("seasonal", "exog"):
        d = e2[e2.generator == g]
        P(f"{g}: median Δ% by class x regime:\n",
          d.pivot_table(index="aclass", columns="regime", values="delta",
                        aggfunc="median").round(1))
        perser = d.groupby(["file", "aclass"])["delta"].mean().unstack()
        diff = perser["mix"] - perser["indep"]
        bs2 = [np.median(rng.choice(diff.values, size=len(diff), replace=True))
               for _ in range(N_BOOT)]
        if g == "seasonal":
            kk = int((diff < 0).sum())
            p2 = 2 * sign_test_geq(max(kk, len(diff) - kk), len(diff))
            P(f"  paired contrast (mix-indep): median={diff.median():+.2f} pp, "
              f"95% CI [{np.percentile(bs2,2.5):+.2f},{np.percentile(bs2,97.5):+.2f}], "
              f"mix better in {kk}/{len(diff)}, two-sided sign p={min(p2,1):.3f}")
        else:
            kk = int((diff > 0).sum())
            P(f"  paired contrast (mix-indep): median={diff.median():+.2f} pp, "
              f"95% CI [{np.percentile(bs2,2.5):+.2f},{np.percentile(bs2,97.5):+.2f}], "
              f"mix worse in {kk}/{len(diff)}")

    P("\n" + "=" * 72)
    P("§4.5 Real domains")
    P("=" * 72)
    rb = real[real.generator == "none"].groupby(
        ["domain", "regime", "arch"])["test_mse"].median().rename("m_base")
    ra = (real[real.generator != "none"].groupby(
        ["domain", "regime", "arch", "generator"])["test_mse"]
        .median().rename("m_aug").reset_index().join(rb, on=["domain", "regime", "arch"]))
    ra["delta"] = 100 * (ra.m_aug - ra.m_base) / ra.m_base
    P("median Δ% (over archs) domain x generator x regime:\n",
      ra.pivot_table(index="domain", columns=["generator", "regime"],
                     values="delta", aggfunc="median").round(1))
    ep = ra[ra.domain == "epias"]
    P("EPİAŞ full-data cells (all should be >0):",
      ep[ep.regime == 1.0][["arch", "generator", "delta"]].round(1).to_dict("records"))
    P("EPİAŞ scarce cells:",
      ep[ep.regime == 0.10][["arch", "generator", "delta"]].round(1).to_dict("records"))
    w = ra[(ra.domain == "weather") & (ra.regime == 1.0) & (ra.generator == "vae")]
    P("weather/VAE/full Δ%:", w["delta"].round(1).tolist(),
      "| baseline MSE:", rb.loc["weather", 1.0].round(4).to_dict())

    P("\n" + "=" * 72)
    P("§4.6 Decision rule (H4) — clean calibration-only reference")
    P("=" * 72)
    g = base.groupby(["file", "target_omega", "regime", "arch"])["test_mse"] \
            .median().reset_index().rename(columns={"test_mse": "e_base"})
    split_rng = np.random.default_rng(0)
    files_all = np.sort(eff["file"].unique())
    cal_files = set(split_rng.choice(files_all, size=len(files_all) // 2,
                                     replace=False))
    m_cal = g[g.file.isin(cal_files)].groupby("target_omega")["e_base"].min()
    g["H"] = (g["e_base"] - g["target_omega"].map(m_cal)) / g["target_omega"].map(m_cal)
    d2 = eff.merge(g[["file", "regime", "arch", "H"]], on=["file", "regime", "arch"])
    cal = d2[d2.file.isin(cal_files)]
    test = d2[~d2.file.isin(cal_files)]
    y = test["delta"] < 0
    grid = np.quantile(cal["H"], np.linspace(0.1, 0.9, 17))
    rule_f = lambda d, t: (d["H"] > t) & ((d["openness"] == 1) | (d["regime"] < 1.0))
    best_acc, h = max((np.mean(rule_f(cal, t) == (cal["delta"] < 0)), t) for t in grid)
    P(f"h*={h:.2f}  calibration acc={best_acc:.3f}")
    preds = {"rule": rule_f(test, h),
             "always": pd.Series(True, index=test.index),
             "never": pd.Series(False, index=test.index),
             "if-scarce": test["regime"] < 1.0}
    out, dci, fci, _ = cluster_boot(test, preds, y, rng)
    for k2, p2 in preds.items():
        a0, f0 = acc_f1(p2, y)
        P(f"  {k2:9s} acc={a0:.3f} CI[{np.percentile(out[k2]['acc'],2.5):.3f},"
          f"{np.percentile(out[k2]['acc'],97.5):.3f}]  F1={f0:.3f} "
          f"CI[{np.percentile(out[k2]['f1'],2.5):.3f},"
          f"{np.percentile(out[k2]['f1'],97.5):.3f}]")
    a_r, _ = acc_f1(preds["rule"], y); a_n, _ = acc_f1(preds["never"], y)
    _, f_r = acc_f1(preds["rule"], y); _, f_i = acc_f1(preds["if-scarce"], y)
    P(f"  paired acc(rule-never): {a_r-a_n:+.3f}  CI [{dci[0]:+.3f},{dci[1]:+.3f}]")
    P(f"  paired F1(rule-ifscarce): {f_r-f_i:+.3f}  CI [{fci[0]:+.3f},{fci[1]:+.3f}]")

    # robustness (a): all-series reference
    m_all = g.groupby("target_omega")["e_base"].min()
    g["Ha"] = (g["e_base"] - g["target_omega"].map(m_all)) / g["target_omega"].map(m_all)
    da = eff.merge(g[["file", "regime", "arch", "Ha"]].rename(columns={"Ha": "H"}),
                   on=["file", "regime", "arch"])
    cala, testa = da[da.file.isin(cal_files)], da[~da.file.isin(cal_files)]
    grida = np.quantile(cala["H"], np.linspace(0.1, 0.9, 17))
    _, ha = max((np.mean(rule_f(cala, t) == (cala["delta"] < 0)), t) for t in grida)
    aa, fa = acc_f1(rule_f(testa, ha), testa["delta"] < 0)
    P(f"  robustness (all-series reference): acc={aa:.3f}  F1={fa:.3f}")
    # robustness (b): permutation-based bands
    per2 = base.groupby("file")["pred_perm"].first()
    pband = pd.qcut(per2, 3, labels=["low", "mid", "high"])
    g2 = g.merge(pband.rename("pb"), left_on="file", right_index=True)
    m2 = g2[g2.file.isin(cal_files)].groupby("pb", observed=True)["e_base"].min()
    g2["Hp"] = (g2["e_base"] - g2["pb"].map(m2).astype(float)) / g2["pb"].map(m2).astype(float)
    dp = eff.merge(g2[["file", "regime", "arch", "Hp"]].rename(columns={"Hp": "H"}),
                   on=["file", "regime", "arch"])
    calp, testp = dp[dp.file.isin(cal_files)], dp[~dp.file.isin(cal_files)]
    gridp = np.quantile(calp["H"], np.linspace(0.1, 0.9, 17))
    _, hp = max((np.mean(rule_f(calp, t) == (calp["delta"] < 0)), t) for t in gridp)
    ap, fp = acc_f1(rule_f(testp, hp), testp["delta"] < 0)
    P(f"  robustness (permutation-based bands): acc={ap:.3f}  F1={fp:.3f}")

    P("\n" + "=" * 72)
    P("§4.7 Exploratory similarity diagnostics")
    P("=" * 72)
    P("F_spec medians:", fid.groupby("generator")["f_spec"].median().round(2).to_dict())
    P("F_probe medians:", fid.groupby("generator")["f_probe"].median().round(2).to_dict())
    m = eff.merge(fid.drop(columns=["openness"]), on=["file", "regime", "generator"])
    rho_dd = m[m.openness == 0][["delta", "f_probe"]].corr(method="spearman").iloc[0, 1]
    P(f"data-derived class Spearman rho(delta, f_probe) = +{rho_dd:.2f}")


if __name__ == "__main__":
    main()
