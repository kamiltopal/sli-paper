"""Phase 5: no-test-feature H4 policy evaluation on held-out backbone seed blocks.

Two deliberately separate data sources:
  archived -- original 10-epoch unequal-update augmentation grid (diagnostic only)
  matched  -- step-matched Phase 2 VAE grid (limited architecture/generator scope)

No new forecaster training, changes to archived results, or test-derived features.

Examples:
 python -m scripts.asoc_h4_policy --source archived
 python -m scripts.asoc_h4_policy --source matched --matched-baseline duplicate_real
 python -m scripts.asoc_h4_policy --source matched --matched-baseline step_baseline \\
     --outdir results/asoc_revision/h4_matched_step
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {0.10, 0.25, 1.0}
GEN_OPENNESS = {"bootstrap": 0, "vae": 0, "seasonal": 1, "exog": 1}
BASE_KEYS = ["file", "regime", "arch"]
AUG_KEYS = BASE_KEYS + ["generator"]
POLICIES = ("h4_validation", "never", "always", "if_scarce", "oracle_test_only")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; this script never generates archived runs")
    return pd.read_csv(path)


def _seed_from_file(name: str) -> int:
    m = re.search(r"_s(\d+)\.npy$", str(name))
    if not m:
        raise ValueError(f"Cannot recover backbone seed from filename: {name}")
    return int(m.group(1))


def _validate_common(frame: pd.DataFrame) -> pd.DataFrame:
    cols = set(AUG_KEYS + ["base_val_mse", "base_test_mse", "aug_test_mse", "openness"])
    missing = cols - set(frame)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    if frame.duplicated(AUG_KEYS).any():
        raise ValueError("Duplicate (file, regime, arch, generator) decisions")
    num = frame[["base_val_mse", "base_test_mse", "aug_test_mse"]].to_numpy(dtype=float)
    if not np.isfinite(num).all() or (num <= 0).any():
        raise ValueError("All baseline validation and both test MSE values must be finite/positive")
    if not set(frame["openness"].astype(int)).issubset({0, 1}):
        raise ValueError("Invalid openness labels")
    if not set(frame["regime"].round(2)).issubset(REGIMES):
        raise ValueError("Unknown data regime")
    frame = frame.copy().reset_index(drop=True)
    frame["series_seed"] = frame["file"].map(_seed_from_file)
    # This column is an EVALUATION LABEL, not a policy input.
    frame["delta_pct"] = 100.0 * (frame["aug_test_mse"] - frame["base_test_mse"]) / frame["base_test_mse"]
    return frame


def archived_decisions(baselines: pd.DataFrame, augmented: pd.DataFrame,
                       include_oracle: bool = False) -> pd.DataFrame:
    """Median across model replications; no outcome enters the policy features."""
    bk = BASE_KEYS + ["rep"]
    ak = AUG_KEYS + ["rep"]
    for name, df, keys, cols in [
        ("baselines", baselines, bk, ["val_mse", "test_mse"]),
        ("augmented", augmented, ak, ["test_mse"]),
    ]:
        missing = set(keys + cols) - set(df)
        if missing:
            raise ValueError(f"{name} missing columns {sorted(missing)}")
        if df.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} per-rep keys")
    b = baselines.groupby(BASE_KEYS, as_index=False)[["val_mse", "test_mse"]].median()
    b = b.rename(columns={"val_mse": "base_val_mse", "test_mse": "base_test_mse"})
    a = augmented.groupby(AUG_KEYS, as_index=False)["test_mse"].median()
    a = a.rename(columns={"test_mse": "aug_test_mse"})
    known = set(a.generator.unique())
    if not known.issubset(GEN_OPENNESS):
        raise ValueError(f"Unknown generator(s): {sorted(known-set(GEN_OPENNESS))}")
    if not include_oracle:
        a = a[a.generator != "seasonal"].copy()  # true signal core unavailable in deployment
    a["openness"] = a.generator.map(GEN_OPENNESS).astype(int)
    merged = a.merge(b, on=BASE_KEYS, how="left", validate="many_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Missing archived baseline for augmented cell")
    merged = merged.drop(columns="_merge")
    return _validate_common(merged)


def matched_decisions(raw: pd.DataFrame, *, baseline: str = "duplicate_real",
                      input_mode: str = "target", arch: str = "dlinear") -> pd.DataFrame:
    """Step-equal forecaster controls; only VAE augmentation is available here."""
    if baseline not in {"step_baseline", "duplicate_real"}:
        raise ValueError("matched baseline must be step_baseline or duplicate_real")
    required = {"file", "regime", "arch", "input_mode", "rep", "pool_seed",
                "condition", "val_mse", "test_mse", "optimizer_steps",
                "epochs", "vae_epochs", "batch_size", "lr"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Matched source missing {sorted(required-set(raw.columns))}")
    part = raw[(raw.arch == arch) & (raw.input_mode == input_mode)
               & raw.condition.isin([baseline, "vae_augmented"])].copy()
    if part.empty:
        raise ValueError(f"No matched {arch}/{input_mode} runs")
    experimental = ["epochs", "vae_epochs", "batch_size", "lr"]
    if len(part[experimental].drop_duplicates()) != 1:
        raise ValueError("Use a single hyperparameter configuration per evaluation")
    keys = ["file", "regime", "arch", "rep", "pool_seed"] + experimental
    if part.duplicated(keys + ["condition"]).any():
        raise ValueError("Duplicate matched condition per rep")
    b = part[part.condition == baseline][keys + ["val_mse", "test_mse", "optimizer_steps"]]
    a = part[part.condition == "vae_augmented"][keys + ["test_mse", "optimizer_steps"]]
    joined = b.merge(a, on=keys, how="outer", validate="one_to_one",
                     suffixes=("_base", "_aug"), indicator=True)
    if (joined._merge != "both").any():
        raise ValueError("Unmatched VAE/baseline condition; finish the grid first")
    if not (joined.optimizer_steps_base == joined.optimizer_steps_aug).all():
        raise ValueError("Unequal optimizer budgets in matched source")
    aggregate = (joined.groupby(BASE_KEYS, as_index=False)
                 [["val_mse", "test_mse_base", "test_mse_aug"]].median())
    aggregate = aggregate.rename(columns={"val_mse": "base_val_mse",
                                          "test_mse_base": "base_test_mse",
                                          "test_mse_aug": "aug_test_mse"})
    aggregate["generator"] = "vae"
    aggregate["openness"] = 0
    return _validate_common(aggregate)


def reference_from_calibration(calibration: pd.DataFrame) -> dict[tuple[str, float], float]:
    """Reference uses ONLY baseline VALIDATION MSE from training seed blocks."""
    refs = calibration.groupby(["arch", "regime"])["base_val_mse"].min()
    return {(str(a), float(r)): float(v) for (a, r), v in refs.items()}


def predecision_features(frame: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Strict feature whitelist; forbidden test columns cannot affect features."""
    z = frame[["file", "series_seed", "regime", "arch", "generator", "openness",
               "base_val_mse"]].copy()
    z["reference"] = [refs.get((str(a), float(r)), np.nan)
                      for a, r in zip(z.arch, z.regime)]
    if z.reference.isna().any():
        raise ValueError("Missing validation reference for held-out arch/regime")
    z["headroom"] = z.base_val_mse / z.reference - 1.0
    if not np.isfinite(z.headroom).all():
        raise ValueError("Invalid validation headroom")
    return z


def h4_select(features: pd.DataFrame, threshold: float) -> np.ndarray:
    eligible = (features.openness.to_numpy(dtype=int) == 1) | (features.regime.to_numpy() < 1.0)
    return eligible & (features.headroom.to_numpy() > threshold)


def _seed_equal_mean(frame: pd.DataFrame, values: np.ndarray) -> float:
    return float(pd.Series(values).groupby(frame.series_seed.to_numpy()).mean().mean())


def tune_threshold(cal: pd.DataFrame, features: pd.DataFrame) -> tuple[float, float]:
    """Choose threshold by per-seed equal-weight expected TEST regret ON CALIBRATION GROUPS.

    Test outcomes on calibration series are supervised training labels, not
    test features. The held-out seed block is never consulted here.
    """
    h = features.headroom.to_numpy()
    candidates = np.r_[-np.inf, np.quantile(h, np.linspace(0, 1, 41)), np.inf]
    candidates = np.unique(candidates)
    delta = cal.delta_pct.to_numpy(dtype=float)
    ranked = []
    for t in candidates:
        pred = h4_select(features, float(t))
        cost = _seed_equal_mean(cal, np.where(pred, delta, 0.0))
        # Conservative deterministic tie-break: fewest augmentations.
        ranked.append((cost, int(pred.sum()), -float(t), float(t)))
    best = min(ranked)
    return best[3], best[0]


def evaluate_seed_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-BACKBONE-seed-out nested calibration, no cross-block peeking."""
    ids = sorted(frame.series_seed.unique())
    if len(ids) < 3:
        raise ValueError("At least three independent backbone seed blocks required")
    selected = []
    folds = []
    for seed in ids:
        cal = frame[frame.series_seed != seed].reset_index(drop=True)
        held = frame[frame.series_seed == seed].reset_index(drop=True)
        refs = reference_from_calibration(cal)
        f_cal = predecision_features(cal, refs)
        h_star, train_cost = tune_threshold(cal, f_cal)
        f_held = predecision_features(held, refs)
        policies = {
            "h4_validation": h4_select(f_held, h_star),
            "never": np.zeros(len(held), dtype=bool),
            "always": np.ones(len(held), dtype=bool),
            "if_scarce": held.regime.to_numpy() < 1.0,
            "oracle_test_only": held.delta_pct.to_numpy() < 0.0,
        }
        folds.append({"heldout_series_seed": int(seed), "calibration_seed_blocks": len(ids)-1,
                      "calibration_decisions": len(cal), "heldout_decisions": len(held),
                      "threshold_validation_headroom": h_star,
                      "calibration_mean_delta_pct": train_cost})
        for policy, pred in policies.items():
            p = held[["file", "series_seed", "regime", "arch", "generator",
                      "base_test_mse", "aug_test_mse", "delta_pct"]].copy()
            p["policy"] = policy
            p["selected_augment"] = pred
            p["realized_mse"] = np.where(pred, held.aug_test_mse, held.base_test_mse)
            p["realized_delta_pct"] = np.where(pred, held.delta_pct, 0.0)
            p["oracle_delta_pct"] = np.minimum(held.delta_pct.to_numpy(), 0.0)
            p["regret_pct_points"] = p.realized_delta_pct - p.oracle_delta_pct
            p["correct_sign_choice"] = pred == (held.delta_pct.to_numpy() < 0.0)
            # Diagnostics only: not available to predecision policy
            p["validation_headroom"] = f_held.headroom.to_numpy()
            p["selected_threshold"] = h_star
            selected.append(p)
    return pd.concat(selected, ignore_index=True), pd.DataFrame(folds)


def report_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize first per held-out seed block, then equally across blocks."""
    group = (decisions.groupby(["policy", "series_seed"], as_index=False)
             .agg(n_decisions=("file", "size"),
                  selected_fraction=("selected_augment", "mean"),
                  sign_accuracy=("correct_sign_choice", "mean"),
                  mean_delta_pct=("realized_delta_pct", "mean"),
                  mean_regret_pct_points=("regret_pct_points", "mean"),
                  mean_realized_mse=("realized_mse", "mean")))
    out = (group.groupby("policy", as_index=False)
           .agg(n_heldout_seed_blocks=("series_seed", "size"),
                n_decisions=("n_decisions", "sum"),
                mean_selected_fraction=("selected_fraction", "mean"),
                mean_sign_accuracy=("sign_accuracy", "mean"),
                mean_delta_pct=("mean_delta_pct", "mean"),
                mean_regret_pct_points=("mean_regret_pct_points", "mean"),
                mean_realized_mse=("mean_realized_mse", "mean")))
    order = {name: i for i, name in enumerate(POLICIES)}
    out = out.sort_values("policy", key=lambda x: x.map(order)).reset_index(drop=True)
    return group, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("archived", "matched"), required=True)
    p.add_argument("--baselines", type=Path, default=Path("results/baselines.csv"))
    p.add_argument("--augmented", type=Path, default=Path("results/augmented.csv"))
    p.add_argument("--matched", type=Path, default=Path("results/asoc_revision/budget_matrix_v2.csv"))
    p.add_argument("--matched-baseline", choices=("step_baseline", "duplicate_real"),
                   default="duplicate_real")
    p.add_argument("--matched-arch", default="dlinear")
    p.add_argument("--matched-input-mode", choices=("target", "all"), default="target")
    p.add_argument("--include-oracle", action="store_true", help="Include privileged seasonal oracle only as sensitivity analysis")
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    resolve = lambda x: x if x.is_absolute() else ROOT / x
    if args.source == "archived":
        dataset = archived_decisions(_read_csv(resolve(args.baselines)),
                                     _read_csv(resolve(args.augmented)),
                                     include_oracle=args.include_oracle)
        tag = "archived_unequal_steps_exploratory"
        default_dir = "h4_archived_with_oracle" if args.include_oracle else "h4_archived"
        warning = "UNEQUAL optimizer-step budgets: descriptive diagnostic only, NOT a valid augmentation gain claim"
    else:
        dataset = matched_decisions(_read_csv(resolve(args.matched)),
                                    baseline=args.matched_baseline,
                                    arch=args.matched_arch,
                                    input_mode=args.matched_input_mode)
        tag = f"matched_{args.matched_baseline}_{args.matched_arch}_{args.matched_input_mode}"
        default_dir = "h4_" + tag
        warning = "Equal optimizer-step budgets, but limited to the source grid's architectures/series/generator"

    decisions, folds = evaluate_seed_blocks(dataset)
    per_seed, summary = report_tables(decisions)
    output = resolve(args.outdir or Path("results/asoc_revision") / default_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output / "h4_decisions.csv", index=False)
    folds.to_csv(output / "h4_folds.csv", index=False)
    per_seed.to_csv(output / "h4_seed_summary.csv", index=False)
    summary.to_csv(output / "h4_summary.csv", index=False)
    audit = dict(source=tag, warning=warning, n_series=int(dataset.file.nunique()),
                 n_backbone_seed_blocks=int(dataset.series_seed.nunique()),
                 n_decisions=int(len(dataset)),
                 feature_whitelist=["baseline validation MSE", "calibration-only reference",
                                    "regime", "architecture", "generator openness"],
                 forbidden_policy_features=["test_mse", "delta_pct", "achieved_omega",
                                            "target_omega", "pred_perm", "future outcomes"],
                 calibration="leave-one-backbone-seed-out; supervised policy outcomes only on calibration folds",
                 objective="equal-seed-weighted mean percent MSE difference; lower better",
                 oracle="hindsight upper bound; NEVER a deployable policy",
                 note="validation loss available after real-only baseline training; this is a post-baseline, pre-augmentation policy")
    (output / "h4_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Source: {tag}; decisions={len(dataset)}; backbone seed blocks={dataset.series_seed.nunique()}")
    print("WARNING:", warning)
    print(summary[["policy", "mean_delta_pct", "mean_regret_pct_points",
                   "mean_sign_accuracy", "mean_selected_fraction"]].round(3).to_string(index=False))
    print(f"Saved audit, decisions, folds and summaries to {output}")
    print("Negative mean_delta_pct = lower MSE than baseline. Oracle uses TEST outcomes only as an unattainable bound.")
    print("Seed blocks may share signal construction and are few; this is NOT an independent-sample confidence interval.")


if __name__ == "__main__":
    main()
