# Phase 5 — H4 deployment-safe decision policy and leakage audit

## Scope

The old `src/sli.py` and `scripts/paper_stats.py` calculate *headroom* from **baseline test MSE**. Computing their band reference on calibration-only series does not fix the leak: the held-out series' own baseline test error remains a rule input. Published H4 metrics from that implementation cannot be interpreted as a deployable, prospective policy.

This additive phase avoids modifying archived code/results and defines a different, **post-baseline-training but pre-augmentation** task: after fitting a real-only baseline, its **validation MSE** is available. Decide whether to augment before inspecting augmented model outcomes or test-set scores. Baseline training is a prerequisite, so this is *not* a zero-training, pre-forecast policy.

The policy is `augment iff [validation headroom > threshold and (generator_openness == 1 or regime < 1)]`. Headroom = `(baseline_val_mse / calibration_only_min_val_mse_within_(arch,regime)) - 1`. Calibration reference and threshold are refitted *on calibration seed blocks only*. It does not reuse full-series `target_omega`, `achieved_omega`, or `pred_perm` as features; these may depend on future/test samples. Architecture, regime and generator availability/openness are known in advance. The generator's **test** performance is never used to make a held-out decision.

## Evaluation

Leave **one complete series RNG seed block** out (all Omega bands and variants with suffix `_sN` stay together); refit threshold using the other blocks, then score held-out decisions. Six seed blocks in the original backbone, three in current equal-step DLinear controls: this is a small and correlated exploratory validation, not a confidence interval. Threshold optimizes **mean percent test MSE difference** on *calibration* series, equally weighting seed blocks, rather than classification accuracy. Predictions on held-out series never see held-out test outcomes.

Baselines: `never`, `always`, `if_scarce`; hindsight `oracle_test_only` (unattainable, use for regret only). Report chosen MSE relative to the baseline, sign accuracy, selection rate, and oracle regret. Negative percent MSE difference indicates benefit. Paired held-out outcomes are scored only *after* the selection has been frozen.

Two distinct sources, never pooled:

1. `archived`: old `results/baselines.csv` and `results/augmented.csv`, aggregated as medians across replications. **Confounded** because augmented training got more optimizer steps. Informative for a *leakage audit*, but cannot establish real augmentation gains. Privileged structural `seasonal` oracle is excluded by default. `exog` uses training exogenous channels and is included.
2. `matched`: Phase 2 `results/asoc_revision/budget_matrix_v2.csv` and **equal-step** `vae_augmented` versus `duplicate_real` or `step_baseline` real-only control. Only VAE/DLinear/target-input and the covered series/regimes are available; do not infer all-generator/all-architecture policy success. Pool configuration must be homogeneous. Aggregated across matching training replications before the seed-block split.

## Commands

Run from repository root; no GPU needed:

```powershell
python -m unittest discover -s tests -p "test_asoc_h4_policy.py" -v
python -m scripts.asoc_h4_policy --source archived
python -m scripts.asoc_h4_policy --source matched --matched-baseline duplicate_real
python -m scripts.asoc_h4_policy --source matched --matched-baseline step_baseline --outdir results/asoc_revision/h4_matched_step
```

Output under `results/asoc_revision/h4_*`: `h4_audit.json`, `h4_decisions.csv`, `h4_folds.csv`, `h4_seed_summary.csv`, `h4_summary.csv`. No modifications to archived experiment tables.

## Interpretation limitations

- Archived unequal-step gains remain **invalid as causal augmentation comparisons** even after fixing policy leakage.
- Real-time policy selection would also incur baseline training and generator costs; those costs are **not** in MSE regret.
- Oracle policy uses test labels and cannot be deployed or described as a learned model.
- Regime is an explicitly supplied data-availability setting, not estimated from later measurements.
- A positive or negative *group-averaged percent difference* does not mean significant evidence. Stronger claims require independent train/test series blocks, fairer seed budget, additional architectures and non-Fourier domains.
- Calibration uses the held-out **series seed** partition, not replications as independent observations. The same RNG seed can affect series at distinct Omega levels; all are kept together.
- This policy is a rule-checking sensitivity analysis, **not** a retroactive repair of the originally pre-registered H4 analysis. Manuscript needs explicit distinction and updated interpretation.
