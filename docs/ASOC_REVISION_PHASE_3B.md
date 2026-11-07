# Applied Soft Computing — Phase 3B: joint-window multivariate VAE

## Research question

Does modeling **target history, historical auxiliary channels and future target together** improve synthetic-augmentation outcomes relative to the frozen original target-only VAE with randomly paired auxiliary histories? Are any such gains attenuated when we deliberately break the generated target–auxiliary relation?

The joint VAE is an **unconditional window generator**, not a validated conditional time-series simulator. It does **not** synthesize consistent across-window trajectories and does not guarantee correct time alignment, cross-channel dependence, or downstream forecasting utility. Treat this as an exploratory controlled ablation, not a new headline conclusion or a causal attribution to attention.

## Five iTransformer conditions (all receive the same three input channels)

1. `itr_all_step`: real only, windows cycled to reach the augmented optimizer-step budget.
2. `itr_all_duplicate`: real windows duplicated 1:1.
3. `itr_all_vae_original`: original **unchanged** target-only VAE, random real auxiliary histories.
4. `itr_all_vae_joint`: joint-window VAE, generates `(lookback, all channels)` plus `(horizon, target)` together.
5. `itr_all_vae_joint_shuffled`: exactly the same joint synthetic **target histories and labels**, but generated auxiliary histories permuted between synthetic windows (aux channels 1+ are moved together). This intentionally destroys target–aux relationships while holding marginal synthetic pools approximately fixed.

All conditions use **identical train, validation and test splits**, the same iTransformer input channel count, `rep` seed set before model construction, optimizer step count and validation checkpoint schedule. Selection is on real validation; test evaluation is once at the selected checkpoint. Original generator training remains 30 epochs for the pilot. `gen_joint_vae` uses the same latent dimension (16) and hidden width (128); its input/output layer sizes and reconstruction objectives are different. Thus **generator parameter count, training time and generator capacity are not strictly matched**. GPU memory and generation wall time are logged; this is **forecaster step-matched**, not compute-matched.

### Joint generator objective

Joint input comprises flattened train `X[:, :, :]` and `y_target_future`. Reconstruct with:

`0.5 * MSE(recon_lookback, lookback) + 0.5 * MSE(recon_target_future, target_future) + 1e-3 * KL`

This weighting is set **before** observing Phase 3B outcomes and prevents the short horizon vector from being underrepresented relative to multichannel lookback. **This differs** from the original target-only VAE's unweighted target-window loss; the two generators differ in both representation and objective. Any apparent advantage cannot be attributed to alignment alone without further controls. The joint generator is trained exclusively on standardized real training windows and returns the same number of synthetic windows as the training set.

### Diagnostics

From training windows only: median absolute gap between real and synthetic target–aux correlation at lags −12…+12, zero lag, and association of each auxiliary's *last observed lookback value* with the target at forecast steps 1, middle and 24. Values are **descriptive** (overlapping windows are not independent) and are not evidence of future causal dependence. Report them alongside downstream test MSE.

## Installation

Requires Phase 1 `scripts/asoc_budget_controls.py` and existing original `src/` code; Phase 3 is useful for comparison but not imported by this patch. Additive only: no changes to old generator source, saved model results, or Phase 1–3 CSVs. On `revision/asoc-major-2026`:

```powershell
 git status --short
 git apply --check .\asoc_phase3b_joint_vae.patch
 git apply .\asoc_phase3b_joint_vae.patch
 python -m unittest discover -s tests -p "test_asoc_joint_vae.py" -v
```

### GPU pilot — **one cell** (five training conditions)

```powershell
python -m scripts.asoc_joint_matrix --levels 0.5 --series-seeds 0 --regimes 0.10 --reps 0 --dry-run
python -m scripts.asoc_joint_matrix --levels 0.5 --series-seeds 0 --regimes 0.10 --reps 0 --limit 1 --gpu-index 0
python -m scripts.asoc_joint_report
```

Expected outputs:

- `results/asoc_revision/joint_vae_controls.csv` — raw results, optimizer steps, validation curves, GPU/runtime, correlation gaps.
- `results/asoc_revision/joint_report/joint_paired.csv` — paired per-cell contrasts.
- `results/asoc_revision/joint_report/joint_summary.csv` — descriptive medians after collapsing repeated model seeds within each series.
- `results/asoc_revision/joint_report/joint_incomplete.csv` — partially completed cells (after interruption).

Each grid cell runs in an independent Python subprocess. If Windows returns native exit code `0xC0000409` again, rerun the *same command*; completed conditions are skipped by a full key. Do not delete the partial output. For testing without GPU only, add `--allow-cpu`.

### Expand *only if* the pilot passes

```powershell
python -m scripts.asoc_joint_matrix --levels 0.2,0.5,0.8 --series-seeds 0,1 --regimes 0.10,1.00 --reps 0,1 --gpu-index 0 --dry-run
python -m scripts.asoc_joint_matrix --levels 0.2,0.5,0.8 --series-seeds 0,1 --regimes 0.10,1.00 --reps 0,1 --gpu-index 0
python -m scripts.asoc_joint_report
```

This is 24 paired cells × 5 conditions = **120 forecaster trainings total**, including the pilot. Although 24 cells are paired, there are only 6 controlled series configurations; Omega bands share generator random seeds, so observations are **not independent**. These exploratory grids do not yield statistical confidence intervals. Do not decide main paper claims from a selected seed or selected configuration.

## Interpretation rules

- `joint_vs_original_pct < 0` means the joint VAE beat the **unchanged original** VAE under matched forecaster inputs and optimizer steps.
- `joint_vs_duplicate_pct < 0` means joint VAE beat duplicated real training data; this is stronger evidence than beating the original VAE alone.
- `joint_shuffled_vs_joint_pct > 0` means shuffling the generated auxiliary histories reduced the joint generator's forecaster advantage. Only then, along with reduction in training-only lag/future-association gaps, does an *association-preservation explanation* become plausible. It is still not a causal proof of mechanism.
- Conversely, no improvement after joint training is scientifically meaningful: the old H3 explanation should be narrowed rather than post-hoc tuned on the test set.

## Remaining limitations

This patch does not address bootstrap block-boundary artifacts, H4's test-MSE headroom leakage, alternative generators, non-Fourier data, generator-compute matching, forecast input length, forecast horizon or ratio sweeps. Cross-channel representation and generator structure remain confounded relative to independent-channel architecture. Repeated related time windows and shared backbone seeds require careful uncertainty treatment for publication.
