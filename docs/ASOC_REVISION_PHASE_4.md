# Applied Soft Computing revision — Phase 4: bootstrap boundary artifact

## Hypothesis

The frozen baseline bootstrap concatenates moving blocks of length 200 before
sliding a lookback-96/horizon-24 window. Windows that straddle a boundary
contain an artificial jump. Poor bootstrap results might reflect that artifact,
not a universal limitation of resampling time series.

This is an *ablation*, not an excuse to change old results after seeing them.
Old `src/generators.gen_bootstrap` and archived experiment CSVs remain **unchanged**.

## Five forecaster conditions (DLinear target-only by default)

1. `step_real`: real-only, cycled to match exactly the augmentation optimizer steps.
2. `duplicate_real`: real windows repeated 1:1, matched update budget.
3. `bootstrap_original`: uses frozen `gen_bootstrap` output **exactly**, random block concatenation then windows.
4. `bootstrap_safe`: same block-concatenated training signal, but only interior windows. It draws from eligible interior windows *with replacement* to match synthetic pool size. This reduces unique examples; the unique-source fraction is reported.
5. `bootstrap_residual`: independent, **training-only** moving-average trend plus fixed seasonal phase profiles (periods 24,168), recombined with *contiguous* joint-channel residual **windows**. There are no artificial seams within individual synthetic windows.

**Important limitation:** the third method is *not STL* (it does not perform STL/LOESS). The chosen periods are fixed heuristic periods and may be inappropriate for the controlled Fourier data. It also changes much more than the boundary handling. Hence the strong causal contrast for the splice artifact is `bootstrap_safe` vs `bootstrap_original`, not `bootstrap_residual` vs original. A residual-window improvement is supplementary, not proof that boundaries alone caused bootstrap underperformance. New methods are fitted/generated solely on the standardized training segment and create no new information about held-out futures.

Both bootstrap approaches may duplicate training signal content. The 1:1 original bootstrap returns one fewer synthetic window than the real-window count due to the frozen `int(T-L-H)` implementation; this is replicated exactly. The safe and residual synthetic pools match that exact count; duplicated-real adds `N` real windows. All five conditions nevertheless use exactly the same optimizer updates and checkpoint frequency, and pool cardinalities are separately logged. Forecaster model initialization uses the same training seed per paired cell.

## Diagnostics

For the **original** generator, we record the fraction of *selected* synthetic windows crossing a splice. For boundary-safe, this is verified to be zero. We record the candidate interior fraction, the distinct-source fraction after replacement sampling, and the 95th percentile of absolute target jumps at boundaries vs elsewhere. We do not equate an artificial jump with a causal failure of forecasting without paired test improvement.

## Install (revision branch only)

```powershell
 git branch --show-current
 git status --short
 git apply --check .\asoc_phase4_bootstrap_controls.patch
 git apply .\asoc_phase4_bootstrap_controls.patch
 python -m unittest discover -s tests -p "test_asoc_bootstrap.py" -v
```

## Pilot (one cell, five trainings)

```powershell
python -m scripts.asoc_bootstrap_matrix --levels 0.5 --series-seeds 0 --regimes 0.10 --reps 0 --dry-run
python -m scripts.asoc_bootstrap_matrix --levels 0.5 --series-seeds 0 --regimes 0.10 --reps 0 --limit 1 --gpu-index 0
python -m scripts.asoc_bootstrap_report
```

The primary raw results are in `results/asoc_revision/bootstrap_controls.csv`;
report files in `results/asoc_revision/bootstrap_report/`:
`bootstrap_paired.csv`, `bootstrap_summary.csv`, `bootstrap_incomplete.csv`.
An interrupted cell can be safely resumed. Each cell runs in its own subprocess,
and every completed condition has a full hyperparameter key. No original results
are overwritten.

## Expand *only after successful pilot*

```powershell
python -m scripts.asoc_bootstrap_matrix --levels 0.2,0.5,0.8 --series-seeds 0,1 --regimes 0.10,1.00 --reps 0,1 --gpu-index 0 --dry-run
python -m scripts.asoc_bootstrap_matrix --levels 0.2,0.5,0.8 --series-seeds 0,1 --regimes 0.10,1.00 --reps 0,1 --gpu-index 0
python -m scripts.asoc_bootstrap_report
```

Full extension: 24 paired cells × 5 models = 120 forecaster trainings, including
pilot. Same backbone seeds across Omega values are correlated; *descriptive*
medians are not confidence intervals or significance. Do not decide hypotheses
by picking favorable Omega values after observing test metrics.

## Reviewer-facing interpretation

- `safe_vs_original_pct < 0`: excluding splice-crossing windows improves test performance with matched optimizer updates (but eligible-window resampling also changes the training distribution and diversity).
- `safe_vs_duplicate_pct < 0`: safe-bootstrap beats the stronger genuine-data control.
- `residual_vs_original_pct < 0`: decomposition-residual bootstrap improves over the original; do not attribute solely to seamlessness, as decomposition is a second change.
- If safer approaches still fail compared with `duplicate_real`, state this clearly. New controls are not automatically validation of the original generator or paper's claim.
- Phase 4 focuses on target-only DLinear; other variants/horizons/architectures need separate validation before broad generalization.
