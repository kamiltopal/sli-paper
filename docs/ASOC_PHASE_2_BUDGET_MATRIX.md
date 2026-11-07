# ASOC major revision — Phase 2: controlled budget-matching replication

## Purpose

The pilot (`om0.5_plain_s0.npy`, 10% real data, DLinear, rep=0) showed
VAE improvement relative to the original 210-step baseline, but little
additional advantage beyond the 410-step real-only baseline, and worse
performance than duplicated-real windows. This observation comes from **one
series and one training seed** and is not a general conclusion.

This phase expands the *same four-condition* protocol across controlled series configurations, training repetitions, and real-data regimes. The hypotheses
and condition definitions are frozen before reading outcomes. Use
`--input-mode target` for this step-budget experiment to avoid the current
random VAE target–auxiliary pairing confound.

## Reproducibility and isolation

- Parent snapshot: `snapshot/pre-revision-local-2026-11-05` (`f3c643c`).
- Runner prerequisite: `scripts/asoc_budget_controls.py` from Phase 1; the
  no-GPU variant also works on CUDA GPU 0, but upgrade to the GPU patch for
  explicit selection and hardware logging.
- Old published CSVs must **never** be overwritten.
- Default new output: `results/asoc_revision/budget_matrix_v2.csv`.
- The runner checks four complete conditions and skips completed cells on
  restart. It aborts on duplicate keys.
- Every invocation reuses the frozen VAE-generation procedure, so no new
  augmentation mechanism is introduced at this stage.
- `series-seeds` refers to **distinct generated time series**; `reps`
  refers to repeated **model initialization/training** on a fixed series.
  The same generator RNG seed is reused at multiple Ω levels, so configurations
  with a shared seed are NOT statistically independent. Aggregate training
  repetitions within each configuration and account for seed-block dependence
  in any confirmatory inference.

## Commands (Windows PowerShell, repository root)

```powershell
python -m unittest discover -s tests -p "test_asoc_budget_matrix_report.py"
python -m scripts.asoc_budget_matrix --dry-run

# Screening: 3 Ω configurations (sharing base seed s0) x 3 training reps
# x 2 regimes = 18 experiment cells x 4 controls = 72 model fits
python -m scripts.asoc_budget_matrix --gpu-index 0

# If needed, start with only the first cell; later re-run full command to resume.
python -m scripts.asoc_budget_matrix --limit 1 --gpu-index 0

python -m scripts.asoc_budget_report
```

## Planned extension if screening is valid

```powershell
# 9 series configurations (3 levels x 3 seed blocks), 3 training reps,
# all 3 real-data regimes: 81 cells x 4 = 324 model fits; existing
# matching cells will be skipped.
python -m scripts.asoc_budget_matrix --series-seeds 0,1,2 --regimes 0.10,0.25,1.00 --gpu-index 0
python -m scripts.asoc_budget_report
```

## Check convergence, separately from the 10-epoch screening

```powershell
# One series, 3 repetitions, 10% regime, twice the original epoch budget:
python -m scripts.asoc_budget_matrix --levels 0.5 --series-seeds 0 --reps 0,1,2 --regimes 0.10 --epochs 20 --gpu-index 0
python -m scripts.asoc_budget_report
```

**Interpretation safeguards**

1. `vae_vs_step_pct` isolates benefit relative to the equal-step real-data
   baseline, whereas `vae_vs_duplicate_pct` compares against repeated-real
   exposure with the same number of optimizer steps. Negative values imply
   lower test MSE for VAE.
2. No test-set outcome is used for model selection; phase-1 checkpoint
   selection uses validation data only.
3. `best_at_last_*` flags whether the best validation checkpoint was the
   final allowable step. High rates indicate the 10-epoch budget may be too
   short; the 20-epoch control probes this without changing the main protocol.
4. Report descriptive medians over series configurations. Because identical
   seed IDs across Ω levels share a generator random source, 9 configurations
   across 3 seeds are NOT 9 independent realizations. Avoid confidence
   intervals and p-values from this small screening matrix. The existing
   results in the paper cannot be directly *paired* with runs using the new
   model-initialization protocol based only on the same `rep` integer.
5. These checks do not yet resolve VAE auxiliary alignment, Fourier-only
   external validity, H4 leakage, or additional generators.
6. Generator fitting cost is logged separately; step-matched forecaster
   controls are not identical to end-to-end **compute-matched** controls.
