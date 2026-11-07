# Applied Soft Computing major revision — phase 1: training-budget controls

**Baseline:** `snapshot/pre-revision-local-2026-11-05` (`f3c643c1`). Never overwrite the original 12,294 training records or the revised real-domain CSVs. The patch adds three Python files: `scripts/asoc_prepare_backbone.py`, `scripts/asoc_budget_controls.py`, `tests/test_asoc_budget_controls.py`, plus this note.

## Reason for the experiment

Original `src/train.py` trains for 10 epochs, not equal optimizer steps. A 1:1 augmented pool doubles the training set, yielding approximately twice as many optimizer steps; therefore its improvement may come from additional training rather than from beneficial synthetic data. The control distinguishes training exposure from new content.

## Matched experiment

For one controlled (series, data regime, architecture, training seed), evaluate:

| Condition | Training data | Optimizer steps | Validation schedule |
|---|---|---|---|
| `epoch_baseline` | Real only | Original 10-epoch budget | At the end of each real-data epoch |
| `step_baseline` | Real only (cycled) | Same as VAE-augmented | Every augmented epoch-equivalent |
| `duplicate_real` | Real pool duplicated 1:1 | Same as VAE-augmented | Every augmented epoch-equivalent |
| `vae_augmented` | Real + VAE pool, 1:1 | Same as both matched controls | Every augmented epoch-equivalent |

All four conditions use the same per-repetition initialization seed set **before model creation**. Model selection uses **validation only**, and test data are read only after best validation checkpoint selection. `optimizer_steps`, `samples_seen`, `best_step`, wall time and per-checkpoint validation losses are logged. A VAE pool is trained once per invocation, using the original VAE generator (default 30 epochs). Generated-pool training cost is logged separately; exact total-compute matching is not claimed.

**Warning:** The existing VAE generator randomly pairs synthetic targets with real auxiliary windows. Use the default `--input-mode target` for the first matched-budget result. `--input-mode all` is an exploratory confounded condition until multivariate pairing is redesigned.

## Prerequisites

If the selected controlled series is **not** already present locally, reconstruct it from the frozen metadata without changing the archived CSVs:

```powershell
python -m scripts.asoc_prepare_backbone --file om0.5_plain_s0.npy
```

The helper verifies the regenerated series specification against `results/backbone_validation.csv` and refuses to overwrite an existing `.npy`. **Do not run** `scripts.validate_backbone` during this phase because it can overwrite the archived `results/backbone_validation.csv`. Do not run `scripts.run_baselines` or `scripts.run_augmented` as part of this initial validation.

## Minimal validation

```powershell
python -m unittest discover -s tests -p "test_asoc_budget_controls.py"
python -m scripts.asoc_budget_controls --file om0.5_plain_s0.npy --regime 0.10 --arch dlinear --rep 0 --epochs 1 --vae-epochs 1 --out results/asoc_revision/smoke_budget.csv
```

The second command is **smoke-only**; `--epochs 1 --vae-epochs 1` must not be reported in the paper. For the first scientific pilot run:

```powershell
python -m scripts.asoc_budget_controls --file om0.5_plain_s0.npy --regime 0.10 --arch dlinear --rep 0
python -m scripts.asoc_budget_controls --file om0.5_plain_s0.npy --regime 0.10 --arch dlinear --rep 1
```

Results append to `results/asoc_revision/budget_controls.csv`; completed condition keys are skipped on repeat. Each command logs all four conditions. To support the main conclusion, repeat across preselected diverse series and seeds, then summarize paired per-series differences; do **not** infer a revised effect from one example. Do not use test errors to choose which experimental conditions to report.

## Limitations remaining

- This patch does not resolve target–auxiliary alignment, bootstrap boundary artifacts, H4 prospective leakage, covariate selection across regimes, alternative generators and ratios, or cross-domain generalization.
- New seed initialization means results should be compared **within the new four-condition protocol**, not pairwise against the old CSV records by `rep` identifier alone.
- VAE randomness, optimizer updates, and training exposure are distinguished only to the extent captured by these controls; generated-pool seeds and training repetitions need their own robustness analysis later.
