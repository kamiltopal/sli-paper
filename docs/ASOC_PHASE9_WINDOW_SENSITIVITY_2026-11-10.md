# Applied Soft Computing — Phase 9: Weather lookback/horizon sensitivity

**Date:** 2026-11-10
**Status:** protocol and code ready; **not run on real Weather GPU data**.
**Revision deadline:** 2026-12-05.
**Experiment type:** exploratory, limited-scope architectural/windowing sensitivity. Do not describe it as preregistered or a domain-level causal result.

## Question and design

Does the *within-configuration* VAE advantage over matched real-data controls persist when we change the observed history length (lookback) or the output prediction horizon?

| Configuration | Lookback | Horizon | Real Weather 10% train windows | Short-budget epoch baseline | Matched optimizer updates / arm | Role |
|---|---:|---:|---:|---:|---:|---|
| A | 96 | 24 | 3,570 | 560 | 1,120 | Phase 7 existing **read-only reference** |
| B | 48 | 24 | 3,618 | 570 | 1,140 | Shorter history; **4 new trainings in pilot** |
| C | 96 | 48 | 3,546 | 560 | 1,110 | Longer forecast; **4 new trainings in pilot** |

**Common settings:** Weather (`data/real/weather.npy`), 10% most-recent training history; target-only iTransformer; 70/10/20 chronological boundaries; separate *training-only* standardization; VAE full-window generator trained on training data only; one synthetic window per real training window, `epochs=10`, `vae_epochs=30`, `batch_size=64`, `lr=0.001`, `latent=16`. Phase 7 reference additionally used `hidden=128`, `horizon_weight=0.5`, both checked on import. Validation-only checkpoint selection. Each configuration uses its own window length, train-window count, and optimizer step budget; **within** one configuration, `step_baseline`, `duplicate_real`, and `vae_augmented` have equal update counts, validation-checkpoint intervals, and model seeds. The epoch baseline intentionally has a shorter budget as a diagnostic control.

**A fixed update count does not imply equal compute**, identical sample exposure, or identical train-loader sample sequences. Training cost and VAE generator cost remain distinct. Synthetic pools are cached and bound to their dataset digest, split, generator configuration, seed, lookback, and horizon; an existing pool is reused across forecaster initialization seeds. The model architecture and VAE input dimensionality themselves change with lookback/horizon and thus the experiment does not isolate a single causal mechanism.

**Calendar audit:** Original Weather raw CSV includes one duplicate (2020-05-12) and nine missing timestamps (2020-05-29) outside the active 10% training slice, validation and test. The code checks the active slice, not the full historical CSV. It does not modify raw or prepared datasets. If the raw timestamp file is absent, training remains possible with a clearly recorded `raw_csv_unavailable` provenance flag (no claim of independently verified regular cadence).

## Files

- `scripts/asoc_real_window_sensitivity.py` — independent Phase 9 runner, cached VAE pools, resume, report.
- `tests/test_asoc_real_window_sensitivity.py` — fast CPU protocol and fake-fit tests.
- New outputs **only** under `results/asoc_revision/`:
  - `window_sensitivity_v1.csv`
  - `window_sensitivity_pools_v1/*.npz`
  - `window_sensitivity_report_v1/window_paired.csv`
  - `window_sensitivity_report_v1/window_per_pool.csv`
  - `window_sensitivity_report_v1/window_summary.csv`
  - `window_sensitivity_report_v1/window_incomplete.csv`

The existing `results/asoc_revision/real_budget_controls.csv` is read and validated, **never rewritten**. Existing Traffic, Weather ratio, jitter/scaling, and original paper results are untouched.

## Windows PowerShell, after git patch

```powershell
conda activate sli_2
git switch revision/asoc-major-2026
git apply --check .\ASOC_Phase9_Weather_Window_Sensitivity_2026-11-10.patch
git apply .\ASOC_Phase9_Weather_Window_Sensitivity_2026-11-10.patch
python -m unittest discover -s tests -p "test_asoc_real_window_sensitivity.py" -v
```

Verify dry-run: **4 pending** for each new configuration (3 real-data controls and 1 VAE condition); **8 new forecaster trainings** in both pilots together:

```powershell
python -m scripts.asoc_real_window_sensitivity --dry-run
```

Run **only first configuration** `48→24` on MX130:

```powershell
python -m scripts.asoc_real_window_sensitivity --configs 48:24 --reps 0 --pool-seeds 0 --limit 1
python -m scripts.asoc_real_window_sensitivity --configs 48:24 --reps 0 --pool-seeds 0 --report
```

The report includes the read-only Phase 7 `96→24` reference for **only model rep=0 and VAE pool=0**. It may warn `INCOMPLETE` if 96→48 has not been run when reporting both configurations. The report compares VAE against baseline controls separately for each configuration.

After reviewing first pilot, run the other configuration:

```powershell
python -m scripts.asoc_real_window_sensitivity --configs 96:48 --reps 0 --pool-seeds 0 --limit 1
python -m scripts.asoc_real_window_sensitivity --configs 48:24,96:48 --reps 0 --pool-seeds 0 --report
```

Full expansion **only if the pilots are interpretable and the remaining revision budget justifies it**:

```powershell
python -m scripts.asoc_real_window_sensitivity --reps 0,1,2 --pool-seeds 0,1 --dry-run
python -m scripts.asoc_real_window_sensitivity --reps 0,1,2 --pool-seeds 0,1
python -m scripts.asoc_real_window_sensitivity --reps 0,1,2 --pool-seeds 0,1 --report
```

The default `--limit` is unlimited. A full expansion (3 forecaster seeds, 2 pools) requires 15 trainings **per new L/H configuration**, with already completed rows skipped. Phase 7 baseline remains untouched. Changing `--epochs`, `--latent`, or other defaults breaks comparability with the Phase 7 baseline unless that reference has identical settings; the reporting code will reject an unmatched reference.

**Send for review:** `window_summary.csv`, `window_per_pool.csv`, `window_paired.csv`, and `window_incomplete.csv` (when nonempty). The raw training CSV may also be needed for integrity audit.

## Interpretation boundaries

- All headline outcomes are **paired within the same L/H** against `step_baseline` and `duplicate_real`. A positive VAE effect requires lower MSE than **both** matched controls, not merely the shorter-budget epoch baseline.
- The `96→24` versus `48→24` comparison holds horizon and nominal test-origin coverage fixed, but window counts, model input parameterization, and fitted VAE differ.
- `96→24` versus `96→48` does **not** compare identical outcome horizons. Do **not** rank their normalized/raw MSE as though they were directly comparable; compare only the sign and magnitude of within-horizon augmentation-versus-control effects. Even these effect percentages are descriptive, not causal estimates.
- Medians are **hierarchical**: first across model seeds within an independently generated synthetic pool, then across pools. Multiple training runs sharing a pool and test period are not independent datasets.
- The Weather dataset/configuration was chosen after reviewing positive Weather 1× results; both the sensitivity and any seed expansion are **exploratory**. No statistical significance claim from two pools.
- Do not claim broader multi-domain robustness from this single-domain sensitivity.
- Changing L/H changes the VAE full-window dimension `lookback + horizon`; an observed effect cannot be assigned solely to forecasting history/horizon independently of generator fit.

## Provisional manuscript insert (use only after real results)

> **Window-length and forecast-horizon sensitivity.** We conducted an exploratory, compute-budget-matched sensitivity analysis on the Weather dataset in the 10% training-history regime. The previously evaluated `(lookback, horizon) = (96, 24)` configuration was retained as a read-only reference. Two additional configurations, `(48, 24)` and `(96, 48)`, were evaluated using the same target-only iTransformer and a VAE augmentation ratio of 1:1. In each configuration, the synthetic augmentation condition was compared against its own real-data step-matched baseline and duplicated-real control. The augmentation conditions and matched controls received equal optimizer update counts within each configuration; generator training time was tracked separately. **[INSERT VERIFIED TABLE AND ACTUAL OUTCOMES.]** Results are reported as within-configuration paired MSE/MAE differences, with nested aggregation over model seeds and independent generation pools. We do not directly rank error values across different forecast horizons, and interpret the analysis as exploratory because both the generator input dimensionality and forecasting task change with window settings.

## Reviewer reply placeholder — not yet a completed empirical response

> We thank the reviewer for requesting lookback and horizon sensitivity analysis. We added a controlled Weather experiment that compares `(48,24)` and `(96,48)` with the original `(96,24)` configuration, holding the target-only architecture, synthetic-to-real ratio, data partitioning protocol, and **within-configuration** optimizer budget protocol fixed. Each VAE run is assessed against both a step-matched real-data model and a duplicated-real control. **[INSERT RESULTS, VARIATION ACROSS SEEDS, FIGURE/TABLE NUMBER AND QUALIFICATIONS AFTER RUN.]** We also clarify that the raw MSE of different horizons cannot be directly interpreted as an isolated augmentation effect. The analysis is restricted to one dataset and is exploratory rather than a claim of universal window-size invariance.
