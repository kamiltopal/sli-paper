# ASOC Phase 7 — Real-domain matched-budget validation

**Scope.** Independent add-on to the initial 144 real-domain runs. Original `results/real_runs.csv`, original `.npy` real-domain arrays and the frozen backbone runs are **never overwritten**. First use `electricity` (ETTh1 target `OT`, 17,420 rows) at the newly requested 25% training regime to minimize pilot costs on an MX130. This is not an independent new real-world dataset; it is a new condition on an existing dataset.

## Preflight

From repo root, with the existing conda environment and branch `revision/asoc-major-2026`:

```powershell
python -m unittest discover -s tests -p 'test_asoc_real_phase7.py' -v
python -m scripts.asoc_real_matrix --dry-run
Get-Item data/real/electricity.npy
```

Do **not** run `scripts.prepare_real` simply to prepare the pilot if the `.npy` is already present: it could overwrite existing arrays. Make a backup and re-verify preprocessing only if source files have changed. The pilot reads the existing array and logs its SHA-256 digest; subsequent runs refuse changed data in the same result CSV.

## Pilot: one real domain, 25% training, one seed each

```powershell
python -m scripts.asoc_real_matrix --domains electricity --regimes 0.25 --archs dlinear --input-mode target --reps 0 --pool-seeds 0 --limit 1 --gpu-index 0
python -m scripts.asoc_real_report
```

Expected: **4 model trainings**, three controls once + one 30-epoch target-only VAE pool. All forecasters share the same initial model seed, split, normalization, selection interval, batch size and learning rate. The short `epoch_baseline` has fewer steps by design. Equal-step conditions use **exactly the same optimizer updates**, and validation is checked at the same interval; no claims of equal GPU time or full generator+forecaster compute.

Upload these outputs first:

- `results/asoc_revision/real_report/real_paired.csv`
- `results/asoc_revision/real_report/real_summary.csv`

Only after checking this pilot, run `--archs itransformer` with `--input-mode target`. Full-data (100%) multivariate iTransformer can be run as `--archs itransformer --regimes 1.00 --input-mode all` in a **separate** comparison. In all-channel mode we use joint-window VAE from Phase 3B, not the original target VAE with randomly resampled auxiliary channels. **For weather/traffic/electricity at 10% or 25%, all-channel mode is intentionally blocked**: their pre-selected auxiliary columns came from the full 70% train span, and using them would leak information outside the available scarce slice. EPİAŞ's predeclared calendar covariates are permitted in scarce all-channel experiments. The two generator definitions must not be pooled in an analysis.

## Scaled protocol (after pilot)

```powershell
python -m scripts.asoc_real_matrix --domains electricity,traffic --regimes 0.10,0.25,1.00 --archs dlinear,itransformer --input-mode target --reps 0,1,2 --pool-seeds 0,1 --gpu-index 0
python -m scripts.asoc_real_report
```

This is a planning example, **not** a blanket recommendation to run the whole grid on a laptop before pilot review. The control baseline/duplicated-real is computed once per model seed (pool_seed=-1); VAE pools are generated once per generator seed per group, and reused across model seeds. Models for each domain/regime/architecture execute in a separate resumable subprocess. A failed row does not invalidate completed rows. The report verifies actual optimizer-step equality, correct pool seeds and original-unit conversions.

## Measurement and interpretation

- Split remains chronological: 70% training / 10% validation / 20% test; each regime uses the last fraction of the training segment. Lookback 96, horizon 24, same as the archived protocol.
- Scaler is fit **only** on the regime-specific training slice; all comparison arms within a regime see identical scaling. For real physical units, MSE is multiplied by target training standard deviation squared; MAE is multiplied by standard deviation. The original-unit column allows cross-regime descriptive inspection; it does **not** remove confounding from longer historical coverage.
- `input_mode=target` avoids covariate reselection and target–auxiliary alignment confounds. Architecture conclusions are restricted to these input configurations. `input_mode=all` uses a separately defined, jointly modeled multichannel generator.
- Pool seeds and training seeds are separate axes: **do not treat all (rep,pool_seed) pairs as independent experiments**. The report first collapses model seeds per pool, then pool seed summaries; with few pools only descriptive medians and ranges are reported, no inflated confidence intervals.
- The target-only VAE consumes standardized training windows; training labels/validation/test are never used to fit the generator. A generator seed is logged separately from model initialization seeds.
- The initial `real_runs.csv` does not share the new protocol's initialization, training budget and possibly covariate-selection setup. Do **not** compare old rows as if they were matched pairs.
- A 1:1 augmentation ratio is held constant here. This phase alone does not resolve ratio/horizon dependence, more advanced generators, data family breadth or compute-time matching.

## Report files

- `real_paired.csv`: one row per model repetition × VAE pool, with paired normalized and original-scale absolute and relative loss differences.
- `real_per_pool.csv`: median over training repetitions **within** each independently generated pool.
- `real_summary.csv`: median/range across pool seeds; no unjustified statistical significance claims.
- `real_incomplete.csv`: VAE rows missing any of the three reference controls.

**Stop criterion for the pilot:** if validation behaves unexpectedly, generator pool is unstable, or any setting is incommensurable, do not expand the grid. Inspect normalized and absolute error, convergence checkpoint, and whether the generator itself materially raises runtime.
