# Phase 8 — Jittering and scaling, fixed-step simple augmentation controls

**Scope:** Exploratory, narrow R1.3 control. These experiments **do not replace** Phase 2–7, do not rerun or modify Traffic/Weather 1× VAE, and do not substantiate causal or domain-general claims.

## Preregistered configuration before new runs

- Domains: **Weather 10%** and **Traffic 10%**. Both target-only iTransformer-lite; lookback 96, horizon 24; 10-epoch equivalent *original* reference, batch 64, Adam 1e-3, same existing Phase 7 normalization and fixed test set.
- Existing Phase 7 original 1× reference data is **read-only**, from `results/asoc_revision/real_budget_controls.csv`. For Weather 10%: matched budget **1,120** updates; Traffic 10%: matched budget **350** updates.
- Offline **1× jitter**: take every training target window (past 96 and next 24), add independent `N(0, 0.03²)` Gaussian noise in **train-standardized units**, to both input and target, only in augmented training examples.
- Offline **1× scaling**: multiply each training target window (past 96 and next 24) with a single window-specific `clip(1+N(0,0.10²),0.7,1.3)` factor. Input and corresponding 24-step training label receive **the same factor**. Do not change evaluation observations.
- Each simple augmentation pool has independent augmentation seed `0` or `1`; cache fixed pool and SHA-256 so model initialization seeds reuse identical synthetic samples. Train windows unmodified are concatenated to 1× perturbed windows.
- Validation checkpoint selection and evaluation run through **the same** `scripts.asoc_budget_controls.fit_condition` as Phase 7: optimizer steps and validation interval are identical for all 1× VAE, jitter, scaling, duplicated-real, step baseline **within the domain**, but generator compute is not matched.
- Model seeds: Weather `0,1,2`; Traffic `0` (because the existing Traffic 1× VAE reference has only rep0). Augmentation seeds: both `0,1`. Therefore Weather full matrix **12 new fits** and Traffic **4 new fits** (16 total); first pilot Weather `rep0/seed0` and Traffic `rep0/seed0` require **two new fits each**.
- Weights/strengths 0.03/0.10 are fixed *before* new results; there is no test-driven parameter tuning. A fixed setting is **not** a full hyperparameter search or a demonstration of best-in-class augmentation.
- Windows overlap in time, and model/augmentation seeds do not become independent domains. First take the median across model seeds within augmentation pool; then median/range across **two pools**. Avoid confidence intervals or significance with these sample sizes. Keep **absolute MSE/MAE** and relative changes.
- Because Weather was previously selected after favorable 1× results, the experiment is an **exploratory follow-up**. Traffic is a pre-existing opposing domain, not a random independent replication of all possible series.

## Windows / Powershell, repository root

First ensure current branch is `revision/asoc-major-2026`, conda `sli_2` active, and **`data/real/traffic/traffic.csv` exists** for the chronology check. If missing, do not regenerate or overwrite data: locate the original source copy.

```powershell
python -m unittest discover -s tests -p "test_asoc_simple_aug_real.py" -v
python -m scripts.asoc_simple_aug_real --domain weather --reps 0 --aug-seeds 0 --dry-run
python -m scripts.asoc_simple_aug_real --domain weather --reps 0 --aug-seeds 0 --limit 2
python -m scripts.asoc_simple_aug_real --domain weather --reps 0 --aug-seeds 0 --report

python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0 --dry-run
python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0 --limit 2
python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0 --report
```

**Do not expand automatically**. Inspect pilot sign, absolute differences, best validation checkpoints, and pool integrity before considering the entire matrix. Since the complete matrix was specified before seeing the simple-augmentation outcomes, do **not** expand only the favorable method or favorable domain without explicitly identifying post-hoc selection.

Optional full expansion:

```powershell
python -m scripts.asoc_simple_aug_real --domain weather --reps 0,1,2 --aug-seeds 0,1 --dry-run
python -m scripts.asoc_simple_aug_real --domain weather --reps 0,1,2 --aug-seeds 0,1
python -m scripts.asoc_simple_aug_real --domain weather --reps 0,1,2 --aug-seeds 0,1 --report
python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0,1 --dry-run
python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0,1
python -m scripts.asoc_simple_aug_real --domain traffic --reps 0 --aug-seeds 0,1 --report
```

## Data integrity & outputs

- The exact data `.npy` hash must match existing Phase 7 references. The timestamp CSV for the **active** 10% training/validation/test period is checked for regular intervals (Weather 10 min, Traffic hourly). Weather's known old anomalies lie outside the active slice; the original prepared `.npy` stays unchanged.
- New results are written **only** to `results/asoc_revision/simple_aug_v1.csv`; cached pools to `simple_aug_pools_v1/`; reports to `simple_aug_report_v1/{domain}_{paired,per_pool,summary,incomplete}.csv`.
- The 1× VAE, step baseline and duplicated-real source rows are **read-only**. Any mismatch in source hash, hyperparameters, step count, train split or metric scale fails before GPU training.
- Relative differences use `100*(simple/reference-1)`; **negative means simple augmentation has lower error**. Compare `simple_vs_vae`, `simple_vs_duplicate` and `simple_vs_step` separately. If the simple method beats VAE, the claimed specificity of VAE synthetic generation weakens; if it does not, retain the limitation that only two simple methods and strengths were considered.

## R1.3 response — pending results

> We agree that generative augmentation should be evaluated against inexpensive, conventional perturbations. In the revision we have specified an exploratory, fixed-budget comparison of Gaussian jittering and window-level magnitude scaling against the original 1× VAE, repeated-real and real-only matched-step references. The study uses the same target-only iTransformer configuration, forecasting horizon, initialization seeds and fixed train/validation/test splits within each domain. Weather 10% and Traffic 10% were selected to include a domain where VAE previously helped and one where it did not. Each augmented training set contains one additional window per real window, and all forecasters in a domain use identical optimizer-step counts and validation checkpoint cadence. We will report the observed results rather than assume that either generative or simple augmentation is preferable. We acknowledge that the comparison tests two frozen perturbation strengths and does not establish optimal tuning or equal end-to-end computation.

**Do not submit this future-tense response unchanged; replace after actual runs with concrete results, tables and section references.**
