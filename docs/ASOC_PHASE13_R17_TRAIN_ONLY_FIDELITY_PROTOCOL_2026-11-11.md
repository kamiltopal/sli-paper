# ASOC Phase 13 — Reviewer R1.7: Train-only VAE window fidelity

**Date:** 2026-11-11. **Status:** tested analysis code, **no empirical Phase 13 fidelity results yet**. The real GPU generator runs must be executed on the original project's environment. This is a new, additive, exploratory analysis; it does not alter Phase 1–12 data or results.

## 1. Research question, and what it does *not* answer

**Question:** Do VAE-generated windows resemble real training windows in marginal, full-window and temporal structure, and does a small reconstructed distance descriptively accompany the sign/magnitude of the *previously recorded* equal-step VAE forecasting benefit?

- This is a **post-hoc, exploratory diagnostic**, not a trained decision rule, test for causality, or proof that fidelity predicts augmentation utility.
- **No forecaster is retrained.** The VAE generator must be **retrained/re-executed** because earlier Phase 7/12 runs did **not persist actual synthetic windows**, only training seeds and aggregate results.
- These regenerated pools may differ from the original trained pools across CUDA devices, Torch versions, or nondeterministic operations. The output explicitly reports `original_vae_pool_persisted=false`, `reconstructed_synthetic_sha256`, generator source SHA-256, Torch version and device. **Never claim the reconstructed windows are bitwise identical to the original pool** without a matching archived reference digest (which the previous protocol did not record). Correlations with previous outcomes are correspondingly indirect.
- All *fidelity* metrics use **only the selected training slice** (not validation/test), and only the target channel. Existing **test MSE** is read by the reporting script *after* diagnostics and joined solely for retrospective description.
- Distinguish **generator/window fidelity** from *absolute forecasting competitiveness*: the Phase 12 AR(2) and EPİAŞ seasonal-naive negative results remain valid and should not be hidden.

## 2. Metrics and baselines (fixed before reviewing new results)

A window is exactly `[target lookback 96 | target future 24]` (120 samples). Data are standardized with the regime's **training segment only**, using `src.data.prepare` and `standardize`. For each generator pool we select up to **512 windows per group** (for scarce cases, `n = min(512, floor(N_real/2), N_synthetic)`) and use **32 fixed seeded random unit projections**. Sampling is without replacement. The same projection directions and real subgroup are used for all references.

| CSV name | Definition | What it tests |
|---|---|---|
| `vae_marginal_w1` | SciPy Wasserstein-1 between flattened target values in sampled real/VAE windows | Marginal values, not temporal order |
| `vae_sliced_w1` | Mean empirical Wasserstein-1 across 32 random one-dimensional projections of the entire 120-element window vector | Joint window distribution with tractable random projections |
| `vae_spectral_js` | Jensen–Shannon **divergence in bits** between normalized mean per-window periodograms | Spectral structure; 0 for identical mean spectra |
| `vae_spectral_tv` | Half L1 distance between those normalized mean periodograms | Spectral mass discrepancy |
| `vae_acf_mae` | Mean absolute difference in within-window autocorrelation at lags 1,2,6,12,24 | Local temporal dependence |
| `vae_tail_p2_absdiff` | Absolute difference of fractions with absolute standardized value >2 | Crude tail fidelity |

Each metric is also computed for `real_real_*` (two disjoint **sets of overlapping training windows**) and `shuffled_*` (training window values permuted independently in time, preserving their marginal values but destroying temporal order). The `real_real` comparison is a *finite-sample reference*, **not an independent held-out sample or an uncertainty interval**; sliding windows can overlap heavily.

- `vae_minus_real_real_*` is a **descriptive** comparison with sampling variation; it is not an adjusted Wasserstein distance and can be negative.
- Distances use the standardized scale of the selected training slice. Only *within-dataset/regime* contrasts are unambiguously comparable; avoid unqualified cross-domain rankings.
- No fitted transformations or normalizers are computed using val/test samples. Negative controls and train–train references are not used for selecting models or tuning hyperparameters.
- We use sampled windows, not all overlapping windows, to constrain memory and runtime. Empirical distances are affected by sample selection and random projection count.

For reference: [SciPy W1 documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html); [Sliced Wasserstein Generative Models](https://arxiv.org/abs/1706.02631). This script implements a simple fixed random-projection sliced W1, **not** a learned discriminator and **not** a 120-dimensional exact optimal transport distance.

## 3. Install patch and validate

Run in project root, on branch `revision/asoc-major-2026`:

```powershell
git status --short --branch
git apply --check .\ASOC_Phase13_R17_Fidelity_2026-11-11.patch
git apply .\ASOC_Phase13_R17_Fidelity_2026-11-11.patch
python -m unittest discover -s tests -p "test_asoc_fidelity.py" -v
```

**Local verification:** 15/15 unit/integration tests passed, including stable deterministic diagnostics, marginal-vs-temporal negative control, partition/hash guard, resume guard, equal optimizer-step pair checking, and series-level aggregation. Only a CPU **stub** was used for one smoke test; actual VAE GPU re-generation and time cost are not independently benchmarked here.

## 4. Phase 12 — recommended primary experiment

The existing `nonfourier_runs.csv` has **12 unique VAE generator pools** (2 families ×3 series ×2 regimes ×1 pool seed). No neural forecaster runs are needed.

```powershell
python -m scripts.asoc_fidelity --source nonfourier --dry-run
python -m scripts.asoc_fidelity --source nonfourier --limit 1 --gpu-index 0
python -m scripts.asoc_fidelity --source nonfourier --gpu-index 0
python -m scripts.asoc_fidelity_report --source nonfourier
```

The output directory is `results/asoc_revision/phase13_fidelity/`:

- `nonfourier_fidelity.csv`: one reconstructed pool-level diagnostics row per series/regime/arch/pool (should be 12).
- `nonfourier_fidelity_joined.csv`: distances plus existing within-model matched-control test MSE contrasts; model seed aggregation **within each pool**.
- `nonfourier_fidelity_summary.csv`: family × regime medians after collapsing pools within each independent series (3 independent series/family).
- `nonfourier_fidelity_correlations.csv`: exploratory Spearman rank correlations on **independent-series summaries** (n=6 per regime, 12 total with paired regimes; *no p-values/CIs*, and the n=12 pooled view reuses six series).

If generation is interrupted, rerun the same command. A completed row is skipped. Change `--sample-windows`, `--projections`, or code version only with a **different `--outdir`**. Results are not written on `--dry-run`.

## 5. Phase 7 — optional four-domain replication

Only after the Phase 12 primary analysis, if the original prepared data files are present and their SHA-256 values match Phase 7. This stage is **not necessary to run Phase 12**. `source=real` defaults to target-only VAE rows; data arrays must be under `data/real/`.

```powershell
python -m scripts.asoc_fidelity --source real --regimes "0.10" --dry-run
python -m scripts.asoc_fidelity --source real --regimes "0.10" --limit 1 --gpu-index 0
python -m scripts.asoc_fidelity --source real --regimes "0.10" --gpu-index 0
python -m scripts.asoc_fidelity_report --source real
```

With the reported Phase 11 result set, we expect **approximately 10 scarce-regime generator groups**: 4 ETTh1 configurations/pools (1 DLinear + 3 iTransformer), plus 2 Traffic, 2 Weather and 2 EPİAŞ. This count is determined by the actual run CSV during dry-run. Do **not** mix target-only with all-channel VAE, or import predictor outcomes as fidelity features.

The real-domain report intentionally avoids a cross-domain correlation across only four heterogeneous datasets and small numbers of generator pools. The output labels medians as descriptive pool summaries, **not independent domain replication**.

## 6. Interpretation and manuscript language

Only interpret after reviewing every series, the real–real reference, the time-shuffle negative control, and the model outcome direction. One possible observation is that a generated pool matches marginal values but loses time dependence; another is that better spectral similarity still fails to produce a forecasting gain. Such relationships are **associations after the fact**, not mechanisms established experimentally. A perfect duplicated-real sample trivially has zero divergence from its source training windows but need not improve test MSE — fidelity is not sufficient for usefulness.

**Provisional Methods (English; may be retained after computation):**

> We conducted an exploratory, training-only assessment of generator fidelity using empirical one-dimensional Wasserstein distance, a fixed 32-projection sliced Wasserstein approximation on 120-step target windows, Jensen–Shannon divergence between normalized mean window spectra, and discrepancies in within-window autocorrelation. Metrics were computed on standardized samples from the real training segment and from reconstructed VAE pools, with real–real training-window subsamples and temporally shuffled windows used as descriptive references. Diagnostic windows did not include validation or test observations. Since earlier experiments did not archive synthetic pools, reconstructed samples cannot be guaranteed bitwise identical to the augmentation samples used in forecasting runs; consequently, associations with saved test outcomes are exploratory.

**Provisional reviewer R1.7 response (English; Results numbers must be added later):**

> We thank the reviewer for requesting an explicit distributional comparison between real and generated windows. We have added training-only Wasserstein-1 and random-projection sliced Wasserstein diagnostics, along with spectral Jensen–Shannon divergence and lagged autocorrelation discrepancy, each accompanied by finite-sample real–real and time-shuffled references. These diagnostics are reported separately from forecast test errors; any relationship between fidelity and forecast gains is descriptive and not evidence of causality. The earlier synthetic pools were not archived, so the analysis reconstructs generator outputs using logged seeds and parameters and transparently records their provenance. [Insert validated results and table/figure references after running the analysis.]

**Submission guard:** Do **not** write “R1.7 fully resolved” or assert a numeric Wasserstein–forecast correlation until the real diagnostic CSVs have been reviewed. If generator reconstruction differs materially across machines, phrase the analysis as a *replication of generator behavior*, not retrospective measurement of identical augmentation windows. To establish exact identity in future experiments, persist per-pool synthetic arrays or SHA fingerprints during training.

## 7. Scientific risks intentionally left visible

1. Test MSE association is post-hoc and uses previously saved forecaster models; no hypothesis test or significance claim.
2. Training windows overlap, so nominal window sample counts do not equal independent observations.
3. Three independent series per family and one generator pool/model seed remain insufficient for broad generalization.
4. Reconstructed VAE pools may differ across hardware/versions; original training data SHA and configuration are checked, but original pool SHA was not saved.
5. 120-element sliced W1 is an empirical approximation. Marginal W1, sliced W1, spectra and ACF measure different properties; no single metric proves generator quality.
6. The strongest simple AR(2) and seasonal-naive forecasts outperforming augmented iTransformer in relevant benchmarks remain substantive negative results.
