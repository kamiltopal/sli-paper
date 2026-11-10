# ASOC Phase 10 — Compact Diffusion Generator Pilot

**Status:** Code + tests prepared, **no genuine Weather results have yet been obtained**.
**Date:** 10 November 2026.
**Paper:** *When Does Synthetic Data Improve Time-Series Forecasting? Evidence on Data Scarcity, Predictability, and Temporal Structure*
**Submission deadline:** 5 December 2026.

## Decision and scientific scope

This is a **small exploratory generative-family comparator**, not a reproduction of TimeGAN, Diffusion-TS, or a comprehensive stronger-generator benchmark. We chose a basic **unconditional DDPM with an MLP noise predictor**, trained on complete **target-channel input-and-future training windows** of length 120. Both the historical and future portions of these training windows are real historical observations strictly within the training split. No future validation or test information enters the generator.

Why this scope: GPU resources are constrained (MX130), and additional complex model engineering could jeopardize the revision deadline. Diffusion is an independent generative family, so a *carefully labeled* pilot provides relevant evidence; it does **not fully settle** reviewers' request for state-of-the-art TimeGAN/diffusion architectures.

Prior methodological foundations: Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020; Yuan & Qiao, *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*, ICLR 2024. The latter uses a substantially different transformer/decomposition design. **Do not label this implementation Diffusion-TS**.

## Fixed design

| Aspect | Value |
|---|---|
| Dataset / regime | Weather, last 10% of chronological training history |
| Target input | Only target channel; no auxiliary covariates |
| Forecaster | Identical local lightweight iTransformer as Phase 7 |
| Lookback / horizon | 96 / 24 |
| Synthetic-to-real ratio | 1:1 |
| Forecaster budget | 1,120 optimizer steps; batch 64; lr 0.001; validation every 112 updates |
| Initial run | Forecaster seed 0, generator seed 0: **one new forecaster training** |
| Optional expansion | Forecaster seeds 0,1,2 × generator seeds 0,1: **six total diffusion trainings** |
| Reference models | **Read only** `results/asoc_revision/real_budget_controls.csv`: Phase 7 matched step, duplicated-real and VAE (as well as epoch baseline) |
| DDPM generator | 30 epochs; 64 reverse steps; denoiser width 128; generator batch 256; lr 0.001 |
| Noise schedule | Linear beta schedule rescaled for 64 steps; terminal alpha-bar ~0.0016 (near-pure noise before reverse diffusion) |
| Generator selection | Fixed epochs, no validation tuning or test-based early stopping |
| Forecaster selection | Minimum validation MSE checkpoint, then one final test evaluation |
| Data integrity | SHA256 matched to frozen Phase 7, same train range, scaler, optimizer steps |

**Critical difference between budgets:** Diffusion and VAE forecasters have equal optimization-step budgets; the two generators **do not** have equal computational budgets. A diffusion sample requires 64 denoising passes, unlike the one-pass VAE decoder. Generator training updates and wall time are recorded separately, as are forecaster training times; do not claim equal total computation. Generator training was not hyperparameter-optimized to be competitive with specialist time-series diffusion models.

**Measurement:** Compare diffusion with VAE, duplicated-real and step baseline, **within identical Weather / regime / model seed / pool-index configurations**, using normalized test MSE and MAE plus original-scale differences. Generator pool seed identities are matched for reporting, but different generator families are not paired realizations of the same random process.

**Independence warning:** Multiple model training seeds share one test segment, and multiple forecaster seeds can share the same generated pool. They are not independent datasets or statistically independent field experiments. Report descriptive hierarchical medians (first across forecaster seeds within generator seed, then across generator seeds), pool ranges, wins, and generator costs. No fabricated p-values, confidence intervals, or causal claims.

**Exploratory-selection warning:** Weather was selected after large favorable VAE effects were observed. Do not treat a favorable diffusion-vs-VAE comparison as preregistered cross-domain evidence. If DDPM is worse, the conclusion is only about *this small window-DDPM* under *this training budget*, not about TimeGAN or diffusion models in general.

**Generator quality diagnostics:** final noise-prediction training loss, train/synthetic mean and std, training-only mean normalized power-spectrum fidelity, generator training and sampling times. These are sanity checks, not proof of utility or freedom from memorization. Do not repeatedly tune on test MSE.

## Apply patch and verify in Windows PowerShell

Put the `.patch` file in your repository root. Do not modify or regenerate existing `data/real/weather.npy` or any existing results.

```powershell
conda activate sli_2
git status --short --branch
git apply --check .\ASOC_Phase10_Compact_Diffusion_FixedStep_2026-11-10.patch
git apply .\ASOC_Phase10_Compact_Diffusion_FixedStep_2026-11-10.patch
python -m unittest discover -s tests -p "test_asoc_real_diffusion_weather.py" -v
```

### First preflight: no training, no writing

```powershell
python -m scripts.asoc_real_diffusion_weather --reps 0 --pool-seeds 0 --dry-run
```

Expected on the current Weather data: **3,570 training windows, 1,120 forecaster steps, one pending diffusion run**. The exact reference source `results/asoc_revision/real_budget_controls.csv` is required. Do not substitute `real_summary.csv` or rerun `scripts.prepare_real`.

If the preflight blocks due to missing Phase 7 rows, mismatched SHA256, or an active timestamp anomaly, **do not bypass the check**. Audit the source files first. The previously identified Weather duplicate and nine missing timestamps were outside the active 10% training/validation/test range.

### One-model, one-pool pilot

```powershell
python -m scripts.asoc_real_diffusion_weather --reps 0 --pool-seeds 0 --limit 1
python -m scripts.asoc_real_diffusion_weather --reps 0 --pool-seeds 0 --report
```

The first run trains **one DDPM generator** and **one diffusion-augmented forecaster**, not the VAE/control forecasters. If interrupted after pool generation, the pool is cached in `results/asoc_revision/diffusion_weather_pools_v1`; the next run uses the same pool. `--limit 1` limits only new forecaster trainings, not generator steps.

Output CSVs:

- `results/asoc_revision/diffusion_weather_runs.csv` — raw diffusion forecaster runs and generator provenance.
- `results/asoc_revision/diffusion_weather_report_v1/diffusion_paired.csv` — diffusion vs Phase 7 VAE and controls for each rep/pool.
- `results/asoc_revision/diffusion_weather_report_v1/diffusion_per_pool.csv` — per-pool model-seed medians.
- `results/asoc_revision/diffusion_weather_report_v1/diffusion_summary.csv` — descriptive pool-level summary.

### Expansion **only if pilot is usable and compute budget permits**

```powershell
python -m scripts.asoc_real_diffusion_weather --reps 0,1,2 --pool-seeds 0,1 --dry-run
python -m scripts.asoc_real_diffusion_weather --reps 0,1,2 --pool-seeds 0,1
python -m scripts.asoc_real_diffusion_weather --reps 0,1,2 --pool-seeds 0,1 --report
```

After a complete one-pair pilot, this would add **five** forecaster trainings and one generator pool, not six additional trainings. If the generator quality diagnostics are pathological or the runtime exceeds available budget, stop after the pilot and explicitly report the limitation.

## Manuscript / reviewer placeholders (DO NOT insert completed-results claims yet)

**Methods — contingent on completing this run:**

> To evaluate whether the VAE findings were specific to a single generative family, we included an exploratory unconditional window-level denoising diffusion probabilistic model (DDPM). A lightweight MLP predicts the added Gaussian noise for standardized target-only windows containing 96 historical and 24 future training time steps. The model was fitted exclusively to the 10% Weather training period. The synthetic windows were used as a 1:1 augmentation of the real training windows, with a fixed forecaster optimizer budget identical to the VAE and real-data controls. DDPM generation cost was measured separately. This compact baseline was not intended as a reproduction of Diffusion-TS or as a claim of state-of-the-art generator performance.

**Results / reviewer response — fill only after actual CSVs are examined:**

> We added a small diffusion-family feasibility comparison. In the Weather 10% target-only iTransformer setup, the compact DDPM [INSERT measured MSE/MAE and generator-cost differences], compared with the pre-existing VAE and matched real-data controls. These results are exploratory and are restricted to one domain, one architecture, and the specific DDPM parameterization. They do not replace a full TimeGAN or specialized time-series diffusion benchmark; we acknowledge that limitation explicitly.

## Stop criterion

Complete this one pilot before deciding whether to expand. After the maximum 3 × 2 matrix, **stop new generator experiments** and prioritize manuscript merging, figures, audited citations, cost tables and point-by-point response letter before 5 December 2026. Do not add an unbounded hyperparameter search.
