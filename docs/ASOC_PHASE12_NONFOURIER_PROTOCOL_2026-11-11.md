# ASOC Major Revision — Phase 12: Non-Fourier Stochastic Process Generalization

**Status:** Implemented and unit-tested, but **no Phase 12 GPU outcomes have been observed**. Protocol and complete list of series are fixed **before** looking at pilot forecasting results. No earlier `data/real` or `results/` experiments are changed.

## Why this experiment?

Reviewers R1.4 and R6.2 questioned whether findings on the original Fourier-mixture synthetic backbone extend to other dynamical process classes. Instead of generating additional parameter variants from the same Fourier model, this controlled *exploratory* experiment uses two explicit, stochastic, non-sinusoidal data-generating processes (DGPs). They were selected for structural distinction, **not** for anticipated augmentation benefit. This is not a comprehensive benchmark of all nonlinear or regime-switching dynamics, and should not be presented as causal evidence for a particular process property.

## Frozen experiment grid

- **Families:** `nonlinear_ar`, `markov_ar`.
- **Independent full-series realizations:** IDs `0,1,2` for each family (RNG seeds 41000–41002, 51000–51002).
- **Series length:** 12,000 samples after 500-step burn-in, float32 univariate. No time-of-day/hidden-state features supplied to models. **No Fourier basis, sinusoids, or hidden test labels are supplied.**
- **Chronological splits:** first 70% train, next 10% validation, last 20% test. Under 10% scarcity only the last 10% of that 70% train span is used for training and normalization. Normalization computed solely on this selected span.
- **Lookback/horizon:** 96/24; target-only forecasting (the existing lite iTransformer is a single-channel configuration here, and **does not test cross-channel mixing**).
- **Regimes:** 0.10 and 1.00, pre-specified for **every one** of the six series, irrespective of pilot outcomes.
- **Model:** existing `src.models.build('itransformer',96,24)`, **no architectural changes**. DLinear can be included as a separate optional, uniformly applied experiment but is *not* in the default grid.
- **Controls:** `epoch_baseline`, `step_baseline`, `duplicate_real`, `vae_augmented`; same `fit_condition` as Phase 7, same batch size 64, Adam lr 0.001, 10 forecaster epochs to define update budgets, 30 VAE epochs, latent 16, ratio 1:1. VAE sees **only the standardized selected training segment**. No validation/test influence on generator training. Checkpoint selected by validation MSE only.
- **Pilot:** one fixed group, `nonlinear_ar`, series 0, regime 0.10, model seed 0, pool seed 0. It is a **technical smoke/pilot**, not a basis for choosing which families/series to report.
- **Full fixed initial matrix:** 2 families × 3 series × 2 regimes × 1 architecture × (3 controls + 1 VAE) = **48 forecaster fits**. Each of 12 groups has four fits. There are **12 paired VAE comparisons** for this initial grid (one per independent series/regime). No post-hoc selection of favorable series. Optional follow-up rep seeds 0,1,2 and pool seeds 0,1 must apply to **the entire** grid, not just winners.
- **Default optimizer budgets:** For T=12,000, selected 10% train contains 840 time points and 721 complete windows: epoch baseline = 120 steps, matched conditions = 230. Full train contains 8,281 windows: epoch baseline = 1,300 steps, matched = 2,590. These are training *steps*, not total compute parity; generator training + sampling measured separately.

## Exact data-generating processes

1. **`nonlinear_ar`:** $x_t = 0.67\tanh(1.5x_{t-1}) - 0.23\tanh(x_{t-2}) + 0.16\tanh(2.7x_{t-1}x_{t-2}) + 0.28\epsilon_t$, where $\epsilon_t\sim N(0,1)$.
2. **`markov_ar`:** a latent two-state Markov process $s_t\in\{0,1\}$ with $P(s_t=s_{t-1}|s_{t-1}=0)=0.985$, and $P(s_t=s_{t-1}|s_{t-1}=1)=0.978$. Conditioned on state, $x_t=\mu_{s_t}+\phi_{s_t}(x_{t-1}-\mu_{s_t})+\sigma_{s_t}\epsilon_t$, where $(\mu_0,\phi_0,\sigma_0)=(-0.7,0.80,0.23)$ and $(\mu_1,\phi_1,\sigma_1)=(1.1,0.38,0.48)$. Initial state 0; burn-in 500. Latent states are **not** revealed to predictors.

There are no exogenous periodic priors or deterministic Fourier basis components. This does **not** imply these stochastic processes lack power spectra; 'non-Fourier' only describes the **generating mechanisms**.

## Additional simple forecast adequacy references

For every test set, report test MSE of (i) persistence, (ii) constant training-target mean, and (iii) direct regularized AR(2) to the 24 horizon outputs. AR(2) parameters fit **training windows only**, without test-label access. These references are not optimizer-step-matched neural arms: they answer the distinct question of whether the forecasters are competitive with simple models. Do not silently exclude these if learned methods perform poorly.

## Setup — Windows PowerShell in repository root

1. Save the `.patch` file next to the repository or use its full path.
2. Ensure clean/known Git status, then apply:

```powershell
conda activate sli_2
git status --short --branch
git apply --check .\ASOC_Phase12_NonFourier_FixedDGP_2026-11-11.patch
git apply .\ASOC_Phase12_NonFourier_FixedDGP_2026-11-11.patch
python -m unittest discover -s tests -p 'test_asoc_nonfourier.py' -v
```

Tested on synthetic arrays and a **mocked neural trainer**, not on a GTX 1080. CUDA build must still be checked on the chosen machine:

```powershell
python -c "import torch; print(torch.__version__,torch.version.cuda); print(torch.cuda.is_available()); x=torch.ones(8,device='cuda:0'); print(x.sum().item())"
```

### 1. Technical pilot, no writes during dry-run

```powershell
python -m scripts.asoc_nonfourier --families nonlinear_ar --ids 0 --regimes 0.10 --reps 0 --pool-seeds 0 --dry-run
# Expected: 1 incomplete group, 4 new forecaster runs.
python -m scripts.asoc_nonfourier --families nonlinear_ar --ids 0 --regimes 0.10 --reps 0 --pool-seeds 0 --limit 1 --gpu-index 0
```

### 2. Pre-fixed full matrix (only AFTER technical pilot checks)

```powershell
python -m scripts.asoc_nonfourier --families nonlinear_ar,markov_ar --ids 0,1,2 --regimes 0.10,1.00 --reps 0 --pool-seeds 0 --dry-run
# Expected after pilot: 11 incomplete groups, 44 new runs.
python -m scripts.asoc_nonfourier --families nonlinear_ar,markov_ar --ids 0,1,2 --regimes 0.10,1.00 --reps 0 --pool-seeds 0 --gpu-index 0
python -m scripts.asoc_nonfourier_report
```

The script is **resume-safe**: it writes each condition once and skips existing exact run keys. Incomplete runs remain visible. Run with `--limit 1` to bound work to one incomplete group; rerun the same command to resume. Array contents are deterministically regenerated, checked against existing Phase 12 arrays, and never overwritten. All new output stays in `results/asoc_revision/nonfourier/`. This does not mutate original Fourier, Phase 7–11 or real-data files.

### 3. Results expected

- `results/asoc_revision/nonfourier/nonfourier_runs.csv` — all individual fits, training cost and provenance
- `results/asoc_revision/nonfourier/report/nonfourier_paired.csv` — equal-step paired comparisons and simple references
- `results/asoc_revision/nonfourier/report/nonfourier_per_pool.csv` — aggregate model initializations **within** synthetic pool
- `results/asoc_revision/nonfourier/report/nonfourier_per_series.csv` — aggregate pool results **within** independent series
- `results/asoc_revision/nonfourier/report/nonfourier_summary.csv` — per family/regime/architecture aggregate **across independent series**
- `results/asoc_revision/nonfourier/report/nonfourier_incomplete.csv` — paired VAE rows missing a control, if any

The summary uses *nested descriptive medians*, not independent-window p-values, confidence intervals or evidence of statistical significance. Compare **within-series paired percentage changes**, not raw MSE between different DGP families. Keep the full predetermined matrix (including null/negative results) and explicitly report simple-reference competitiveness and generator runtime.

## Integrity checks

- All six realizations and both regimes remain in the final analysis independent of outcomes.
- Fail on duplicate run keys, changed prepared arrays, unmatched optimizers, missing comparators or wrong raw-scale conversions.
- No future/test labels are used for generator training, scaler fitting, neural checkpoint selection or AR(2) fitting.
- Training windows are fully contained within the training interval; validation/test use their own label intervals and historical context from the preceding lookback.
- Model seed and VAE pool seed are separate axes; each generated series has an independently seeded DGP realization. Nested family analysis treats the *series* as the only independent experimental replicate; no pseudo-replication across overlapping windows.
- Full 1.00 training budget is longer than scarce 0.10: do not claim equal wall time *across regimes*; compare matched arms **within each regime**.
- Failures are a legitimate and informative result. Changing DGP parameters after inspecting test outcomes would require a *new, separately versioned exploratory experiment*, not an edit to this grid.

## Reviewer-appropriate positioning

> To test whether the observed synthetic-augmentation effects were specific to the original Fourier-mixture generator, we conducted an additional, pre-specified controlled simulation on independently seeded nonlinear autoregressive and hidden-regime-switching autoregressive processes. For each process realization, training regime, and model initialization, we compared VAE augmentation against epoch-limited, step-matched real-only, and duplicated-real controls using the same forecast architecture and validation-based checkpoint selection. We further reported simple forecast references to distinguish relative augmentation effects from absolute forecasting adequacy. Results were aggregated at the independent-series level rather than treating overlapping windows or pool seeds as independent experimental units.

Insert numeric findings **only after all results are collected**. These two families improve external structural variation but cannot establish universal generalization beyond these fixed DGPs.
