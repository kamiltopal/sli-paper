# Phase 7 add-on — Weather 10% fixed-budget augmentation-ratio sensitivity

**Purpose:** address R3.3/R5.5 without re-running Traffic or existing 1x Weather experiments. This is **not** a new real-domain replication and does **not** retroactively validate H2–H4.

## Locked protocol

- Source: existing `data/real/weather.npy`, Weather target `OT`, **target-only** iTransformer-lite; original source timestamp CSV must exist.
- Domain scope: Weather 10% *most recent training span*; train/validation/test chronology as original Phase 7, lookback 96, horizon 24. The two known timestamp anomalies at May 12/29, 2020 are **outside** the active 10% train/validation/test span. Script checks this again before training.
- Ratios: **0.5×, 1×, 2× synthetic-to-real training windows**. Existing 1× is imported **read-only** from `results/asoc_revision/real_budget_controls.csv`; VAE is not retrained for that ratio.
- Same **1,120 optimizer steps** for *each* new forecaster run, independent of synthetic ratio. Same LR 1e-3, batch 64, **validation every 112 steps**, minimum validation-MSE checkpoint, evaluation on fixed test span.
- Controls: same-seed `step_baseline` from original 1× Phase 7 (1,120 steps) and **ratio-matched duplicated-real**, each also 1,120 steps. An epoch baseline with 560 steps is reported only for context, never described as budget-matched.
- Generator: unchanged original `src.generators.gen_vae`, VAE 30 epochs, latent 16, pool seeds 0 and 1. A generator is trained per pool/ratio and saved as a `.npz` artifact, then reused over forecaster seeds 0–2 to avoid pool identity drift during resume.
- Ratio-matched duplicated-real: 0.5× samples a fixed half of real windows (without replacement); 2× adds two complete copies of real training windows. This differs in data diversity, not optimizer count. It is a fairer control for **quantity**, not a complete substitute for a realistic augmentation generator.
- Independent units: only **two** generator pools in a **single domain**. Three forecaster seeds do **not** turn them into six independent domains. Report descriptive nested medians/ranges, no unearned CI or significance claims.
- *Sample exposures differ with ratio* even though optimizer steps and validation frequency are fixed. Total generator+forecaster compute is **not** matched. This is a fixed-step **ratio sensitivity**, not a fully compute-matched design.
- This is an **exploratory follow-up with a fixed ratio grid**. Weather was selected after observing its favorable 1× result; the new grid is locked before its own runs. Do not select a ratio on the held-out test and call it a validated deployment policy; policy selection must use validation or fresh out-of-sample data.

## Commands (Windows / PowerShell, repo root, conda `sli_2`)

```powershell
python -m unittest discover -s tests -p "test_asoc_real_ratio_weather.py" -v
python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0 --dry-run
# Pilot for implementation verification; four new model fits (2 duplicates + 2 VAE):
python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0
python -m scripts.asoc_real_ratio_weather --reps 0 --pool-seeds 0 --report
# Continue the fixed 3 forecaster seeds x 2 independent VAE pools:
python -m scripts.asoc_real_ratio_weather --reps 0,1,2 --pool-seeds 0,1 --dry-run
python -m scripts.asoc_real_ratio_weather --reps 0,1,2 --pool-seeds 0,1
python -m scripts.asoc_real_ratio_weather --reps 0,1,2 --pool-seeds 0,1 --report
```

The completed full grid creates **18 new forecaster fits**, 20,160 new optimizer steps, and **four distinct VAE generator pools** (two per new ratio); the 1× runs are re-used. Following the pilot, only **14** new fits remain if all four pilot fits finish. Do not run an extra 1× Phase7 matrix.

## Outputs — only under `results/asoc_revision/`

- `ratio_weather_v1.csv`: resumable raw new arm results, 0.5×/2× only.
- `ratio_weather_pools_v1/*.npz`: SHA-tracked, frozen synthetic pools for cross-seed reuse.
- `ratio_weather_report_v1/ratio_paired.csv`: one row per ratio/model seed/pool seed; includes the old 1× rows as a *reference*, not new training.
- `ratio_weather_report_v1/ratio_per_pool.csv`: first aggregate model training seeds **within** each synthetic pool.
- `ratio_weather_report_v1/ratio_summary.csv`: then aggregate pool-level medians/ranges.
- `ratio_weather_report_v1/ratio_incomplete.csv`: missing ratio-matched new contrasts.

**Interpretation:** e.g. a ratio could beat its ratio-matched duplicated-real control and still lose to the fixed-step original baseline. Report both. Larger ratios can be worse through changing exposure/sample composition despite equal optimizer steps. If 0.5× or 2× are better than 1×, treat it as descriptive on Weather, not a learned global policy.

## Reviewer response draft (provisional, do not claim experiments until results exist)

> **R3.3 / R5.5 — Augmentation-ratio sensitivity.** We thank the reviewer for identifying the dependence on augmentation ratio. We added an exploratory 0.5×/1×/2× synthetic-to-real window-ratio study, selected after observing the favorable 1× Weather result, under the 10% training-history regime using the target-only iTransformer configuration. Importantly, all ratio arms are trained for the same number of optimizer steps with the same validation-checkpoint schedule, and we include a real-window duplication control matched to each synthetic ratio. We reuse the existing 1× runs and independently seeded VAE pools, separating generator-pool from forecaster-initialization variation. We report absolute/relative MSE and MAE and limit interpretation to this domain and configuration. **[Insert actual numerical results after completion.]** Longer lookback and forecast-horizon sensitivity remain a separately scoped request and are not addressed by this ratio experiment alone.
