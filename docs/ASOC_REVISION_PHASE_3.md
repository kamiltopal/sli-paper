# Applied Soft Computing major revision — phase 3: channel/pairing controls

## Purpose and scope

The original `gen_vae` synthesizes the target channel using an unconditional window VAE, then assigns auxiliary windows from randomly sampled real training windows. Consequently, training examples that combine the generated target and random auxiliaries may contain **incorrect cross-channel relationships**. Reviewers 1, 2, 3 and 5 asked for controls separating this from the channel-mixing architecture's behaviour.

Phase 3 is an **additive, targeted diagnostic**. It leaves `src/generators.py`, `src/models.py`, the original 12,294-run results, and Phase 1–2 outputs unchanged. All evaluations use the same real validation/test split, initial model seed, optimizer step count, and checkpoint schedule.

**Scientific caveat:** A post-hoc nearest-context match is **not** genuine time alignment or a validated conditional generator. An unconditional VAE does not preserve the original target–covariate joint distribution by construction. This ablation estimates *sensitivity to pairing choice*, not the causal effect of the attention mechanism in isolation. Rebuilding a joint conditional VAE (or a documented oracle-aligned control) would require a separate, broader experiment.

## Eight equal-step conditions (iTransformer only)

| Label | Input channels | Added pool |
|---|---|---|
| `itr_target_step` | target only | none, cycle real windows for matched updates |
| `itr_target_duplicate` | target only | duplicated real windows |
| `itr_target_vae` | target only | generated target windows (VAE) |
| `itr_all_step` | target + 2 auxiliaries | none, cycle real windows for matched updates |
| `itr_all_duplicate` | target + 2 auxiliaries | duplicated real windows |
| `itr_all_duplicate_permuted` | target + 2 auxiliaries | duplicated real targets; **only added-half auxiliaries** randomly permuted |
| `itr_all_vae_random` | target + 2 auxiliaries | original VAE targets + randomly paired real auxiliary windows |
| `itr_all_vae_context` | target + 2 auxiliaries | **same VAE targets** + real auxiliary windows matched by nearest *training target lookback* context |

The nearest-context match uses 12 non-overlapping target lookback bin means and standard deviations, plus the last level and full-window difference. Its feature scale, reference KD-tree, and auxiliary windows are derived only from the regime's real training windows. Neither validation nor test information is used for this mapping. The generated target and horizon labels are identical across `itr_target_vae`, `itr_all_vae_random`, and `itr_all_vae_context` within a cell.

The permuted-real negative control isolates the effect of deliberately corrupting auxiliary alignment **without generating a single synthetic target**. Correlation-gap diagnostics compare the real and augmented training target–auxiliary correlations. Those are descriptive association checks, not evidence of future-predictive validity.

## Installation and tests

Prerequisite: Phase 1 (`scripts/asoc_budget_controls.py`, plus CUDA update) applied. Phase 2 is independent and may be present locally. On the `revision/asoc-major-2026` branch, apply only the Phase 3 patch:

```powershell
 git apply --check .\asoc_phase3_channel_pairing.patch
 git apply .\asoc_phase3_channel_pairing.patch
 python -m unittest discover -s tests -p "test_asoc_channel_controls.py"
```

## GPU pilot

Use the MX130 GPU. CUDA is required by default; `--allow-cpu` is for testing only. Start with **one** experimental cell so the memory impact of iTransformer is known:

```powershell
python -m scripts.asoc_channel_matrix --dry-run
python -m scripts.asoc_channel_matrix --levels 0.5 --series-seeds 0 --regimes 0.10 --reps 0 --limit 1 --gpu-index 0
python -m scripts.asoc_channel_report
```

Expected output:

- `results/asoc_revision/channel_controls.csv` — all raw condition/validation/step records
- `results/asoc_revision/channel_report/channel_paired.csv` — matched test-MSE contrasts
- `results/asoc_revision/channel_report/channel_summary.csv` — descriptive aggregate
- `results/asoc_revision/channel_report/channel_incomplete.csv` — partial cells (e.g., after a CUDA/Windows abort)

Then expand if the pilot succeeds:

```powershell
python -m scripts.asoc_channel_matrix --levels 0.2,0.5,0.8 --series-seeds 0 --regimes 0.10 --reps 0,1,2 --gpu-index 0
python -m scripts.asoc_channel_report
```

This expansion is 3 series configurations x 3 training repetitions x 8 conditions = 72 trainings, **not** 72 independent series. For robust inference, extend to distinct backbone seeds and variants later, and consider %25 and %100 regimes. The matrix runner executes one cell per subprocess, resumes complete conditions by CSV key, and avoids overwriting older paper results.

## Interpreting the comparisons

- `all_duplicate_permuted_vs_duplicate_pct` directly checks harm from misaligned **real** auxiliaries.
- `all_vae_context_vs_random_pct` changes only the pairing procedure for the **same VAE target pool**.
- `target_vae_vs_duplicate_pct` estimates VAE-target influence without auxiliary information (within iTransformer).
- `all_duplicate_vs_target_duplicate_pct` estimates the value of auxiliary inputs under a real-only, step-matched training condition.
- Contrasts involving `itr_all_*` versus `itr_target_*` compare **input configurations**, not a pure channel-attention mechanism effect.

Negative percentage change = first condition lower MSE. Do not use results from this pilot to retrospectively choose a new primary H3 hypothesis. Keep any revised mechanistic interpretation appropriately qualified and explain exploratory versus pre-specified experiments in the paper and reviewer response.
