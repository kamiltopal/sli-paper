# ASOC Major Revision — Phase 11: EPİAŞ + Traffic reliability expansion

**Date:** 11 November 2026
**Branch:** `revision/asoc-major-2026`
**Manuscript deadline:** 5 December 2026
**Status:** Protocol and read-only preflight implemented; **no new GPU training executed by this package**.
**Scientific question:** Does the apparent 10%-regime VAE augmentation effect generalize beyond Weather/ETTh1, and is the negative Traffic observation stable across model and generator seeds?

## Scope and locked analysis plan

- Same Phase 7 target-only iTransformer, `lookback=96`, `horizon=24`, `epochs=10`, `vae_epochs=30`, `batch_size=64`, `lr=0.001`, `latent=16`, `input-mode=target`.
- Four forecaster arms: `epoch_baseline` (shorter optimizer budget), `step_baseline` (real only, matched updates), `duplicate_real` (reused real windows, matched updates), `vae_augmented` (1:1 synthetic/real, matched updates).
- **EPİAŞ PTF** (market-clearing price, TL/MWh), 10% history: pilot `rep=0`, pools `0,1` = **5 forecaster runs** (three shared controls + two VAE). If preflight passes and the pilot is technically valid, complete prespecified `reps=0,1,2`, `pool-seeds=0,1` = **10 more forecasters**. Keep findings regardless of sign.
- **Traffic**, 10% history: previously completed `rep=0`, pools `0,1` (five forecasters). Add `rep=1,2` across both VAE pools = **10 new forecasters**. No rerun of existing Traffic, and no Traffic 100% repeat. If partial rows already exist, CLI pending counts change; never force a fixed total.
- Do not select additional data domains or seed values based on whether the pilot is favorable. Pool seeds are **not independent test sets**. Describe two-level medians (forecaster seeds within pool, then independent synthetic pools), provide per-pool results and absolute MSE/MAE.
- EPİAŞ and Traffic can be compared on **within-domain VAE-minus-control paired percent differences**, not directly on absolute normalized MSE across unrelated targets/scales. Equal optimizer steps do **not** equal total compute cost.

## Step A — apply patch and verify tests (PowerShell, repo root)

```powershell
conda activate sli_2
git switch revision/asoc-major-2026
git status --short --branch
git apply --check .\ASOC_Phase11_Epias_Traffic_Preflight_2026-11-11.patch
git apply .\ASOC_Phase11_Epias_Traffic_Preflight_2026-11-11.patch
python -m unittest discover -s tests -p 'test_asoc_epias_preflight.py' -v
```

The patch adds **only** `scripts/asoc_epias_preflight.py`, `tests/test_asoc_epias_preflight.py`, and this document. No existing source, `.npy` data, or results CSV is modified.

## Step B — EPİAŞ 10% provenance and chronology gate (NO TRAINING)

Run with the *same data/results directory* used for prior Phase 7 experiments:

```powershell
python -m scripts.asoc_epias_preflight --regime 0.10
```

Expected keys: `pass: true`, `raw_matches_prepared: true`, finite `(T,3) float32`, `prepared_sha256`, `source_sha256`, `split`, `time_audit`, `existing_phase7_epias_rows`. Actual row counts and optimizer budgets are determined by your local data — **do not assume** example sizes. Save the JSON text and share it for review. If `pass: false`, **stop EPİAŞ training** and examine `errors` and `irregular_examples`. The audit never modifies the source CSV or prepared array.

The original `scripts.prepare_real.load_epias` drops duplicate clock times (`keep='first'`) but **does not impute missing hourly records** or sort time. The preflight mirrors this logic in memory and requires exact equality to your existing `.npy`. Raw timestamps are naive local clock times; inspect possible DST or source policy changes rather than silently filling hours. A global irregularity strictly **outside** the selected 10% training/validation/test segment is disclosed but does not automatically block the 10% pilot. An irregularity inside the selected time span *does* block it. For a 100% run, repeat the gate using `--regime 1.00` before training; passing 10% does not validate 100%.

**Important**: Do **not** execute `python -m scripts.prepare_real`, which overwrites `.npy` files for multiple domains.

## Step C — Traffic expansion without reruns

First, check CUDA on the machine that will actually run the experiment:

```powershell
python -c "import torch; print('PyTorch',torch.__version__,'CUDA',torch.version.cuda); print('Available',torch.cuda.is_available()); print('Devices',[(torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]); x=torch.ones(10,device='cuda:0'); print('CUDA kernel OK',float((x*x).sum()))"
```

If the GTX 1080 is on a *different computer*, the original prepared arrays **and** the complete `results/asoc_revision/real_budget_controls.csv` need to be copied before starting. Verify SHA hashes and plan; **never merge two independently appended versions of that CSV by blindly overwriting**, and don't have two machines append to the same CSV concurrently. The GPU index is `0` if the GTX 1080 is the only CUDA GPU on that computer; it is not necessarily index `1`.

Plan all Traffic rep seeds; expect `10 new rows` **only if** the original five rep0 runs are present and there are no partial new runs:

```powershell
python -m scripts.asoc_real_matrix --domains traffic --regimes 0.10 --archs itransformer --input-mode target --reps 0,1,2 --pool-seeds 0,1 --gpu-index 0 --dry-run
```

Execute staged additions (both rep seeds are predeclared regardless of outcome):

```powershell
python -m scripts.asoc_real_matrix --domains traffic --regimes 0.10 --archs itransformer --input-mode target --reps 0,1 --pool-seeds 0,1 --gpu-index 0
python -m scripts.asoc_real_matrix --domains traffic --regimes 0.10 --archs itransformer --input-mode target --reps 0,1,2 --pool-seeds 0,1 --gpu-index 0
```

If the dry run shows rep0 controls/VAE are missing, pause: the prior results may not be present on that machine. The `asoc_real_matrix` implementation checks SHA changes in existing domain rows before running and is resume-safe when the original results file is intact.

## Step D — EPİAŞ 10% pilot ONLY AFTER Step B passes

```powershell
python -m scripts.asoc_real_matrix --domains epias --regimes 0.10 --archs itransformer --input-mode target --reps 0 --pool-seeds 0,1 --gpu-index 0 --dry-run
python -m scripts.asoc_real_matrix --domains epias --regimes 0.10 --archs itransformer --input-mode target --reps 0 --pool-seeds 0,1 --gpu-index 0 --limit 1
python -m scripts.asoc_real_report
```

Review the new EPİAŞ rows of `results/asoc_revision/real_report/real_summary.csv` and `real_per_pool.csv`; sign and magnitude are **unknown until the experiment runs**. If data were modified, create a **versioned, separately named** `.npy` and results CSV with a documented sensitivity analysis — never silently replace the original array or claim comparability to existing rows.

Then, if the EPİAŞ pilot passed technical checks (finite metrics, correct dataset digest, valid step counts, consistent train split, correct target, checkpoint logging), execute the two preplanned additional model seeds:

```powershell
python -m scripts.asoc_real_matrix --domains epias --regimes 0.10 --archs itransformer --input-mode target --reps 0,1,2 --pool-seeds 0,1 --gpu-index 0 --dry-run
python -m scripts.asoc_real_matrix --domains epias --regimes 0.10 --archs itransformer --input-mode target --reps 0,1,2 --pool-seeds 0,1 --gpu-index 0
python -m scripts.asoc_real_report
```

## Step E — reporting gate and manuscript integration

- Verify traffic `n_pool_seeds=2`, `n_training_reps_min=max=3`, `n_pairs=6` for the 10% iTransformer target-only summary. EPİAŞ should show the same values once its expansion is complete. **Check actual pending row counts**, not just desired totals.
- Inspect any `vae_best_at_last` fraction, nonfinite/zero errors, generator duration and GP(U)/PyTorch metadata. `real_summary.csv` summarizes pairs, not independent domains; avoid an unwarranted p-value.
- For manuscript use: compare VAE against **both matched controls** with MSE and MAE; show negative/positive findings, raw-scale MSE/MAE, regime-specific normalization, and a caveat that adjacent forecast origins overlap.
- Existing Phase-7 output contains aggregate metrics, not per-origin losses. A HAC/bootstrap confidence interval cannot be inferred from two generator pools or from `real_summary.csv` alone. New forecast checkpoints/prediction logging would require a separately controlled amendment.
- Copy output CSV(s) to a backup before migrating between devices; preserve all pre-existing Weather/ETTh1/Traffic rows in place.

## Stop criterion

After EPİAŞ 10% and Traffic 10% have 3 model seeds × 2 pool seeds, prioritize analysis and manuscript writing. Do not expand Traffic to 100% or launch a fresh generator family unless the original reviewer requirement and timetable justify it. Planned manuscript submission: **5 December 2026**.
