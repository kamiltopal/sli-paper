# ASOC Phase 16: approximate wall-clock-matched pilot

## Question and frozen design
Does the Phase 14 TimeGAN advantage remain when its **fresh generator fit + sampling + forecaster fit/validation elapsed wall time** is reallocated to real-only, duplicated-real, or VAE forecasting? This is an additive pilot, not a replacement for the equal-step Phase 7/14/15 studies. Weather and Traffic; regime 0.10; target-only iTransformer; 96/24 windows; seed 0, pool seed 0; batch 64; Adam lr .001. TimeGAN uses the unchanged Phase 14 GEN settings. VAE uses the unchanged Phase 7 generator, 30 epochs, latent 16, ratio 1. No test metric enters time allocation, stopping, or checkpoint selection.

Each domain has four sequential arms on the same CUDA device in one uninterrupted process, in fixed order: TimeGAN (fresh pool and 1,120 Weather / 350 Traffic optimizer steps), real-only, duplicated-real, VAE (fresh pool). The TimeGAN budget is measured separately in each domain. The remaining arms spend the measured target on their own generator (if any) and forecaster. All arms use the same validation checkpoint interval: ceil(2*Nreal/64), 112 Weather and 35 Traffic. Select minimum validation MSE over all visited checkpoints. Longer training means more validation opportunities; that is part of the alternative budget allocation. At a completed checkpoint, stop when the elapsed budget reaches/exceeds the target. Record actual elapsed and overshoot. Relative discrepancy <=5% qualifies as approximately matched; otherwise report the row but do not call the pair budget-matched. No performance-driven rerun or tuning. Eight new fits, two fresh pools per generator across domains; no reuse of historical generation times.

## Accounting
Before each arm: two discarded real-only warmup optimizer steps (not in budget). Start synchronized wall clock before generator initialization. Include fresh generator training/sampling, arm-specific data concatenation, forecaster initialization, training, validation and best-state copying. End after final validation and device synchronization. Common loading/splitting/standardization, warmup, post-selection final test evaluation and artifact writes are excluded. Record test evaluation separately, generator time (including generator setup), forecaster loop time, total budgeted time and actual optimizer steps. CUDA synchronization makes these host wall-clock measurements, not FLOPs, energy, GPU-hours, or full application latency. Fresh pools are saved rather than reusing Phase 14/15 pools. Source fingerprints, reference and original Phase 14 integrity gate inherited from Phase 15; original files untouched. Fixed order and a single seed/device/session per domain mean thermal/load/order effects and seed variation cannot be ruled out. Run on the same GPU, without other GPU jobs, for both domains; do not compare absolute durations across different hardware.

## Run (PowerShell, repo root, existing Phase 14/15 patches installed)

```powershell
 git apply --check .\ASOC_Phase16_Compute_Budget_2026-11-12.patch
 git apply .\ASOC_Phase16_Compute_Budget_2026-11-12.patch
 python -m unittest discover -s tests -p "test_asoc_phase16_compute.py" -v
 python -m scripts.asoc_phase16_compute --dry-run
 python -m scripts.asoc_phase16_compute --smoke --gpu-index 0
 python -m scripts.asoc_phase16_compute --domains weather --gpu-index 0
 python -m scripts.asoc_phase16_compute --domains weather --report
 python -m scripts.asoc_phase16_compute --domains traffic --gpu-index 0
 python -m scripts.asoc_phase16_compute --report
```

The smoke uses tiny synthetic data and two training stopping rules, not the real experimental budget. Eight unit tests use CPU/test doubles and check fixed/timed stopping, validation-only checkpointing, origin aggregation, tolerance, group persistence/tamper detection, and reporting. The real-data preflight needs the same local raw/prepared data and complete original Phase 14 tree as Phase 15. `--dry-run` should show Weather 4 / Traffic 4 pending, then 0 each after completion. Do not modify Phase 14/15 source or historical files to bypass a failed provenance check. `--report` runs no training. An interrupted domain stays `<domain>.inprogress`; preserve it (rename to an archival name or use new --outdir), rerun the full domain group so comparisons share a session. Completed domains are validated and skipped. Do not run concurrent writers. All final rows including out-of-tolerance and unfavorable results are retained.

## Outputs
`results/asoc_revision/phase16_compute/<domain>/runs.csv`, `origins/*.npz`, `pools/*.npz`; `report/runs.csv`, `report/comparisons.csv`. Compare TimeGAN against each arm using normalized MSE/MAE, raw MSE/MAE in runs, budgeted seconds, actual steps, budget delta and tolerance flag. Primary interpretation is within-domain descriptive, not a significance claim. Historical equal-step MSE is a diagnostic reference, not a budget-matched comparator. A matched pilot in two domains is not sufficient to establish population-level cost-efficiency or hardware-invariant superiority. Archive the complete Phase 16 directory and report before manuscript integration.
