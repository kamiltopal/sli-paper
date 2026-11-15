# ASOC Phase 15 — matched origin-level reference replay

**Status:** locked follow-up protocol, no new GPU results yet. Additive: never changes historical Phase 7/14 results or prepared arrays. Requires the exact Phase 14 source files and complete 12-row Phase 14 results, four original TimeGAN pools and twelve origin traces. Run from repository root in the same `sli_2` environment. Preserve files and use one writer.

## Question and design

Does the observed Phase 14 TimeGAN contrast remain directionally stable when matched comparators are independently replayed and origin-level loss differences are assessed with temporal dependence? The original Phase 7 reference traces were never persisted; Phase 15 does **not** recover them. It replays step-only and duplicated-real forecasters and creates **new, persisted** VAE pools with the original target-only VAE code, 30 epochs, latent 16 and pool seeds 0,1. All runs are versioned separately; even exact aggregate agreement would not establish identity of the original VAE pool. TimeGAN is not retrained. A replay can differ from history due to stochastic/hardware differences; retain every valid result and report historical-vs-replay differences, never selectively replace an unfavorable run.

| Domain | Step-only | Duplicate | VAE replay | Total | Matched updates per fit |
|---|---:|---:|---:|---:|---:|
| Weather | 3 | 3 | 6 | 12 | 1120 |
| Traffic | 3 | 3 | 6 | 12 | 350 |

Model seeds 0,1,2; VAE pool seeds 0,1, reused across all three forecasters. Same 10% training-history slice, 96/24 target-only iTransformer, original Phase 7 chronological windows, train-only standardization, 1:1 augmentation or duplication, batch 64, Adam lr .001, 10 augmented epoch equivalents, validation checkpoint schedule and selection. Step-only and duplicate controls are shared, stored once per model seed. Epoch-only is not replayed because primary comparisons are equal-step. The original 12 TimeGAN fits remain frozen. New rows are replay-only; do not append them to historical `real_budget_controls.csv`.

## Statistics (declared before replay outcomes)

For each domain, comparator and metric (MSE/MAE), align the first-target index and form TimeGAN minus replayed comparator losses for each of six model/pool comparisons. Negative favors TimeGAN. Average the six differences **at each origin**; this weights the two pools and three seeds equally without pretending they are six independent test samples. Step/duplicate controls may be reused across pools; bootstrap this aggregate time series rather than resampling six pairs as independent observations. Also report all pair-level mean and percent contrasts. These mean-based estimates are not the hierarchical median effect used in Table 12.

Use circular moving-block bootstrap percentile intervals (2,000 draws, RNG seed 1501), with identical time-block sampling across paired differences (equivalently sample the pre-averaged difference series). Weather block lengths 24/144/288 origins; Traffic 24/168/336. Primary lengths 144 Weather (one day), 168 Traffic (one week); alternatives assess sensitivity to dependence and block selection. Weather daily cycle is 144 10-minute observations; Traffic weekly cycle is 168 hourly observations. Report all intervals, not only favorable ones. Blocks preserve local dependence but wrap the boundary; validity requires approximate stationarity/weak dependence of the loss-difference process and is not guaranteed by the block lengths. Intervals are conditional on these fitted models, these two selected domains and their single held-out test periods. They do not capture domain sampling, generator/forecaster uncertainty across new fits, temporal regime shifts, or establish a universal effect. No independent-window t-tests; no p-values or multiplicity-adjusted confirmatory claims. Primary MSE and secondary MAE; exploratory comparisons. If stationarity is questionable or conclusions vary with block length, disclose the instability instead of claiming significance. Full forecasts/checkpoints are still not stored: only per-origin average MSE/MAE.

## Apply and run (PowerShell)

First preserve the actual Phase 14 sources in Git, if not yet committed. Do not change their bytes, as config IDs include source hashes. `git status --short` and inspect changes; commit the four Phase 14 files explicitly (not generated outputs or patches). Then apply Phase 15:

```powershell
git apply --check .\ASOC_Phase15_Matched_Origins_2026-11-12.patch
git apply .\ASOC_Phase15_Matched_Origins_2026-11-12.patch
python -m unittest discover -s tests -p 'test_asoc_phase15_origins.py' -v
python -m scripts.asoc_phase15_origins --dry-run
```

Dry-run reads original raw timestamp CSVs, prepared arrays, Phase 7 reference, and all Phase 14 pools/losses. Verifies source/config and content digests; it writes nothing. Expected pending: 12 Weather, 12 Traffic (24 total), if no Phase 15 rows exist. Any mismatch is a stop condition. Do not modify data, source or historical results to bypass gates. Phase 14 file names/paths and raw CSV bytes must be intact. CPU test fixture cannot replace the real GPU preflight.

Pilot first real-only replay, then complete all predeclared runs regardless of result:

```powershell
python -m scripts.asoc_phase15_origins --domains weather --limit 1 --gpu-index 0
python -m scripts.asoc_phase15_origins --domains weather --gpu-index 0
python -m scripts.asoc_phase15_origins --domains traffic --gpu-index 0
python -m scripts.asoc_phase15_origins --report
```

If interrupted, dry-run again and resume; completed rows are hash-checked and skipped. A missing or changed referenced VAE pool/origin file causes failure. The script locks the relevant source hashes and reference/TimeGAN run CSV hashes in the configuration ID. Do not move across machines or change sources mid-campaign without versioning a new output directory. Cache retains the exact new VAE pool, its content SHA, generator environment and stage duration. Use `--outdir` only for an explicitly separate version, never for the old Phase 14 directory.

## Outputs

`results/asoc_revision/phase15_origins/runs.csv` (24 new fits, historical comparator aggregate and replay-minus-historical MSE recorded), `pools/*.npz` (four new VAE pools), `origins/*.npz` (24 new loss traces), `report/paired.csv`, `report/domain_means.csv`, `report/block_bootstrap.csv`. Full report requires all 12 replay rows per requested domain. Negative mean differences favor TimeGAN. Report is conditional on replay comparators, not original historical traces. Do not combine output with the earlier release as if already verified; preserve hashes and update evidence manifest after completion.

No total generator+forecaster compute equality is achieved here. Conditional generator comparison and a validated prospective selector also remain open. This phase targets R1.8/R2.7 uncertainty and makes the TimeGAN contrast more auditable, not every remaining reviewer request.
