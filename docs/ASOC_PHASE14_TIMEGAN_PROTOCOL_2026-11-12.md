# ASOC Phase 14 — compact TimeGAN, fixed-step comparison

Date: 12 November 2026. Base: `revision/asoc-major-2026`, `a62162bc59e8a70fbdf31d00e0e532fb9161a6fa`. Additive patch: two scripts, tests, this protocol. No historical source/data/result file is changed. No GPU run or new real-domain result is claimed by this package.

## Scientific scope and locked settings

Reference: Yoon, Jarrett and van der Schaar (2019), *Time-series Generative Adversarial Networks*, NeurIPS 32. [Paper](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html), [authors' implementation](https://github.com/jsyoon0823/TimeGAN). This is an independently written compact PyTorch GRU adaptation of the temporal-only TimeGAN architecture/objectives, not an exact replication of the original TensorFlow benchmark. No static features or external covariates, no conditional generator, no hyperparameter optimization. The resource-limited schedule is not evidence of a converged or state-of-the-art TimeGAN benchmark.

| Item | Locked value |
|---|---|
| Domains | Weather, Traffic, in that order |
| Regime | 0.10, target-only iTransformer |
| Input/target lengths | 96/24 (120-point synthetic joint target windows) |
| Chronology | same original Phase 7 train/validation/test partitions and train-only standardization; raw active timestamps required |
| Augmentation | N synthetic + N real windows |
| Forecaster | original architecture, Adam, lr 0.001, batch 64, 10 augmented epoch equivalents, minimum validation MSE checkpoint |
| Seeds | model 0,1,2; pool 0,1 per domain |
| Generator | GRU hidden 24, 3 layers (supervisor 2), uniform noise, batch 64, Adam lr 0.001; train-only min/max scaling to [0,1] and inverse scaling |
| Autoencoder / supervisor pretrain | 200 / 200 updates |
| Joint training | 400 iterations, 2 generator/supervisor + 2 embedder/recovery updates and 1 discriminator update each (fixed, no discriminator loss threshold) |
| Losses | reconstruction, next-step latent supervision, adversarial logits for real/latent synthetic, first/second moment matching |
| Expected | 4 distinct pools, 12 TimeGAN forecasters (6 per domain) |

Weather N=3570, matched updates=1120; Traffic N=1109, matched updates=350, from the verified historical archive. Local preflight must reproduce these. Weather's previously observed timestamp anomalies are outside the active 10% span; no silent repair, sorting, or data regeneration. Raw CSV must have same length as prepared array and active timestamps must be regular. This gate does not independently verify target-column equality with raw CSV; data hash is matched to Phase 7. Historical reference CSV is frozen by SHA-256; use an unchanged copy. A new reference hash or source-code change requires a new output directory, not overwriting completed runs.

## Workflow (PowerShell from repository root)

```powershell
conda activate sli_2
git switch revision/asoc-major-2026
git status --short
git apply --check .\ASOC_Phase14_TimeGAN_2026-11-12.patch
git apply .\ASOC_Phase14_TimeGAN_2026-11-12.patch
python -m unittest discover -s tests -p 'test_asoc_timegan.py' -v
python -m scripts.asoc_real_timegan --domains weather,traffic --dry-run
python -m scripts.asoc_real_timegan --smoke --gpu-index 0
```

Tests: standalone 8 passed, repository parity test skipped in the standalone distribution; it must pass in the actual repo. GPU smoke uses tiny synthetic inputs of the real 96/24 shape with two steps per stage and two iTransformer optimizer updates; no experiment output is written. `--smoke --allow-cpu` permits CPU fallback (uses CUDA if available). `--dry-run` writes nothing, checks actual arrays, active chronology and all 30 existing control/VAE reference rows. Stop if any gate fails. Do not change data to make it pass. The real campaign requires CUDA.

### First real Weather pilot (one pool and one forecaster)

```powershell
python -m scripts.asoc_real_timegan --domains weather --reps 0 --pool-seeds 0 --gpu-index 0
```

Expected one row and one saved pool. Check finite training losses, model/test shapes, generator/forecaster times and hash records. This pilot uses exactly the same locked generator and forecaster configuration as the full campaign. Keep this result regardless of direction. Severe collapse is a finding to disclose, not a reason for selecting a more favorable seed; technical exceptions need a separately recorded protocol amendment. No new generator hyperparameters should be selected using test outcomes.

### Expand regardless of pilot sign, only if technically valid

```powershell
python -m scripts.asoc_real_timegan --domains weather,traffic --dry-run
python -m scripts.asoc_real_timegan --domains weather --gpu-index 0
python -m scripts.asoc_real_timegan --domains traffic --gpu-index 0
python -m scripts.asoc_real_timegan --domains weather,traffic --report
```

After successful pilot, expect 11 pending forecasters (5 Weather + 6 Traffic), unless other valid rows already exist. Reports contain 6 pairs per domain, all against the original frozen references. Use one writer; do not run concurrent jobs or blindly merge CSVs across machines. Persisted pools are reused across model seeds, never regenerated when a completed row references a missing/changed pool. A pool's training and sampling times are copied into each run row: deduplicate by pool file SHA-256 for campaign-level generation cost, not model seed. Forecaster time includes validation and test; no end-to-end equal-compute claim. The report uses median within pool, then median across pools. Raw and normalized MSE/MAE are included; overlapping test origins are not independent samples. The published comparison must disclose the generator budget and possible quality limitations.

## Output files and provenance

`results/asoc_revision/phase14_timegan/`:
- `runs.csv`: one row per domain/rep/pool; generator stages, losses, times, hardware, reference and data digests, configuration ID, validation curve, raw and normalized test losses, selected checkpoint step, pool/origin digests.
- `pools/*.npz`: 1:1 synthetic windows, configuration metadata, data-content SHA-256, generator statistics. Cache survives a process restart. Preserve the exact files.
- `origins/*.npz`: integer `origin` = **first target index** (last observed index = origin-1), per-window standardized squared/absolute error averaged across horizon; means match run aggregate. These are loss traces, not full forecasts or model checkpoints.
- `report/{paired,per_pool,summary,incomplete}.csv`: paired against archived VAE/real-only/duplicated-real. Report refuses incomplete requested campaign; intermediate report files may exist.

The old 81-file release remains unchanged. NPZ pools/origin traces are excluded by the old CSV/JSON-only packager; retain a separate new artifact with hashes when preparing the expanded release. Do not claim Phase 14 closes the paired temporal confidence-interval request: the original Phase 7 comparators have no per-origin traces and their original VAE pools were not persisted. A separately declared reference replay could record new real-only/duplicate traces and regenerated-VAE traces; those would be new runs, not historical originals. Need paired same-origin loss differences before a temporal block-bootstrap analysis. No TimeGAN efficacy, fully matched end-to-end compute, conditional generation or generalization across independent test periods is established before real results.
