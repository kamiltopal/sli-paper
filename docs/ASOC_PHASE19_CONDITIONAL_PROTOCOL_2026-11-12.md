# ASOC Phase 19 — EPİAŞ conditional VAE and matched-information pilot

**Scope locked before outcomes:** EPİAŞ, last 10% of original 70% train partition, target PTF, same prepared data SHA and chronological split as Phase 7, 96 history/24 horizon, 1:1 augmentation, 2 forecaster seeds (0,1), 1 generator seed (0). No Phase 1–18 files are modified. This is a targeted architectural ablation, not a benchmark of canonical PatchTST versus canonical iTransformer or a demonstration of causal superiority across datasets.

## Controls

- `target_only`: common three-token encoder/readout with auxiliary histories set to zero and diagonal attention mask. This uses target history only at forecast time; conditioning may affect the synthetic target distribution in its cVAE arm.
- `independent_fusion`: same three-token model; diagonal attention mask prohibits cross-channel attention in every encoder layer, while the final readout consumes all three encoded channel histories. Auxiliary input can affect the output, but not the target token in the encoder.
- `channel_mixing`: same embeddings, encoder, readout, dimensions, parameter count and initialization, with unrestricted attention. The matched-information comparison is the last two modes. These are controlled custom variants, not re-implementations of named published architectures.
- Each mode has `step_baseline`, `duplicate_real`, `unconditional_vae`, `conditional_vae` for each seed. All 24 have identical optimizer update counts `10*ceil(2*n_real/64)` and validation interval `ceil(2*n_real/64)`, Adam lr 0.001, batch size 64, validation-only best checkpoint, no early test-based stopping. Same seed resets initialization and loader RNG. Final test MSE/MAE, normalized and original units, and every origin loss retained. Same architecture and generator settings across seeds; final evaluation data identical.
- Generator: MLP VAE on 120 target values; encoder and decoder also take a 192-value flattened auxiliary history (2×96). Conditional branch uses the actual **training** calendar histories; unconditional branch uses zeros in both encoder and decoder. Both use same nominal parameter count, latent 16, hidden 128, Adam lr .001, 30 epochs, batch 256, beta .001, same seed and number of updates. Both draw the same deterministic permutation of all training auxiliary histories, once each; unconditional target is sampled without seeing the history, conditional target is sampled given that history. No future auxiliary input, validation or test data enters the generator. The two original pools are persisted and used unchanged by all three forecasters. The unconditional comparator is a new controlled VAE, not the original Phase 7 MLP VAE. No claims of matched total wall-clock compute.
- The model receives 3 channels including calendar history; test uses only observed past inputs. Calendar variables are from the original EPİAŞ raw reconstruction, not a learned covariate selection. A failing raw/prepared chronology or data hash gate stops the experiment; do not repair or overwrite source arrays.

## Execution from repo root (Windows PowerShell)

```powershell
conda activate sli_2
git switch revision/asoc-major-2026
git apply --check .\ASOC_Phase19_Conditional_Matched_Information_2026-11-12.patch
git apply .\ASOC_Phase19_Conditional_Matched_Information_2026-11-12.patch
python -m unittest discover -s tests -p 'test_asoc_phase19_conditional.py' -v
python -m scripts.asoc_phase19_conditional --dry-run
python -m scripts.asoc_phase19_conditional --smoke --gpu-index 0
python -m scripts.asoc_phase19_conditional --limit 1 --gpu-index 0
```

Dry-run uses the existing Phase 11 EPİAŞ preflight and Phase 7 reference; no training or output writes. Expect 24 pending fits on a fresh output folder. Actual window and step counts are read from verified local data. Smoke uses tiny synthetic arrays; never report smoke values as research results. The first real fit is target-only / step baseline / rep 0. After technical checks, continue **regardless of performance sign**:

```powershell
python -m scripts.asoc_phase19_conditional --gpu-index 0
python -m scripts.asoc_phase19_conditional --report
Compress-Archive -Path '.\results\asoc_revision\phase19_conditional\*' -DestinationPath '..\ASOC_Phase19_Complete_Results.zip' -Force
```

All outputs are in `results/asoc_revision/phase19_conditional/`: protocol/environment JSON, runs.csv, two `pools/*.npz` with sampled training indices, 24 `origins/*.npz`, report/paired.csv, summary.csv, interaction.csv and architecture.csv. Resume uses config ID and source hashes, checks previous origins and pool contents, rejects mixed environments, and skips completed keys. Run on one machine/process at a time; do not concurrently append or edit CSVs. New configuration or environment requires a new `--outdir`. GPU is required by default; `--allow-cpu` is useful for smoke, not historical GPU comparability. CPU smoke and 8 unit tests passed in the preparation environment; actual EPİAŞ data and CUDA not run here.

## Reporting

Report per-seed values, absolute MSE/MAE, augmentation-versus-both-controls, cVAE-versus-unconditional, and difference-in-differences (mixing augmentation effect minus independent-fusion augmentation effect). Also report target-only versus independent-fusion (information access) and independent-fusion versus channel-mixing (architecture conditional on same history). Do not treat two model seeds or overlapping origins as independent domain replicates. This is one domain and one pool seed, and does not establish a universal architecture/generator ranking, optimal conditional generator, or a new validated H4 selection policy. No post-hoc tuning based on the pilot. No expansion unless a separate scientific decision justifies it.
