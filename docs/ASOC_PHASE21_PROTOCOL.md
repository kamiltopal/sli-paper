# ASOC Phase 21 — replication and validation sensitivity

Frozen follow-up design (post-review and post-Phase 20; not a prospective preregistration of the original study). Base code: `06dbf89d47247838d5ab5e1aeb87641bd2a72d52`. No earlier results are overwritten. One writer only; do not edit code or change device/environment while resuming. Do not select new seeds, pools or hyperparameters after the pilot.

## Design

- Weather and ETTh1 (`electricity` in code), 10% recent training slice, chronological splits and 96/24 windows exactly as Phase 20. Target-only iTransformer-lite. Training segment scaler, Adam 0.001, batch 64, 1:1 augmentation, 1120/350 optimizer steps respectively. Validation only selects checkpoints.
- Replication: centered forecaster, 20 model seeds (0–19); 5 newly fitted VAE pools per domain (0–4), 30 epochs, latent 16, original MLP generator, target windows. Step-only and duplicate-real controls once per seed (not duplicated for each pool). 140 fits/domain, 280 total. Pools are new generation runs, not claims of new statistically independent domains or byte-identical historical pools. On same hardware and seed a pool may be identical to an older one.
- Interaction: Weather only, seeds 0–9, original/centered representation x VAE/TimeGAN x two preserved historical pools each. 80 fits. Uses exactly the Phase 15 replay VAE and Phase 14 TimeGAN pool hashes from Phase 20; these are not the five new replication pools. No new TimeGAN generation. 360 total fits, 720 policy-level records.
- Nested checkpoint grids: Weather 28/112, ETTh1 5/35. One training trajectory per fit; evaluate at fine grid, choose earliest minimum validation MSE on fine and coarse subsets, evaluate both selected checkpoints after training. Save validation curve and per-origin MSE/MAE for both. Restore Python/NumPy/PyTorch CPU/CUDA RNG after evaluation to prevent the extra evaluations changing minibatch/dropout randomness. This differs from historical Phase 20's evaluation RNG behavior: not an exact replay, even at the coarse grid. No test outcome controls training or continuation.
- Primary descriptive replication effect: coarse hierarchical median (median over model seeds within each pool, then median over five pools); report fine sensitivity and both MSE/MAE. Report paired wins over both controls, absolute adequacy vs persistence/ridge, and real-control procedural variability (not a statistical noise floor). Interaction: matched same-pool-seed TimeGAN/VAE relative effect, original vs centered, under both policies. Two pools and one test period do not establish population inference. No formal iid inference across crossed seed/pool pairs. All results, positive or negative, retained.

## Commands (PowerShell, repository root)

```powershell
conda activate sli_2
git apply --check .\ASOC_Phase21_Replication_Validation.patch
git apply .\ASOC_Phase21_Replication_Validation.patch
python -m unittest discover -s tests -p 'test_asoc_phase21_replication.py' -v
python -m scripts.asoc_phase21_replication --dry-run
python -m scripts.asoc_phase21_replication --smoke --gpu-index 0
python -m scripts.asoc_phase21_replication --gpu-index 0 --limit 10
python -m scripts.asoc_phase21_replication --dry-run
# after technical pilot passes, finish the fixed grid regardless of signs
python -m scripts.asoc_phase21_replication --gpu-index 0
python -m scripts.asoc_phase21_replication --report
Compress-Archive -Path .\results\asoc_revision\phase21_replication -DestinationPath .\ASOC_Phase21_Complete_Results.zip -Force
```

Dry-run verifies the original prepared arrays and raw targets/cadence, Phase 7 references and all four preserved Weather pools via Phase 20 preflight. Expected 0/360 initially, 10/350 after pilot. GPU smoke is tiny synthetic data, no research output. Pilot runs 6 replication and 4 interaction fits at model seed 0; all ten new VAE pools are created before the first fit. Full run resumes complete job JSON records, checks source/protocol, environment, hash and metrics; no historical data writes. An interrupted fit may be retrained; completed fits are not repeated. CPU option requires CUDA hidden (`$env:CUDA_VISIBLE_DEVICES = '-1'` before launch, then `--cpu`); never mix GPU/CPU within the same result directory. CPU/GPU numerical results need not be identical. Validation work may dominate runtime, especially ETTh1; no timing guarantee.

Outputs: `protocol.json`, `environment.json`, `records/*.json`, `origins/*.npz`, `pools/*.npz`, `runs.csv` (two rows per fit), `report/paired.csv`, `per_pool.csv`, `summary.csv`, `dual_wins.csv`, `absolute.csv`, `checkpoint_sensitivity.csv`, `representation_interaction.csv`. Partial reports are explicitly incomplete. Historical pools are referenced by paths/hashes and remain in previous private archive; do not publish raw data/pools without permission. Unit tests use tiny dependency doubles for trajectory/report logic; actual integration and GPU smoke are separate gates.
