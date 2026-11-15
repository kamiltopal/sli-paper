"""Phase 10: fixed-step Weather 10% target-only compact DDPM pilot.

Reuses, without refitting, the validated Phase 7 VAE and three real controls.
This is an *unconditional, window-level MLP-DDPM*, NOT Diffusion-TS/TimeGAN,
and NOT evidence of a state-of-the-art diffusion benchmark.

  python -m scripts.asoc_real_diffusion_weather --dry-run
  python -m scripts.asoc_real_diffusion_weather --reps 0 --pool-seeds 0
  python -m scripts.asoc_real_diffusion_weather --report
  python -m scripts.asoc_real_diffusion_weather --reps 0,1,2 --pool-seeds 0,1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.asoc_budget_controls import (count_steps, fit_condition,
                                          only_channels, select_device)
from scripts.asoc_real_controls import parse_ints, sha256
from src.data import WindowConfig, chrono_split, apply_regime, prepare

ROOT = Path(__file__).resolve().parents[1]
VERSION = "phase10_weather_target_ddpm_v1"
REF_VERSION = "phase7_real_equal_step_v1"
REF = Path("results/asoc_revision/real_budget_controls.csv")
OUT = Path("results/asoc_revision/diffusion_weather_runs.csv")
REPORT = Path("results/asoc_revision/diffusion_weather_report_v1")
CACHE = Path("results/asoc_revision/diffusion_weather_pools_v1")
REQUIRED_REF = {"experiment_version", "domain", "data_sha256", "regime",
                "arch", "input_mode", "rep", "condition", "pool_seed",
                "epochs", "batch_size", "lr", "vae_epochs", "latent", "hidden",
                "horizon_weight", "lookback", "horizon", "optimizer_steps",
                "matched_steps", "original_steps", "test_mse", "test_mae",
                "mse_raw", "mae_raw", "target_sd", "train_start", "train_end",
                "real_windows"}
CONTROLS = ("epoch_baseline", "step_baseline", "duplicate_real")
KEY = ("experiment_version", "data_sha256", "rep", "pool_seed",
       "gen_epochs", "diffusion_steps", "denoiser_width", "denoiser_batch",
       "gen_lr", "sampling_batch", "epochs", "batch_size", "lr")


def root_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def fingerprint(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, allow_nan=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def check_weather_active_timestamps(*, path: Path, start: int) -> str:
    """Audit only actively used train/val/test sample timestamps, read-only."""
    if not path.is_file():
        return "raw_timestamp_csv_unavailable"
    dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce")
    if start >= len(dates) or dates.iloc[start:].isna().any():
        raise ValueError("Invalid timestamps in the active training/evaluation span")
    if dates.iloc[start:].duplicated().any() or not (
        dates.iloc[start:].diff().dropna() == pd.Timedelta(minutes=10)
    ).all():
        raise ValueError("Weather timestamp anomalies intersect active 10% train/val/test span")
    return "active_train_val_test_10min_regular"


def load_reference(path: Path, *, digest: str, reps: tuple[int, ...],
                   pools: tuple[int, ...], n_real: int, train_start: int,
                   train_end: int, original_steps: int,
                   matched_steps: int, epochs: int, batch_size: int,
                   lr: float, vae_epochs: int, latent: int, hidden: int,
                   horizon_weight: float) -> dict[tuple[int, str, int], pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 7 reference is missing: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_REF - set(df.columns)
    if missing:
        raise ValueError(f"Phase 7 reference missing columns: {sorted(missing)}")
    subset = df[(df.experiment_version == REF_VERSION) &
                (df.domain == "weather") &
                np.isclose(df.regime.astype(float), 0.10) &
                (df.arch == "itransformer") & (df.input_mode == "target") &
                (df.lookback == 96) & (df.horizon == 24) &
                (df.epochs == epochs) & (df.batch_size == batch_size) &
                np.isclose(df.lr.astype(float), lr) &
                (df.vae_epochs == vae_epochs) & (df.latent == latent) &
                (df.hidden == hidden) &
                np.isclose(df.horizon_weight.astype(float), horizon_weight)].copy()
    if subset.empty:
        raise ValueError("No matching Weather 10%/iTransformer/target Phase 7 reference")
    if not (subset.data_sha256 == digest).all():
        raise ValueError("Data hash mismatch in Phase 7 Weather reference; STOP")
    lookup: dict[tuple[int, str, int], pd.Series] = {}
    for rep in reps:
        for condition, pool_seed in ([ (c, -1) for c in CONTROLS ] +
                                      [("vae_augmented", ps) for ps in pools]):
            found = subset[(subset.rep == rep) & (subset.condition == condition) &
                           (subset.pool_seed == pool_seed)]
            if len(found) != 1:
                raise ValueError(f"Expected one Phase 7 row, got {len(found)}: "
                                 f"rep={rep}, condition={condition}, pool={pool_seed}")
            row = found.iloc[0]
            expected_steps = (original_steps if condition == "epoch_baseline"
                              else matched_steps)
            for k, expected in (("real_windows", n_real), ("train_start", train_start),
                                ("train_end", train_end),
                                ("original_steps", original_steps),
                                ("matched_steps", matched_steps),
                                ("optimizer_steps", expected_steps)):
                if int(row[k]) != expected:
                    raise ValueError(f"Phase 7 budget/partition mismatch: {k}, "
                                     f"rep={rep}, condition={condition}")
            for k in ("test_mse", "test_mae", "mse_raw", "mae_raw", "target_sd"):
                if not np.isfinite(float(row[k])) or float(row[k]) <= 0:
                    raise ValueError(f"Bad reference numeric value: {k}")
            if not np.isclose(float(row.mse_raw),
                              float(row.test_mse) * float(row.target_sd)**2,
                              rtol=1e-5):
                raise ValueError("Phase 7 original-scale MSE inconsistent")
            if not np.isclose(float(row.mae_raw),
                              float(row.test_mae) * float(row.target_sd),
                              rtol=1e-5):
                raise ValueError("Phase 7 original-scale MAE inconsistent")
            lookup[(rep, condition, pool_seed)] = row
        for ps in pools:
            a = lookup[(rep, "vae_augmented", ps)]
            for c in CONTROLS:
                r = lookup[(rep, c, -1)]
                if not np.isclose(a.target_sd, r.target_sd, rtol=0, atol=1e-8):
                    raise ValueError("Reference normalization mismatch")
    return lookup


class WindowDenoiser(nn.Module):
    """Unconditional MLP epsilon predictor for a complete target window."""
    def __init__(self, window: int, width: int = 128, time_dim: int = 32):
        super().__init__()
        if window < 2 or width < 8 or time_dim < 4 or time_dim % 2:
            raise ValueError("Invalid window/denoiser dimensions")
        self.time_dim = time_dim
        self.network = nn.Sequential(
            nn.Linear(window + time_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, window))

    def forward(self, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=noisy.device, dtype=noisy.dtype) /
                          max(1, half - 1))
        phase = t.to(dtype=noisy.dtype).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)
        return self.network(torch.cat((noisy, emb), dim=1))


def diffusion_schedule(steps: int, device: torch.device):
    if not 32 <= steps <= 512:
        raise ValueError("Diffusion steps must be between 32 and 512")
    # Rescaled schedule: an unscaled 0.02 end-beta at T=64 leaves ~52%
    # signal at the terminal step, invalidating N(0,1) ancestral starts.
    betas = torch.linspace(1e-4, min(0.25, 12.0 / steps), steps, device=device)
    alphas = 1.0 - betas
    cum = alphas.cumprod(0)
    previous = torch.cat((torch.ones_like(cum[:1]), cum[:-1]))
    posterior_variance = (betas * (1 - previous) / (1 - cum)).clamp(min=0)
    return betas, alphas, cum, posterior_variance


def spectral_fidelity(real_full: np.ndarray, synthetic_full: np.ndarray) -> float:
    """Training-only mean normalized power spectrum overlap [0,1]."""
    def mean_spectrum(w):
        centered = w - w.mean(axis=1, keepdims=True)
        power = (np.abs(np.fft.rfft(centered, axis=1)) ** 2).mean(axis=0)
        return power / max(float(power.sum()), 1e-12)
    return float(1.0 - 0.5 * np.abs(mean_spectrum(real_full) -
                                    mean_spectrum(synthetic_full)).sum())


def generate_pool(real: tuple[np.ndarray, np.ndarray], *, seed: int, epochs: int,
                  steps: int, width: int, batch_size: int, lr: float,
                  sample_batch: int, device: torch.device) -> tuple[tuple, dict]:
    """Fit DDPM on TRAIN-only (lookback+horizon) target windows and sample 1x."""
    if seed < 0 or min(epochs, batch_size, sample_batch) <= 0 or lr <= 0:
        raise ValueError("Invalid diffusion hyperparameters")
    real_x, real_y = real
    n, lookback = real_x.shape[0], real_x.shape[1]
    horizon = real_y.shape[1]
    if n < 5 or len(real_y) != n or real_x.shape[2] != 1:
        raise ValueError("Expected target-only train windows")
    full = np.concatenate((real_x[:, :, 0], real_y), axis=1).astype(np.float32)
    if not np.isfinite(full).all():
        raise ValueError("Nonfinite real windows")
    torch.manual_seed(seed + 830_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 830_000)
    rng = torch.Generator(device=device).manual_seed(seed + 830_000)
    data = torch.as_tensor(full, device=device)
    model = WindowDenoiser(lookback+horizon, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    betas, alphas, cum, posterior = diffusion_schedule(steps, device)
    t0 = time.perf_counter()
    losses = []
    updates = 0
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n, device=device, generator=rng)
        total_loss = 0.0
        weighted_n = 0
        for start in range(0, n, batch_size):
            x0 = data[permutation[start:start+batch_size]]
            t = torch.randint(0, steps, (len(x0),), device=device, generator=rng)
            eps = torch.randn(x0.shape, device=device, generator=rng)
            a = cum[t].sqrt().unsqueeze(1)
            b = (1.0-cum[t]).sqrt().unsqueeze(1)
            noisy = a*x0 + b*eps
            loss = (model(noisy, t) - eps).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Diffusion training produced nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().item() * len(x0)
            weighted_n += len(x0)
            updates += 1
        losses.append(total_loss/weighted_n)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = round(time.perf_counter()-t0, 3)
    model.eval()
    t0 = time.perf_counter()
    chunks = []
    with torch.no_grad():
        for start in range(0, n, sample_batch):
            b = min(sample_batch, n-start)
            x = torch.randn((b,lookback+horizon), device=device, generator=rng)
            for step in range(steps-1, -1, -1):
                t = torch.full((b,), step, dtype=torch.long, device=device)
                epsilon = model(x, t)
                x = (x - betas[step]/(1.0-cum[step]).sqrt()*epsilon) / alphas[step].sqrt()
                if step:
                    noise = torch.randn(x.shape, device=device, generator=rng)
                    x = x + posterior[step].sqrt()*noise
            chunks.append(x.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_seconds = round(time.perf_counter()-t0, 3)
    synth = np.concatenate(chunks).astype(np.float32)
    if synth.shape != (n,lookback+horizon) or not np.isfinite(synth).all():
        raise FloatingPointError("Diffusion produced invalid synthetic windows")
    result = (synth[:,:lookback,None], synth[:,lookback:])
    stats = dict(train_seconds=train_seconds, sample_seconds=sampling_seconds,
                 train_updates=updates, final_noise_mse=float(losses[-1]),
                 first_noise_mse=float(losses[0]),
                 train_window_mean=float(full.mean()),
                 synthetic_window_mean=float(synth.mean()),
                 train_window_std=float(full.std()),
                 synthetic_window_std=float(synth.std()),
                 spectral_fidelity=spectral_fidelity(full,synth),
                 device=str(device),
                 total_synthetic_windows=n)
    return result, stats


def pool_meta(*, data_sha256: str, pool_seed: int, gen_epochs: int,
              diffusion_steps: int, denoiser_width: int, denoiser_batch: int,
              gen_lr: float, sampling_batch: int, real_windows: int) -> dict:
    return dict(version=VERSION, domain="weather", regime=0.1,
                lookback=96, horizon=24, input_mode="target",
                data_sha256=data_sha256, pool_seed=pool_seed,
                gen_epochs=gen_epochs, diffusion_steps=diffusion_steps,
                denoiser_width=denoiser_width, denoiser_batch=denoiser_batch,
                gen_lr=gen_lr, sampling_batch=sampling_batch, real_windows=real_windows)


def cache_pool(cache_dir: Path, real: tuple, metadata: dict, *,
               sample_batch: int, device: torch.device) -> tuple[tuple, dict, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"weather_ddpm_pool{metadata['pool_seed']}_{fingerprint(metadata)}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            saved = json.loads(str(stored["metadata"].item()))
            if saved != metadata:
                raise ValueError("Cached diffusion pool provenance mismatch")
            x, y = stored["x"].copy(), stored["y"].copy()
            stats = json.loads(str(stored["stats"].item()))
        if x.shape != real[0].shape or y.shape != real[1].shape or not (
            np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Cached pool dimensions or finiteness mismatch")
        print(f"REUSE diffusion pool: {path}", flush=True)
        return (x,y), stats, path
    print(f"GENERATE diffusion pool seed={metadata['pool_seed']} ", flush=True)
    generated, stats = generate_pool(
        real, seed=metadata["pool_seed"], epochs=metadata["gen_epochs"],
        steps=metadata["diffusion_steps"], width=metadata["denoiser_width"],
        batch_size=metadata["denoiser_batch"], lr=metadata["gen_lr"],
        sample_batch=sample_batch, device=device)
    temp = path.with_suffix(".npz.partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, x=generated[0], y=generated[1],
                            metadata=json.dumps(metadata, sort_keys=True),
                            stats=json.dumps(stats, sort_keys=True))
    os.replace(temp, path)
    return generated, stats, path


def run_key(record: dict) -> tuple:
    return tuple(record[k] for k in KEY)


def existing_runs(path: Path) -> tuple[set, pd.DataFrame]:
    if not path.exists():
        return set(), pd.DataFrame(columns=KEY)
    frame = pd.read_csv(path)
    absent = set(KEY) - set(frame.columns)
    if absent:
        raise ValueError(f"Existing output lacks columns: {sorted(absent)}")
    if frame.duplicated(list(KEY)).any():
        raise ValueError("Duplicated diffusion run identity; audit output")
    return set(frame[list(KEY)].itertuples(index=False, name=None)), frame


def paired_report(runs: pd.DataFrame, references: dict, expected: list[dict]) -> tuple:
    """Only complete, correctly matched seed/pool observations enter summaries."""
    items = []
    for row in runs.to_dict("records"):
        key = run_key(row)
        if key not in {run_key(x) for x in expected}:
            continue
        rep, pool = int(row["rep"]), int(row["pool_seed"])
        refs = {c: references[(rep,c,-1)] for c in CONTROLS}
        refs["vae_augmented"] = references[(rep,"vae_augmented",pool)]
        mse = float(row["test_mse"])
        mae = float(row["test_mae"])
        if int(row["optimizer_steps"]) != int(refs["vae_augmented"].matched_steps) or any(
                int(refs[c].optimizer_steps) != int(row["optimizer_steps"])
                for c in ("step_baseline", "duplicate_real", "vae_augmented")):
            raise ValueError("Diffusion/reference optimizer budgets mismatch")
        if not np.isclose(float(row["target_sd"]),
                          float(refs["vae_augmented"].target_sd), atol=1e-8, rtol=0):
            raise ValueError("Diffusion/reference target scaler mismatch")
        if not np.isclose(float(row["mse_raw"]), mse*float(row["target_sd"])**2,rtol=1e-5):
            raise ValueError("Diffusion raw MSE scaling mismatch")
        if not np.isclose(float(row["mae_raw"]), mae*float(row["target_sd"]),rtol=1e-5):
            raise ValueError("Diffusion raw MAE scaling mismatch")
        entry = dict(rep=rep,pool_seed=pool,domain="weather",regime=0.1,
                     lookback=96,horizon=24,matched_steps=int(row["optimizer_steps"]),
                     diffusion_test_mse=mse,diffusion_test_mae=mae,
                     diffusion_mse_raw=float(row["mse_raw"]),
                     diffusion_mae_raw=float(row["mae_raw"]),
                     diffusion_best_at_last=(int(row["best_step"])==int(row["optimizer_steps"])))
        for name, reference in refs.items():
            label = "vae" if name == "vae_augmented" else name
            for metric, observed in (("test_mse",mse),("test_mae",mae),
                                     ("mse_raw",float(row["mse_raw"])),
                                     ("mae_raw",float(row["mae_raw"]))):
                baseline = float(reference[metric])
                entry[f"{label}_{metric}"] = baseline
                entry[f"diffusion_vs_{label}_{metric}_pct"] = 100*(observed-baseline)/baseline
                entry[f"diffusion_vs_{label}_{metric}_abs"] = observed-baseline
        entry["diffusion_beats_both_matched"] = (entry["diffusion_vs_step_baseline_test_mse_pct"]<0 and
                                                  entry["diffusion_vs_duplicate_real_test_mse_pct"]<0)
        entry["diffusion_beats_vae"] = entry["diffusion_vs_vae_test_mse_pct"] < 0
        items.append(entry)
    paired = pd.DataFrame(items)
    if paired.empty:
        return paired, pd.DataFrame(), pd.DataFrame()
    effects = [k for k in paired.columns if k.startswith("diffusion_vs_")]
    pool_rows = []
    for ps, group in paired.groupby("pool_seed",sort=True):
        r = {"pool_seed":int(ps),"n_model_seeds":int(group.rep.nunique()),
             "n_pairs":len(group),
             "fraction_beats_both":float(group.diffusion_beats_both_matched.mean()),
             "fraction_beats_vae":float(group.diffusion_beats_vae.mean())}
        for e in effects: r["median_"+e] = float(group[e].median())
        pool_rows.append(r)
    per_pool = pd.DataFrame(pool_rows)
    summary = {"n_pool_seeds":len(per_pool),
               "n_pairs":len(paired),
               "n_model_seeds_min":int(per_pool.n_model_seeds.min()),
               "n_model_seeds_max":int(per_pool.n_model_seeds.max()),
               "fraction_pools_beats_both":float((per_pool.fraction_beats_both>0.5).mean()),
               "fraction_pools_beats_vae":float((per_pool.fraction_beats_vae>0.5).mean())}
    for e in effects:
        summary["median_"+e] = float(per_pool["median_"+e].median())
        summary["min_pool_"+e] = float(per_pool["median_"+e].min())
        summary["max_pool_"+e] = float(per_pool["median_"+e].max())
    return paired, per_pool, pd.DataFrame([summary])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", default="0")
    parser.add_argument("--pool-seeds", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--horizon-weight", type=float, default=0.5)
    parser.add_argument("--gen-epochs", type=int, default=30)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--denoiser-width", type=int, default=128)
    parser.add_argument("--denoiser-batch", type=int, default=256)
    parser.add_argument("--gen-lr", type=float, default=1e-3)
    parser.add_argument("--sampling-batch", type=int, default=128)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data/real/weather.npy"))
    parser.add_argument("--raw-timestamps", type=Path, default=Path("data/real/weather/weather.csv"))
    parser.add_argument("--reference", type=Path, default=REF)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, default=0,help="Max new forecaster trainings; 0=all")
    args = parser.parse_args()
    try:
        reps = parse_ints(args.reps)
        pools = parse_ints(args.pool_seeds)
        if (min(args.epochs,args.batch_size,args.vae_epochs,args.latent,args.hidden,
                args.gen_epochs,args.denoiser_width,args.denoiser_batch,args.sampling_batch)<1 or
                args.lr <= 0 or args.gen_lr <= 0 or args.limit<0 or args.gpu_index<0 or
                not 0<args.horizon_weight<1):
            raise ValueError("Invalid hyperparameters or limit")
        if args.denoiser_width<8:
            raise ValueError("denoiser-width >= 8 required")
        if args.dry_run and args.report:
            raise ValueError("Choose --dry-run or --report")
        diffusion_schedule(args.diffusion_steps,torch.device("cpu"))
        data_path = root_path(args.data)
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing {data_path}")
        digest = sha256(data_path)
        X = np.load(data_path).astype(np.float32)
        if X.ndim!=2 or X.shape[1]<1 or len(X)<1000 or not np.isfinite(X).all():
            raise ValueError("Invalid weather.npy")
        cfg = WindowConfig(lookback=96,horizon=24)
        prepared = prepare(X,regime=.10,cfg=cfg)
        real = only_channels(prepared["train"],"target")
        val = only_channels(prepared["val"],"target")
        test = only_channels(prepared["test"],"target")
        n = len(real[0]); t1,_=chrono_split(len(X))
        tstart=apply_regime(t1,.10,cfg.lookback)
        initial_steps,matched_steps=count_steps(n,n,args.batch_size,args.epochs)
        target_sd=float(prepared["sd"][0,0]);target_mean=float(prepared["mu"][0,0])
        active_audit=check_weather_active_timestamps(path=root_path(args.raw_timestamps),start=tstart)
        refs=load_reference(root_path(args.reference),digest=digest,reps=reps,pools=pools,
                            n_real=n,train_start=tstart,train_end=t1,
                            original_steps=initial_steps,matched_steps=matched_steps,
                            epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,
                            vae_epochs=args.vae_epochs,latent=args.latent,hidden=args.hidden,
                            horizon_weight=args.horizon_weight)
        common=dict(experiment_version=VERSION,data_sha256=digest,
                    gen_epochs=args.gen_epochs,diffusion_steps=args.diffusion_steps,
                    denoiser_width=args.denoiser_width,
                    denoiser_batch=args.denoiser_batch,gen_lr=args.gen_lr,
                    sampling_batch=args.sampling_batch,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr)
        expected=[{**common,"rep":rep,"pool_seed":pool} for pool in pools for rep in reps]
        out=root_path(args.out)
        done,previous=existing_runs(out)
        if len(previous):
            if (previous.data_sha256 != digest).any():
                raise ValueError("Existing diffusion results have a different data hash")
            if not (previous.experiment_version==VERSION).all():
                raise ValueError("Mixed output protocol versions")
            for entry in previous.to_dict("records"):
                if (int(entry["rep"]) in reps and int(entry["pool_seed"]) in pools and
                    any(entry[k] != common[k] for k in common)):
                    raise ValueError("Existing run hyperparameters changed; use a NEW --out")
        missing=[row for row in expected if run_key(row) not in done]
        print(f"Phase 10 | Weather 10% | L96 H24 | real_windows={n} | "
              f"forecaster_steps={matched_steps} | active_cadence={active_audit}",flush=True)
        print(f"PLAN: {len(expected)} desired diffusion rows; "
              f"{len(missing)} pending; Phase7 reference READ-ONLY; out={out}",flush=True)
        for row in missing:
            print(f"  PENDING diffusion model seed={row['rep']} pool seed={row['pool_seed']}",flush=True)
        if args.dry_run:
            return
        if args.report:
            paired, per_pool, summary=paired_report(previous,refs,expected)
            if paired.empty:
                raise ValueError("No matching diffusion results to report")
            report_dir=root_path(args.report_dir);report_dir.mkdir(parents=True,exist_ok=True)
            paired.to_csv(report_dir/"diffusion_paired.csv",index=False)
            per_pool.to_csv(report_dir/"diffusion_per_pool.csv",index=False)
            summary.to_csv(report_dir/"diffusion_summary.csv",index=False)
            print("Saved paired/per-pool/summary reports:",report_dir)
            print(summary[["n_pairs","n_pool_seeds","median_diffusion_vs_vae_test_mse_pct",
                           "median_diffusion_vs_duplicate_real_test_mse_pct",
                           "median_diffusion_vs_step_baseline_test_mse_pct"]].to_string(index=False))
            print("Exploratory descriptive pool medians only; not a CI or significance test")
            return
        if not missing:
            print("No pending diffusion forecaster runs")
            return
        selected=missing[:args.limit] if args.limit else missing
        device=select_device(allow_cpu=args.allow_cpu,gpu_index=args.gpu_index)
        print("Training device:",device,flush=True)
        out.parent.mkdir(parents=True,exist_ok=True)
        for ps in pools:
            pending_for_pool=[row for row in selected if row["pool_seed"]==ps]
            if not pending_for_pool:
                continue
            meta=pool_meta(data_sha256=digest,pool_seed=ps,
                           gen_epochs=args.gen_epochs,diffusion_steps=args.diffusion_steps,
                           denoiser_width=args.denoiser_width,
                           denoiser_batch=args.denoiser_batch,gen_lr=args.gen_lr,
                           sampling_batch=args.sampling_batch,real_windows=n)
            synthetic,stats,pool_file=cache_pool(root_path(args.cache_dir),real,meta,
                                                  sample_batch=args.sampling_batch,
                                                  device=device)
            train_pair=(np.concatenate((real[0],synthetic[0]),axis=0),
                        np.concatenate((real[1],synthetic[1]),axis=0))
            interval=math.ceil(2*n/args.batch_size)
            for planned in pending_for_pool:
                rep=planned["rep"]
                if device.type=="cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                result=fit_condition(arch="itransformer",train_pair=train_pair,
                                     val_pair=val,test_pair=test,init_seed=rep,
                                     max_steps=matched_steps,validation_interval=interval,
                                     batch_size=args.batch_size,lr=args.lr,device=device)
                if int(result["optimizer_steps"]) != matched_steps:
                    raise ValueError("Forecaster did not use required matched steps")
                row=dict(**planned,**result,domain="weather",regime=.10,arch="itransformer",
                     input_mode="target",lookback=96,horizon=24,
                     condition="diffusion_augmented",real_windows=n,
                     train_start=tstart,train_end=t1,
                     val_windows=len(val[0]),test_windows=len(test[0]),
                     target_mean=target_mean,target_sd=target_sd,
                     mse_raw=result["test_mse"]*target_sd**2,
                     mae_raw=result["test_mae"]*target_sd,
                     original_steps=initial_steps,matched_steps=matched_steps,
                     generator_name="unconditional_mlp_ddpm_full_target_windows",
                     generator_train_seconds=stats["train_seconds"],
                     generator_sample_seconds=stats["sample_seconds"],
                     generator_train_updates=stats["train_updates"],
                     generator_final_noise_mse=stats["final_noise_mse"],
                     generator_spectral_fidelity=stats["spectral_fidelity"],
                     generator_train_window_mean=stats["train_window_mean"],
                     generator_synthetic_window_mean=stats["synthetic_window_mean"],
                     generator_train_window_std=stats["train_window_std"],
                     generator_synthetic_window_std=stats["synthetic_window_std"],
                     pool_cache_path=str(pool_file),
                     pool_file_sha256=sha256(pool_file),
                     active_timestamps_audit=active_audit,
                     device=str(device),pytorch_version=str(torch.__version__),
                     gpu_peak_memory_mb=(round(torch.cuda.max_memory_allocated(device)/(1024**2),2)
                                         if device.type=="cuda" else 0.0))
                pd.DataFrame([row]).to_csv(out,mode="a",index=False,header=not out.exists())
                done.add(run_key(row))
                print(f"  DONE rep={rep} pool={ps} steps={matched_steps} "
                      f"MSE={row['test_mse']:.6f} MAE={row['test_mae']:.6f}",flush=True)
        print("Completed new diffusion rows. Run with --report to summarize.",flush=True)
    except (ValueError,FileNotFoundError,FloatingPointError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
