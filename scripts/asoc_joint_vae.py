"""Exploratory, *joint-window* VAE for ASOC Phase 3B.

Generate (X_lookback_all_channels, y_future_target) together. This is a
joint, unconditional window model, NOT a causal conditional generator and NOT
a generator of a coherent multivariate time-series trajectory. The model
sees only training windows; it never sees validation/test data.

Reconstruction objective is 0.5 * lookback MSE + 0.5 * target-horizon MSE.
This choice prevents the much smaller horizon tensor from being ignored by
an unweighted flat reconstruction loss. It was fixed before pilot outcomes.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scripts.asoc_budget_controls import seed_everything


class JointWindowVAE(nn.Module):
    def __init__(self, lookback: int, channels: int, horizon: int,
                 *, latent: int = 16, hidden: int = 128):
        super().__init__()
        if min(lookback, channels, horizon, latent, hidden) <= 0:
            raise ValueError("VAE dimensions must be positive")
        self.lookback, self.channels, self.horizon = lookback, channels, horizon
        self.latent = latent
        self.x_size = lookback * channels
        input_size = self.x_size + horizon
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, input_size))

    def forward(self, flat: torch.Tensor):
        params = self.encoder(flat)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: torch.Tensor):
        flat = self.decoder(z)
        return (flat[:, :self.x_size].reshape(-1, self.lookback, self.channels),
                flat[:, self.x_size:])


def joint_reconstruction_loss(reconstruction: torch.Tensor, actual: torch.Tensor,
                              *, lookback_size: int, horizon_weight: float = 0.5):
    if not 0 < horizon_weight < 1:
        raise ValueError("horizon_weight must be between 0 and 1")
    x_loss = (reconstruction[:, :lookback_size] - actual[:, :lookback_size]).square().mean()
    y_loss = (reconstruction[:, lookback_size:] - actual[:, lookback_size:]).square().mean()
    return (1 - horizon_weight) * x_loss + horizon_weight * y_loss


def gen_joint_vae(train_pair, *, seed: int, epochs: int, device: torch.device,
                  latent: int = 16, hidden: int = 128, batch_size: int = 256,
                  lr: float = 1e-3, kl_weight: float = 1e-3,
                  horizon_weight: float = 0.5):
    """Produce exactly n_real windows, with all channels generated jointly.

    Returns (x_syn, y_syn), metrics. Resets RNG, deterministic on supported
    hardware to the extent PyTorch CUDA kernels permit; seeds are recorded.
    """
    x, y = train_pair
    if x.ndim != 3 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[2] < 2:
        raise ValueError("Expected paired (N,L,C>=2) and (N,H) training arrays")
    if len(x) < 2 or min(epochs, latent, hidden, batch_size) < 1 or lr <= 0 or kl_weight < 0:
        raise ValueError("Invalid VAE training settings or too few windows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite training windows")

    # Must seed BEFORE creating encoder/decoder. Forecaster seeding is independent.
    seed_everything(seed)
    n, lookback, channels = x.shape
    horizon = y.shape[1]
    net = JointWindowVAE(lookback, channels, horizon, latent=latent, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    actual = torch.as_tensor(np.concatenate((x.reshape(n, -1), y), axis=1),
                             dtype=torch.float32, device=device)
    final_loss = None
    net.train()
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            batch = actual[order[start:start + batch_size]]
            rec, mu, logvar = net(batch)
            reconstruction = joint_reconstruction_loss(
                rec, batch, lookback_size=net.x_size, horizon_weight=horizon_weight)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction + kl_weight * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().item())
    net.eval()
    with torch.no_grad():
        z = torch.randn(n, latent, device=device)
        sx, sy = net.decode(z)
        sx = sx.cpu().numpy().astype(np.float32)
        sy = sy.cpu().numpy().astype(np.float32)
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise RuntimeError("Joint VAE produced nonfinite values")
    return (sx, sy), {"joint_last_batch_loss": final_loss,
                      "joint_generator_parameters": sum(p.numel() for p in net.parameters()),
                      "joint_generator_training_windows": n}


def shuffle_synthetic_aux(pair, *, seed: int):
    """Break only within-window target/aux pairing in the synthetic half.

    Keeps each auxiliary pair together, and preserves target lookback and
    target forecast labels *exactly*. The forecaster receives identical
    augmentation counts in the intact and shuffled conditions.
    """
    x, y = pair
    if x.ndim != 3 or x.shape[2] < 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Need at least two multivariate synthetic windows")
    order = np.random.default_rng(seed + 871_973).permutation(len(x))
    if np.array_equal(order, np.arange(len(x))):
        order = np.roll(order, 1)
    x_shuffled = x.copy()
    x_shuffled[:, :, 1:] = x[order, :, 1:]
    return x_shuffled, y.copy()


def _correlations(x, y=None, max_lag: int = 12):
    """Training-only associations. Lags use only historical lookback samples.

    For lag +k, correlate target[t] with auxiliary[t-k]. For lag -k,
    correlate target[t] with auxiliary[t+k]. The forecast diagnostic uses
    aux[-1] versus target future at h=1, floor(H/2), H.
    """
    if x.ndim != 3 or x.shape[2] < 2 or not 0 <= max_lag < x.shape[1]:
        raise ValueError("Invalid windows or lag specification")
    if y is not None and (y.ndim != 2 or len(y) != len(x)):
        raise ValueError("Forecast targets must align with windows")

    def safe_corr(a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if min(a.std(), b.std()) < 1e-10:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    l = x.shape[1]
    lag_corr = []
    forecast_corr = []
    for aux in range(1, x.shape[2]):
        row = []
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                t = x[:, lag:, 0]
                a = x[:, :l-lag, aux]
            elif lag < 0:
                t = x[:, :l+lag, 0]
                a = x[:, -lag:, aux]
            else:
                t = x[:, :, 0]
                a = x[:, :, aux]
            row.append(safe_corr(t, a))
        lag_corr.append(row)
        if y is not None:
            hids = sorted({0, (y.shape[1] - 1)//2, y.shape[1] - 1})
            forecast_corr.append([safe_corr(x[:, -1, aux], y[:, h]) for h in hids])
    return np.asarray(lag_corr, dtype=np.float64), np.asarray(forecast_corr, dtype=np.float64)


def cross_channel_diagnostics(real_pair, synthetic_pair, *, max_lag: int = 12):
    rx, ry = real_pair
    sx, sy = synthetic_pair
    if rx.shape[1:] != sx.shape[1:] or ry.shape[1] != sy.shape[1]:
        raise ValueError("Real and synthetic shapes do not agree")
    real_lag, real_future = _correlations(rx, ry, max_lag)
    syn_lag, syn_future = _correlations(sx, sy, max_lag)
    def gap(a, b):
        d = np.abs(a - b)
        return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
    return {"lag_corr_gap": gap(real_lag, syn_lag),
            "future_assoc_gap": gap(real_future, syn_future),
            "zero_lag_corr_gap": gap(real_lag[:, max_lag], syn_lag[:, max_lag])}
