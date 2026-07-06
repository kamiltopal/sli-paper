"""Synthetic-data generators (protocol §3.2, §3.4).

Openness labels are PRE-REGISTERED here, per the mechanical checklist:
does the generator use any information source absent from the target
training window, structurally justifiable as future-relevant?

  bootstrap   O=0  (resamples the real training series itself)
  vae         O=0  (trained solely on real training windows)
  seasonal    O=1  (receives the domain-fixed structural form: for the
                    backbone, the true core spec — the oracle prior)
  exog        O=1  (conditions on exogenous covariate channels)

All generators consume the TRAINING segment only and return a synthetic
(inputs, targets) window set sized `ratio` x the real window count.
"""
from __future__ import annotations

import numpy as np

OPENNESS = {"bootstrap": 0, "vae": 0, "seasonal": 1, "exog": 1}


# ---------------------------------------------------------------- closed --
def gen_bootstrap(Xtr: np.ndarray, cfg, ratio: float = 1.0,
                  block: int = 200, seed: int = 0, **_):
    """Moving-block bootstrap of the training segment, then windowing."""
    from .data import make_windows
    rng = np.random.default_rng(seed)
    T, C = Xtr.shape
    n_blocks = T // block + 1
    starts = rng.integers(0, T - block, size=n_blocks)
    parts = [Xtr[s:s + block] for s in starts]
    Xb = np.concatenate(parts, axis=0)[:T]
    xi, yi = make_windows(Xb, 0, len(Xb), cfg)
    n = int(ratio * max(1, (T - cfg.lookback - cfg.horizon)))
    keep = rng.choice(len(xi), size=min(n, len(xi)), replace=False)
    return xi[keep], yi[keep]


def gen_vae(Xtr: np.ndarray, cfg, ratio: float = 1.0, seed: int = 0,
            epochs: int = 30, latent: int = 16, **_):
    """Small MLP-VAE over full (lookback+horizon) target-channel windows;
    aux channels are bootstrapped alongside (VAE models the target)."""
    import torch
    import torch.nn as nn
    from .data import make_windows

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L, H = cfg.lookback, cfg.horizon
    W = L + H
    xi, yi = make_windows(Xtr, 0, len(Xtr), cfg)
    full = np.concatenate([xi[:, :, 0], yi], axis=1).astype(np.float32)  # (N, W)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = nn.Sequential(nn.Linear(W, 128), nn.ReLU(), nn.Linear(128, 2 * latent)).to(dev)
    dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, W)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

    data = torch.as_tensor(full, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(data), device=dev)
        for i in range(0, len(data), 256):
            b = data[perm[i:i + 256]]
            h = enc(b)
            mu, logv = h[:, :latent], h[:, latent:]
            z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
            rec = dec(z)
            loss = ((rec - b) ** 2).mean() + 1e-3 * (
                -0.5 * (1 + logv - mu ** 2 - logv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step()

    n = int(ratio * len(full))
    with torch.no_grad():
        z = torch.randn(n, latent, device=dev)
        synth = dec(z).cpu().numpy()                      # (n, W) target ch.
    # aux channels for synthetic windows: bootstrap real aux windows
    keep = rng.choice(len(xi), size=n, replace=True)
    x_aux = xi[keep].copy()
    x_aux[:, :, 0] = synth[:, :L]
    return x_aux, synth[:, L:]


# ------------------------------------------------------------------ open --
def gen_seasonal(Xtr: np.ndarray, cfg, ratio: float = 1.0, seed: int = 0,
                 spec=None, train_start: int = 0, **_):
    """Structural-prior generator (backbone oracle form): regenerates the
    TRUE deterministic core from the series spec and adds fresh noise.
    Openness source: the domain-fixed structural form (core spec), which
    is not derivable from the training window alone."""
    from .signals import _build_core
    from .data import make_windows
    rng = np.random.default_rng(seed + 20_000)
    assert spec is not None, "seasonal generator needs the SignalSpec row"
    core = _build_core(int(spec["seed"]),
                       _spec_obj(spec), float(spec["concentration"]))
    seg = core[train_start:train_start + len(Xtr)]
    w = float(spec["mix_weight"])
    synth = w * seg + (1 - w) * rng.standard_normal(len(seg))
    synth = (synth - synth.mean()) / (synth.std() + 1e-8)
    Xs = Xtr.copy()
    Xs[:, 0] = synth
    xi, yi = make_windows(Xs, 0, len(Xs), cfg)
    n = int(ratio * len(xi))
    keep = rng.choice(len(xi), size=min(n, len(xi)), replace=False)
    return xi[keep], yi[keep]


def gen_exog(Xtr: np.ndarray, cfg, ratio: float = 1.0, seed: int = 0, **_):
    """Exogenous-conditioned generator: fits target ~ aux channels (ridge)
    on the training segment, then synthesizes target = prediction +
    block-bootstrapped residuals. Openness source: exogenous covariates."""
    from .data import make_windows
    rng = np.random.default_rng(seed + 30_000)
    y, A = Xtr[:, 0], Xtr[:, 1:]
    Ai = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    coef = np.linalg.lstsq(Ai.T @ Ai + 1e-3 * np.eye(Ai.shape[1]),
                           Ai.T @ y, rcond=None)[0]
    pred = Ai @ coef
    resid = y - pred
    block = 100
    n_blocks = len(y) // block + 1
    starts = rng.integers(0, len(y) - block, size=n_blocks)
    rb = np.concatenate([resid[s:s + block] for s in starts])[:len(y)]
    synth = pred + rb
    synth = (synth - synth.mean()) / (synth.std() + 1e-8)
    Xs = Xtr.copy()
    Xs[:, 0] = synth
    xi, yi = make_windows(Xs, 0, len(Xs), cfg)
    n = int(ratio * len(xi))
    keep = rng.choice(len(xi), size=min(n, len(xi)), replace=False)
    return xi[keep], yi[keep]


# --------------------------------------------------------------------------
def _spec_obj(row):
    """Rebuild a SignalSpec from a validation-CSV row (for core regen)."""
    from .signals import SignalSpec
    return SignalSpec(target_omega=float(row["target_omega"]),
                      seed=int(row["seed"]), variant=str(row["variant"]),
                      n_points=int(row["n_points"]),
                      n_components=int(row["n_components"]))


GENERATORS = {"bootstrap": gen_bootstrap, "vae": gen_vae,
              "seasonal": gen_seasonal, "exog": gen_exog}
