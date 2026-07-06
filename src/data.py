"""Windowing, chronological splits, and data regimes (protocol §3.2).

Torch-free on purpose: numpy in/out, so it is unit-testable anywhere.
Convention: series array X has shape (T, C); column 0 is the target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowConfig:
    lookback: int = 96
    horizon: int = 24


def chrono_split(T: int, train=0.7, val=0.1):
    """Chronological index boundaries: [0,t1) train, [t1,t2) val, [t2,T) test."""
    t1 = int(T * train)
    t2 = int(T * (train + val))
    return t1, t2


def apply_regime(train_end: int, regime: float, lookback: int) -> int:
    """Data regime = use only the LAST `regime` fraction of the training
    span (keeps the boundary with val/test intact, mimics 'less history').
    Returns the new training start index.
    """
    span = train_end
    start = int(train_end - regime * span)
    return max(0, min(start, train_end - lookback - 1))


def standardize(X: np.ndarray, stats_slice: slice):
    """Standardize all channels using mean/std computed on stats_slice
    (must be the *training* portion only — no leakage)."""
    mu = X[stats_slice].mean(axis=0, keepdims=True)
    sd = X[stats_slice].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def make_windows(X: np.ndarray, lo: int, hi: int, cfg: WindowConfig):
    """Sliding windows fully contained in [lo, hi).

    Returns (inputs, targets):
      inputs  (N, lookback, C)
      targets (N, horizon)      -- target channel (col 0) only
    """
    L, H = cfg.lookback, cfg.horizon
    starts = np.arange(lo, hi - L - H + 1)
    if len(starts) <= 0:
        raise ValueError(f"segment [{lo},{hi}) too short for L={L}, H={H}")
    idx_in = starts[:, None] + np.arange(L)[None, :]
    idx_out = starts[:, None] + L + np.arange(H)[None, :]
    return X[idx_in], X[idx_out, 0]


def prepare(X: np.ndarray, regime: float, cfg: WindowConfig):
    """Full pipeline: split -> regime -> standardize (train stats) -> windows.

    Returns dict with train/val/test (inputs, targets) and the stats.
    """
    T = len(X)
    t1, t2 = chrono_split(T)
    tr_start = apply_regime(t1, regime, cfg.lookback)
    Xs, mu, sd = standardize(X, slice(tr_start, t1))
    out = {
        "train": make_windows(Xs, tr_start, t1, cfg),
        "val": make_windows(Xs, t1 - cfg.lookback, t2, cfg),
        "test": make_windows(Xs, t2 - cfg.lookback, T, cfg),
        "mu": mu, "sd": sd, "train_start": tr_start,
    }
    return out
