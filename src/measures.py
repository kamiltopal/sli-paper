"""Predictability / complexity measures used by the SLI.

Two independent ordinal proxies (protocol §3.4):
  - H_spec : spectral predictability  Omega = 1 - normalized spectral entropy
  - H_perm : permutation predictability = 1 - normalized permutation entropy

Both map to [0, 1]; higher = more predictable. They are used for
cross-cell *ordering*, never as absolute error floors.
"""
from __future__ import annotations

import numpy as np
from math import factorial
from scipy.signal import welch


def spectral_predictability(x: np.ndarray, fs: float = 1.0,
                            nperseg: int | None = None) -> float:
    """Omega = 1 - H_spec, H_spec = Shannon entropy of the normalized
    Welch power spectrum divided by log(#freq bins).

    Pure white noise  -> Omega ~ 0
    Pure sinusoid     -> Omega ~ 1
    """
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(1024, len(x))
    _, psd = welch(x - x.mean(), fs=fs, nperseg=nperseg)
    psd = psd[psd > 0]
    p = psd / psd.sum()
    h = -np.sum(p * np.log(p)) / np.log(len(p))
    return float(1.0 - h)


def permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 1,
                        normalize: bool = True) -> float:
    """Bandt–Pompe permutation entropy.

    order=4, delay=1 is a robust default for 20k-point series
    (rule of thumb: len(x) >> order!).
    """
    x = np.asarray(x, dtype=float)
    n = len(x) - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short for given order/delay")
    # embedding matrix, shape (n, order)
    idx = np.arange(order) * delay
    emb = x[np.arange(n)[:, None] + idx[None, :]]
    # ordinal patterns via argsort ranks
    patterns = np.argsort(np.argsort(emb, axis=1), axis=1)
    # hash each pattern row to a single int
    base = order ** np.arange(order)
    codes = patterns @ base
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    if normalize:
        h /= np.log(factorial(order))
    return float(h)


def permutation_predictability(x: np.ndarray, order: int = 4,
                               delay: int = 1) -> float:
    """H_perm proxy: 1 - normalized permutation entropy."""
    return 1.0 - permutation_entropy(x, order=order, delay=delay)


def measure_all(x: np.ndarray) -> dict:
    """Convenience: both proxies at defaults."""
    return {
        "omega_spec": spectral_predictability(x),
        "pred_perm": permutation_predictability(x),
    }
