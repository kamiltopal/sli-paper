"""Phase 4: boundary-aware bootstrap controls (training segment only).

The original block bootstrap is reproduced unchanged through src.generators.
Here we recover its exact sampled segment and index selection solely to audit
boundary crossings. The safer variant shares its sampled blocks but excludes
any window spanning a splice, then resamples eligible windows with replacement
so its pool size is identical to the original. The decomposition-residual
variant uses training-only trend/period profiles and contiguous residual
WINDOWS (not concatenated residual blocks). It is NOT statistical STL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapDiagnostics:
    total_windows: int
    candidate_safe_windows: int
    original_crossing_fraction: float
    safe_crossing_fraction: float
    boundary_jump_p95: float
    regular_jump_p95: float
    safe_unique_source_fraction: float

    def as_dict(self):
        return dict(total_windows=self.total_windows,
                    candidate_safe_windows=self.candidate_safe_windows,
                    candidate_safe_fraction=self.candidate_safe_windows / self.total_windows,
                    original_crossing_fraction=self.original_crossing_fraction,
                    safe_crossing_fraction=self.safe_crossing_fraction,
                    boundary_jump_p95=self.boundary_jump_p95,
                    regular_jump_p95=self.regular_jump_p95,
                    safe_unique_source_fraction=self.safe_unique_source_fraction)


def _window_config_check(segment: np.ndarray, cfg, block: int):
    if segment.ndim != 2 or not np.isfinite(segment).all():
        raise ValueError("Expected finite (T, C) training segment")
    T, C = segment.shape
    W = cfg.lookback + cfg.horizon
    if C < 1 or cfg.lookback <= 0 or cfg.horizon <= 0 or block < W or T <= block:
        raise ValueError(f"Require T>block>=L+H; got T={T}, block={block}, W={W}")
    return T, C, W


def bootstrap_pair_and_audit(segment: np.ndarray, cfg, *, block: int = 200,
                             seed: int = 0):
    """Return frozen original, size-matched safe pool, and boundary diagnostics.

    Reproduces src.generators.gen_bootstrap with the same PCG64 choice call.
    Boundary-safe windows are then drawn from exactly the same resampled blocks
    but only from contiguous interiors. Replacement is essential because ~60%
    of candidate windows may cross concatenation boundaries when W=120,B=200.
    """
    from src.data import make_windows
    from src.generators import gen_bootstrap

    T, _, W = _window_config_check(segment, cfg, block)
    rng = np.random.default_rng(seed)
    n_blocks = T // block + 1
    starts = rng.integers(0, T - block, size=n_blocks)
    sampled = np.concatenate([segment[s:s + block] for s in starts], axis=0)[:T]
    windows, future = make_windows(sampled, 0, len(sampled), cfg)
    n = int(max(1, T - W))  # exactly the frozen generator's ratio=1 length
    original_keep = rng.choice(len(windows), size=min(n, len(windows)), replace=False)
    # Check the original frozen reference. Never quietly reinterpret old method.
    original = gen_bootstrap(segment, cfg, ratio=1.0, block=block, seed=seed)
    if not (np.array_equal(original[0], windows[original_keep]) and
            np.array_equal(original[1], future[original_keep])):
        raise RuntimeError("Frozen original bootstrap differs from traced sampler")

    candidate_starts = np.arange(len(windows))
    crosses = (candidate_starts // block) != ((candidate_starts + W - 1) // block)
    eligible = np.flatnonzero(~crosses)
    if len(eligible) < 1:
        raise ValueError("No boundary-safe windows; increase block length")
    safe_rng = np.random.default_rng(seed + 41003)
    chosen = safe_rng.choice(eligible, size=len(original_keep), replace=True)
    safe = windows[chosen].copy(), future[chosen].copy()

    boundaries = np.arange(block, T, block)
    jumps = np.abs(np.diff(sampled[:, 0].astype(np.float64)))
    edge_idx = boundaries - 1
    regular_mask = np.ones(len(jumps), dtype=bool)
    regular_mask[edge_idx] = False
    info = BootstrapDiagnostics(
        total_windows=len(windows), candidate_safe_windows=len(eligible),
        original_crossing_fraction=float(crosses[original_keep].mean()),
        safe_crossing_fraction=float(crosses[chosen].mean()),
        boundary_jump_p95=float(np.quantile(jumps[edge_idx], 0.95)),
        regular_jump_p95=float(np.quantile(jumps[regular_mask], 0.95)),
        safe_unique_source_fraction=float(len(np.unique(chosen)) / len(chosen)),
    )
    return original, safe, info


def train_only_decomposition(segment: np.ndarray, *,
                             trend_window: int = 169,
                             periods: tuple[int, ...] = (24, 168)):
    """Deterministic training-only moving-average trend + seasonal phase means.

    Fixed periods are generic working assumptions, not oracle Fourier periods.
    Not STL; no loess, robust iteration or automatic frequency selection.
    For synthetic Fourier streams these fixed periods may be misspecified.
    """
    if segment.ndim != 2 or len(segment) < 3 or not np.isfinite(segment).all():
        raise ValueError("Expected finite training-only multichannel segment")
    T, C = segment.shape
    if trend_window < 3 or trend_window % 2 != 1:
        raise ValueError("trend_window must be odd and >=3")
    if not periods or any(p < 2 for p in periods) or len(set(periods)) != len(periods):
        raise ValueError("periods must be distinct integers >=2")
    k = min(trend_window, T if T % 2 else T - 1)
    half = k // 2
    padded = np.pad(segment.astype(np.float64), ((half, half), (0, 0)), mode="edge")
    trend = np.empty((T, C), dtype=np.float64)
    kernel = np.ones(k, dtype=np.float64) / k
    for c in range(C):
        trend[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    remaining = segment.astype(np.float64) - trend
    base = trend.copy()
    for p in periods:
        if T < 2 * p:
            continue
        profile = np.stack([remaining[i::p].mean(axis=0) for i in range(p)])
        periodic = profile[np.arange(T) % p]
        base += periodic
        remaining -= periodic
    return base.astype(np.float32), remaining.astype(np.float32)


def residual_window_pool(segment: np.ndarray, cfg, *, n: int,
                         seed: int = 0, trend_window: int = 169,
                         periods: tuple[int, ...] = (24, 168)):
    """Compose base windows with contiguous multichannel residual windows.

    For each synthetic example, the deterministic component is taken from a
    training time i; the residual L+H-window from another training time j.
    All channels of the residual block move together, preserving *within-block*
    cross-channel dependence. No concatenation seams occur INSIDE a window.
    Generated windows are not claimed to form a coherent long trajectory.
    """
    if n <= 0:
        raise ValueError("Synthetic pool must be nonempty")
    T, C, W = _window_config_check(segment, cfg, block=cfg.lookback + cfg.horizon)
    base, remainder = train_only_decomposition(
        segment, trend_window=trend_window, periods=periods)
    rng = np.random.default_rng(seed + 42007)
    origin = rng.integers(0, T - W + 1, size=n)
    source = rng.integers(0, T - W + 1, size=n)
    offsets = np.arange(W)[None, :]
    full = base[origin[:, None] + offsets] + remainder[source[:, None] + offsets]
    Xs = full[:, :cfg.lookback, :].astype(np.float32)
    ys = full[:, cfg.lookback:, 0].astype(np.float32)
    if Xs.shape != (n, cfg.lookback, C) or ys.shape != (n, cfg.horizon):
        raise RuntimeError("Unexpected residual-bootstrap shapes")
    return (Xs, ys), {
        "residual_origin_unique_fraction": float(len(np.unique(origin)) / n),
        "residual_source_unique_fraction": float(len(np.unique(source)) / n),
        "residual_training_std": float(remainder[:, 0].std()),
    }
