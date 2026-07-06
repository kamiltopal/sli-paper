"""Synthetic backbone signal generator (protocol §3.2).

Signals are Fourier composites whose spectral predictability Omega is
*set by construction*: a deterministic multi-sinusoid core is mixed with
white noise, and the mixing weight is calibrated by bisection so that
the measured Omega of the generated series hits the requested target.

Variants:
  - "plain"    : sinusoid mix + white noise (the calibrated backbone)
  - "seasonal_ar": adds AR(1)-correlated noise instead of white
  - "regime"   : two alternating amplitude/frequency regimes

Every series is reproducible from (target_omega, seed, variant).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .measures import spectral_predictability


@dataclass
class SignalSpec:
    target_omega: float
    seed: int
    variant: str = "plain"       # plain | seasonal_ar | regime
    n_points: int = 20_000
    n_components: int = 6        # sinusoids in the deterministic core
    ar_coef: float = 0.7         # for seasonal_ar
    regime_period: int = 4_000   # for regime switching
    achieved_omega: float | None = None
    mix_weight: float | None = None
    concentration: float | None = None


def _deterministic_core(rng: np.random.Generator, spec: SignalSpec,
                        concentration: float = 1.0) -> np.ndarray:
    """Sum of sinusoids with random incommensurate freqs and phases.

    `concentration` in (0, 1]: geometric amplitude decay ratio.
    1.0 -> near-equal amplitudes (energy spread, lower max Omega);
    ->0 -> single dominant sinusoid (energy concentrated, Omega -> ~0.86+).
    """
    t = np.arange(spec.n_points, dtype=float)
    freqs = rng.uniform(0.002, 0.12, size=spec.n_components)
    phases = rng.uniform(0, 2 * np.pi, size=spec.n_components)
    amps = concentration ** np.arange(spec.n_components, dtype=float)
    core = np.zeros(spec.n_points)
    for f, ph, a in zip(freqs, phases, amps):
        core += a * np.sin(2 * np.pi * f * t + ph)
    return core / core.std()


def _noise(rng: np.random.Generator, spec: SignalSpec) -> np.ndarray:
    if spec.variant == "seasonal_ar":
        eps = rng.standard_normal(spec.n_points)
        ar = np.empty(spec.n_points)
        ar[0] = eps[0]
        for i in range(1, spec.n_points):
            ar[i] = spec.ar_coef * ar[i - 1] + eps[i]
        return ar / ar.std()
    return rng.standard_normal(spec.n_points)


def _apply_regime(core: np.ndarray, rng: np.random.Generator,
                  spec: SignalSpec) -> np.ndarray:
    """Alternate amplitude scaling in blocks -> regime switching."""
    if spec.variant != "regime":
        return core
    out = core.copy()
    scales = rng.uniform(0.4, 1.6, size=spec.n_points // spec.regime_period + 1)
    for b, s in enumerate(scales):
        lo = b * spec.regime_period
        hi = min((b + 1) * spec.regime_period, spec.n_points)
        out[lo:hi] *= s
    return out / out.std()


def _mix(core: np.ndarray, noise: np.ndarray, w: float) -> np.ndarray:
    """w in [0,1]: weight of the deterministic core."""
    x = w * core + (1.0 - w) * noise
    return (x - x.mean()) / x.std()


def _build_core(rng_seed: int, spec: SignalSpec,
                concentration: float) -> np.ndarray:
    """Core with regime variant applied, from a fixed seed (so both
    calibration stages see the same frequencies/phases/regimes)."""
    rng = np.random.default_rng(rng_seed)
    core = _deterministic_core(rng, spec, concentration)
    return _apply_regime(core, rng, spec)


def generate(spec: SignalSpec, tol: float = 0.01, margin: float = 0.04,
             max_iter: int = 40) -> tuple[np.ndarray, SignalSpec]:
    """Generate a series whose measured Omega hits spec.target_omega
    within tol, via two-stage calibration:

      Stage A: bisect the core's amplitude `concentration` until the
               noise-free core has Omega >= target + margin (the mix
               stage can only interpolate downward from the core).
      Stage B: bisect the core/noise mixing weight w to hit the target.

    Returns (series, spec_with_achieved_fields).
    """
    rng = np.random.default_rng(spec.seed)
    noise = _noise(rng, spec)

    # --- Stage A: concentration so that core Omega exceeds target ---------
    conc = 1.0
    core = _build_core(spec.seed, spec, conc)
    need = spec.target_omega + margin
    if spectral_predictability(core) < need:
        lo_c, hi_c = 0.02, 1.0          # smaller conc => higher Omega
        for _ in range(max_iter):
            conc = 0.5 * (lo_c + hi_c)
            core = _build_core(spec.seed, spec, conc)
            om_c = spectral_predictability(core)
            if abs(om_c - need) < tol:
                break
            if om_c < need:
                hi_c = conc
            else:
                lo_c = conc

    # --- Stage B: mixing weight to hit the target --------------------------
    lo, hi = 0.0, 1.0
    w = 0.5
    x = _mix(core, noise, w)
    for _ in range(max_iter):
        om = spectral_predictability(x)
        if abs(om - spec.target_omega) < tol:
            break
        if om < spec.target_omega:
            lo = w
        else:
            hi = w
        w = 0.5 * (lo + hi)
        x = _mix(core, noise, w)

    spec.achieved_omega = spectral_predictability(x)
    spec.mix_weight = w
    spec.concentration = conc
    return x, spec


def generate_multichannel(spec: SignalSpec, n_aux: int = 2,
                          aux_noise: float = 0.6,
                          tol: float = 0.01) -> tuple[np.ndarray, SignalSpec]:
    """Target channel calibrated as in `generate`, plus `n_aux` auxiliary
    channels sharing the same deterministic core with independent noise.

    Returns (X, spec) where X has shape (n_points, 1 + n_aux); column 0 is
    the calibrated target. Channel-mixing architectures can exploit the
    auxiliaries; channel-independent ones cannot — this is what makes H3
    testable on the backbone.
    """
    x, spec = generate(spec, tol=tol)
    core = _build_core(spec.seed, spec, spec.concentration)
    rng = np.random.default_rng(spec.seed + 10_000)
    chans = [x]
    for _ in range(n_aux):
        eps = rng.standard_normal(spec.n_points)
        aux = (1 - aux_noise) * core + aux_noise * eps
        chans.append((aux - aux.mean()) / aux.std())
    return np.stack(chans, axis=1), spec


def make_backbone(levels=(0.2, 0.5, 0.8), seeds=range(6),
                  variants=("plain",), n_points=20_000):
    """Yield (series, spec) for the frozen v1 backbone grid."""
    for om in levels:
        for variant in variants:
            for seed in seeds:
                spec = SignalSpec(target_omega=om, seed=seed,
                                  variant=variant, n_points=n_points)
                yield generate(spec)


def spec_to_row(spec: SignalSpec) -> dict:
    return asdict(spec)
