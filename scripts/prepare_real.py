"""Stage-3 data preparation (protocol §3.3).

Converts raw downloads in data/real/<domain>/ into unified arrays:
  data/real/<domain>.npy   float32, shape (T, 3), channel 0 = target.

PRE-REGISTERED channel rule (fixed before any real-domain training):
  - Benchmark domains (weather, traffic, electricity/ETT): target = 'OT'
    column (or last column if 'OT' absent); aux = the TWO covariates with
    highest |Pearson r| to the target on the first 70% (training span).
  - EPİAŞ: target = PTF (TL/MWh); aux = calendar features
    hour-of-day (sin) and day-of-week (scaled) — exogenous by construction.

Expected raw files (place them like this):
  data/real/weather/weather.csv          (Autoformer benchmark format)
  data/real/traffic/traffic.csv          (Autoformer benchmark format)
  data/real/electricity/ETTh1.csv        (ETDataset format)
  data/real/epias/ptf.csv                (Şeffaflık export: Tarih;Saat;PTF...)

Run:  python -m scripts.prepare_real
Prints Ω_spec and H_perm per domain -> results/real_domains.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.measures import spectral_predictability, permutation_predictability  # noqa: E402

RAW = ROOT / "data" / "real"


def _pick_aux(df: pd.DataFrame, target: str) -> list[str]:
    """Two covariates most |corr|elated with target on the train span."""
    tr = df.iloc[: int(len(df) * 0.7)]
    corr = (tr.drop(columns=[target]).apply(
        lambda c: tr[target].corr(c)).abs().sort_values(ascending=False))
    return corr.index[:2].tolist()


def load_benchmark(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "date"])
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    target = "OT" if "OT" in df.columns else df.columns[-1]
    aux = _pick_aux(df, target)
    print(f"  {path.name}: target={target}, aux={aux}")
    return df[[target] + aux].to_numpy(dtype=np.float32)


def load_epias(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=";", dtype=str)
    dt = pd.to_datetime(df["Tarih"] + " " + df["Saat"],
                        format="%d.%m.%Y %H:%M")
    tl_col = [c for c in df.columns if "TL" in c][0]
    ptf = (df[tl_col].str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False).astype(float))
    out = pd.DataFrame({"dt": dt, "ptf": ptf}).dropna()
    dup = out["dt"].duplicated().sum()
    diffs = out["dt"].diff().value_counts()
    print(f"  epias: {len(out)} rows, duplicate timestamps={dup}, "
          f"gap profile={dict(list(diffs.items())[:3])}")
    out = out.drop_duplicates("dt")           # DST artifacts: keep first
    hour = out["dt"].dt.hour.to_numpy()
    dow = out["dt"].dt.dayofweek.to_numpy()
    X = np.stack([out["ptf"].to_numpy(),
                  np.sin(2 * np.pi * hour / 24),
                  dow / 6.0], axis=1).astype(np.float32)
    return X


LOADERS = {
    "weather": lambda: load_benchmark(RAW / "weather" / "weather.csv"),
    "traffic": lambda: load_benchmark(RAW / "traffic" / "traffic.csv"),
    "electricity": lambda: load_benchmark(RAW / "electricity" / "ETTh1.csv"),
    "epias": lambda: load_epias(RAW / "epias" / "ptf.csv"),
}


def main():
    rows = []
    for name, fn in LOADERS.items():
        try:
            X = fn()
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
            continue
        np.save(RAW / f"{name}.npy", X)
        tgt = X[:, 0].astype(float)
        rows.append({
            "domain": name, "T": len(X), "channels": X.shape[1],
            "omega_spec": spectral_predictability(tgt),
            "pred_perm": permutation_predictability(tgt),
        })
        print(f"  -> {name}.npy {X.shape}  Ω={rows[-1]['omega_spec']:.3f} "
              f"perm={rows[-1]['pred_perm']:.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "real_domains.csv",
                              index=False)
    print("\nwrote results/real_domains.csv")


if __name__ == "__main__":
    main()
