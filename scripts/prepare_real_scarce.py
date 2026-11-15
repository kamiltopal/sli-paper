"""Rebuild the three benchmark domains' 3-channel arrays with covariates
selected from ONLY the scarce slice (last 10% of the training span).

Writes data/real/<domain>_scarce.npy with channels [OT, cov1, cov2].
EPİAŞ is excluded (a-priori calendar covariates; nothing to reselect).

Run from repo root:  python -m scripts.prepare_real_scarce
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    "weather": ROOT / "data/real/weather/weather.csv",
    "traffic": ROOT / "data/real/traffic/traffic.csv",
    "electricity": ROOT / "data/real/electricity/ETTh1.csv",
}


def main():
    for name, path in SRC.items():
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in ("date",) if c in df.columns])
        t1 = int(len(df) * 0.70)
        train = df.iloc[:t1]
        slice10 = train.iloc[int(len(train) * 0.90):]
        corr = (slice10.corr(numeric_only=True)["OT"].abs()
                .drop("OT").sort_values(ascending=False))
        covs = list(corr.head(2).index)
        X = df[["OT"] + covs].to_numpy(dtype=np.float32)
        out = ROOT / "data/real" / f"{name}_scarce.npy"
        np.save(out, X)
        print(f"{name:12s} covs={covs} (|r|={corr.head(2).round(3).tolist()})"
              f" -> {out.name}  shape={X.shape}")


if __name__ == "__main__":
    main()
