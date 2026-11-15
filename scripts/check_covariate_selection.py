"""Quick check (analysis-only, no training): does regime-internal covariate
selection change the selected covariates?

For weather / traffic / ETTh1, selects the two |Pearson|-top covariates of
OT using (a) the full training span (first 70% of rows — as done in
prepare_real) and (b) only the last 10% of that training span (the scarce
regime's actual slice), and reports whether the selections agree.

EPİAŞ is excluded by design: its covariates are a-priori calendar features.

Run from the repo root:  python -m scripts.check_covariate_selection
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "weather": ROOT / "data/real/weather/weather.csv",
    "traffic": ROOT / "data/real/traffic/traffic.csv",
    "electricity": ROOT / "data/real/electricity/ETTh1.csv",
}


def top2(df: pd.DataFrame) -> list[tuple[str, float]]:
    c = df.corr(numeric_only=True)["OT"].abs().drop("OT").sort_values(ascending=False)
    return [(k, round(v, 3)) for k, v in c.head(2).items()]


def main():
    any_diff = False
    for name, path in PATHS.items():
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in ("date",) if c in df.columns])
        t1 = int(len(df) * 0.70)
        train = df.iloc[:t1]
        slice10 = train.iloc[int(len(train) * 0.90):]  # last 10% of train span
        full_sel = top2(train)
        slice_sel = top2(slice10)
        same = {k for k, _ in full_sel} == {k for k, _ in slice_sel}
        any_diff |= not same
        print(f"{name:12s} full-span top2: {full_sel}")
        print(f"{'':12s} 10%-slice top2: {slice_sel}   -> "
              f"{'SAME' if same else 'DIFFERENT'}")
    print()
    if any_diff:
        print("At least one domain selects different covariates under the "
              "10% slice.\nAffected runs to redo (only these): for each "
              "DIFFERENT domain, the scarce-regime iTransformer cells "
              "(generators none/vae/stl x 3 seeds = 9 runs per domain), "
              "after rebuilding that domain's .npy with slice-based "
              "selection. DLinear cells never see the covariates and are "
              "unaffected.")
    else:
        print("All selections identical: no rerun needed. Add one sentence "
              "to §3.6: 'A regime-internal reselection using only the 10% "
              "slice yields the same covariates in all three benchmark "
              "domains.'")


if __name__ == "__main__":
    main()
