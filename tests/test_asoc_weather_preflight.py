"""Read-only Weather audit regression tests; no GPU or external dataset required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_weather_preflight import audit_weather


class WeatherPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.csv = directory / "weather.csv"
        self.npy = directory / "weather.npy"
        t = np.arange(2000)
        self.frame = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=len(t), freq="10min"),
            "A": np.sin(t / 11), "B": np.cos(t / 15),
            "OT": 5 * np.sin(t / 11) + np.cos(t / 15) + t * 0.01,
        })
        self._save_pair()

    def _save_pair(self):
        self.frame.to_csv(self.csv, index=False)
        rebuilt = self.frame[["A", "B", "OT"]]
        # Same rank-by-train-correlation rule as scripts.prepare_real.
        target = "OT"
        tr = rebuilt.iloc[:int(len(rebuilt) * .7)]
        aux = tr[["A", "B"]].apply(lambda c: tr[target].corr(c)).abs().sort_values(ascending=False).index[:2].tolist()
        np.save(self.npy, rebuilt[[target] + aux].to_numpy(dtype=np.float32))

    def test_complete_and_read_only(self):
        old = self.npy.read_bytes()
        r = audit_weather(self.csv, self.npy)
        self.assertTrue(r["pass"], r["errors"])
        self.assertEqual(r["regimes"][0]["regime"], 0.10)
        self.assertEqual(len(r["auxiliary"]), 2)
        self.assertEqual(old, self.npy.read_bytes())

    def test_reject_changed_prepared_array(self):
        x = np.load(self.npy)
        x[500, 0] += 4
        np.save(self.npy, x)
        r = audit_weather(self.csv, self.npy)
        self.assertFalse(r["pass"])
        self.assertTrue(any("differs" in e for e in r["errors"]))

    def test_reject_missing_and_timestamp_gap(self):
        self.frame.loc[500, "A"] = np.nan
        self.frame.loc[700, "date"] = self.frame.loc[700, "date"] + pd.Timedelta(minutes=10)
        self._save_pair()
        r = audit_weather(self.csv, self.npy)
        self.assertFalse(r["pass"])
        self.assertGreater(r["raw_missing_total"], 0)
        self.assertGreater(r["timestamps"]["non_10min_intervals"], 0)


if __name__ == "__main__":
    unittest.main()
