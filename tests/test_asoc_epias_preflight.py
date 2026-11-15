"""Read-only EPİAŞ gate tests, with synthetic price series and timestamps."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_epias_preflight import audit, reconstruct, time_audit


class EpiasPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "ptf.csv"
        self.npy = self.root / "epias.npy"
        self.runs = self.root / "real_budget_controls.csv"
        self.times = pd.date_range("2020-01-01", periods=3000, freq="h")
        self.write()

    def write(self, *, duplicate_at=None, gap_at=None, reorder_at=None,
              changed=False, conflicting=False):
        times = list(self.times)
        if duplicate_at is not None:
            times[duplicate_at] = times[duplicate_at-1]
        if gap_at is not None:
            for i in range(gap_at, len(times)):
                times[i] += pd.Timedelta(hours=1)
        if reorder_at is not None:
            times[reorder_at],times[reorder_at-1] = times[reorder_at-1],times[reorder_at]
        values = [200+i/1000 for i in range(len(times))]
        if duplicate_at is not None and not conflicting:
            values[duplicate_at] = values[duplicate_at-1]
        df = pd.DataFrame({
            "Tarih": [t.strftime("%d.%m.%Y") for t in times],
            "Saat": [t.strftime("%H:%M") for t in times],
            "PTF (TL/MWh)": [f"{v:.3f}".replace('.',',') for v in values],
        })
        df.to_csv(self.csv, sep=";", index=False)
        prepared, _, _ = reconstruct(pd.read_csv(self.csv, sep=';', dtype=str))
        np.save(self.npy, prepared + (1 if changed else 0))

    def get(self, regime=0.1):
        return audit(csv_path=self.csv, npy_path=self.npy,
                     runs_path=self.runs, regime=regime)

    def test_clean_source_has_matching_array_and_steps(self):
        result = self.get()
        self.assertTrue(result["pass"])
        self.assertTrue(result["raw_matches_prepared"])
        self.assertEqual(result["prepared_shape"], [3000,3])
        self.assertEqual(result["split"]["train_start"], 1890)
        self.assertEqual(result["split"]["real_train_windows"], 91)

    def test_duplicate_outside_selected_training_can_be_disclosed_without_blocking(self):
        self.write(duplicate_at=100)
        result = self.get()
        self.assertTrue(result["pass"])
        self.assertEqual(result["raw_preparation"]["removed_duplicate_timestamps"],1)
        self.assertEqual(result["time_audit"]["selected_regime_irregular_intervals"],0)
        self.assertTrue(result["warnings"])

    def test_gap_inside_selected_training_blocks(self):
        self.write(gap_at=2000)
        result = self.get()
        self.assertFalse(result["pass"])
        self.assertEqual(result["time_audit"]["selected_regime_irregular_intervals"],1)

    def test_gap_in_validation_blocks(self):
        self.write(gap_at=2250)
        result = self.get()
        self.assertFalse(result["pass"])
        self.assertGreaterEqual(result["time_audit"]["selected_regime_irregular_intervals"],1)

    def test_full_training_catches_early_gap(self):
        self.write(gap_at=100)
        self.assertTrue(self.get(0.1)["pass"])
        self.assertFalse(self.get(1.0)["pass"])

    def test_prepared_provenance_mismatch_fails_closed(self):
        self.write(changed=True)
        result = self.get()
        self.assertFalse(result["pass"])
        self.assertFalse(result["raw_matches_prepared"])

    def test_existing_sha_mismatch_fails_closed(self):
        pd.DataFrame([{"domain":"epias","data_sha256":"other"}]).to_csv(self.runs,index=False)
        result = self.get()
        self.assertFalse(result["pass"])
        self.assertTrue(any("different data SHA" in x for x in result["errors"]))

    def test_no_input_files_modified_by_audit(self):
        before = (self.csv.read_bytes(), self.npy.read_bytes())
        _ = self.get()
        self.assertEqual(before,(self.csv.read_bytes(), self.npy.read_bytes()))

    def test_missing_raw_blocks(self):
        self.csv.unlink()
        self.assertFalse(self.get()["pass"])

    def test_out_of_order_blocks_even_if_reconstruction_matches(self):
        self.write(reorder_at=2050)
        self.assertFalse(self.get()["pass"])


if __name__ == "__main__":
    unittest.main()
