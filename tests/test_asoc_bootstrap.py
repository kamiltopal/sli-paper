"""Phase 4 controls: contamination, isolation, resume and paired budgets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_bootstrap_generators import (bootstrap_pair_and_audit,
    residual_window_pool, train_only_decomposition)
from scripts.asoc_bootstrap_controls import CONDITIONS, KEY_COLS, _done, _concat
from scripts.asoc_bootstrap_matrix import planned_cells, completed
from scripts.asoc_bootstrap_report import paired_table, summary_table
from src.data import WindowConfig, make_windows


class BootstrapPhase4Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        t = np.arange(600, dtype=np.float32)
        self.X = np.column_stack([
            0.2 * np.cos(t / 10) + .01 * t + rng.normal(0, .02, len(t)),
            np.sin(t / 13) + rng.normal(0, .02, len(t)),
            np.cos(t / 15) + rng.normal(0, .02, len(t))]).astype(np.float32)
        self.cfg = WindowConfig(lookback=24, horizon=8)

    def test_frozen_reproduction_and_seams(self):
        from src.generators import gen_bootstrap
        orig, safe, d = bootstrap_pair_and_audit(self.X, self.cfg, block=60, seed=3)
        frozen = gen_bootstrap(self.X, self.cfg, ratio=1.0, block=60, seed=3)
        self.assertTrue(np.array_equal(orig[0], frozen[0]))
        self.assertTrue(np.array_equal(orig[1], frozen[1]))
        self.assertEqual(orig[0].shape, safe[0].shape)
        self.assertEqual(orig[1].shape, safe[1].shape)
        self.assertGreater(d.original_crossing_fraction, .0)
        self.assertEqual(d.safe_crossing_fraction, 0.)
        self.assertLess(d.candidate_safe_windows, d.total_windows)
        self.assertLessEqual(d.safe_unique_source_fraction, 1.0)

    def test_reproducible_no_future_or_test_dependency(self):
        a = bootstrap_pair_and_audit(self.X, self.cfg, block=60, seed=4)
        b = bootstrap_pair_and_audit(self.X.copy(), self.cfg, block=60, seed=4)
        self.assertTrue(np.array_equal(a[1][0], b[1][0]))
        self.assertEqual(a[2], b[2])
        p1, _ = residual_window_pool(self.X, self.cfg, n=100, seed=8,
                                     trend_window=41, periods=(8, 24))
        p2, _ = residual_window_pool(self.X.copy(), self.cfg, n=100, seed=8,
                                     trend_window=41, periods=(8, 24))
        self.assertTrue(np.array_equal(p1[0], p2[0]))
        self.assertTrue(np.array_equal(p1[1], p2[1]))
        self.assertEqual(p1[0].shape, (100, 24, 3))
        self.assertEqual(p1[1].shape, (100, 8))
        self.assertTrue(np.isfinite(p1[0]).all())
        self.assertTrue(np.isfinite(p1[1]).all())
        # Segment-only generators cannot see changes outside the train segment.
        complete = np.concatenate((self.X, 10_000 + self.X))
        self.assertTrue(np.array_equal(
            residual_window_pool(complete[:600], self.cfg, n=100, seed=8,
                trend_window=41, periods=(8, 24))[0][0], p1[0]))

    def test_residual_window_contiguity_and_invalid(self):
        base, resid = train_only_decomposition(self.X, trend_window=41, periods=(8,24))
        np.testing.assert_allclose(base + resid, self.X, rtol=1e-5, atol=2e-5)
        with self.assertRaises(ValueError):
            train_only_decomposition(self.X, trend_window=40)
        with self.assertRaises(ValueError):
            bootstrap_pair_and_audit(self.X, self.cfg, block=20)

    def _raw(self):
        common = dict(file="om0.5_plain_s0.npy", regime=.1, arch="dlinear",
                      rep=0, pool_seed=0, epochs=1, batch_size=64, lr=.001,
                      block=200, trend_window=169, periods="24,168",
                      version="phase4_window_bootstrap_v1")
        rows = []
        for c, mse in zip(CONDITIONS, (1.0, .95, 1.2, .90, .89)):
            rows.append({**common, "condition": c, "test_mse": mse,
                "val_mse": .3, "optimizer_steps": 100, "best_step": 50,
                "original_crossing_fraction": .5, "safe_crossing_fraction": 0.,
                "candidate_safe_fraction": .4, "boundary_jump_p95": 2.,
                "regular_jump_p95": .5, "safe_unique_source_fraction": .6,
                "residual_training_std": .2})
        return pd.DataFrame(rows)

    def test_report_paired_and_mismatched_steps(self):
        raw = self._raw()
        paired, incomplete = paired_table(raw)
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(incomplete), 0)
        self.assertAlmostEqual(float(paired.safe_vs_original_pct.iloc[0]), -25.)
        summary = summary_table(paired)
        self.assertEqual(int(summary.n_training_cells.iloc[0]), 1)
        raw.loc[0, "optimizer_steps"] = 99
        with self.assertRaisesRegex(ValueError, "Optimizer"):
            paired_table(raw)

    def test_resumption_reject_duplicate_and_partial(self):
        raw = self._raw()
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "phase4.csv"
            raw.iloc[:3].to_csv(file, index=False)
            done = _done(file)
            self.assertEqual(len(done), 3)
            grp = completed(file)
            self.assertEqual(len(grp), 1)
            self.assertEqual(len(next(iter(grp.values()))), 3)
            partial, incomplete = paired_table(pd.read_csv(file))
            self.assertEqual(len(partial), 0)
            self.assertEqual(len(incomplete), 1)
            pd.concat((raw, raw.iloc[:1])).to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _done(file)

    def test_matrix_and_aug_pool(self):
        cells = list(planned_cells(levels=(.2, .5, .8), series_seeds=(0, 1),
            variants=("plain",), regimes=(.1, 1.), reps=(0,1), arch="dlinear",
            epochs=10, batch_size=64, lr=.001, block=200, trend_window=169,
            periods="24,168"))
        self.assertEqual(len(cells), 24)
        self.assertEqual(len({tuple(cell.values()) for cell in cells}), 24)
        real = make_windows(self.X, 0, len(self.X), self.cfg)
        added = _concat(real, real)
        self.assertEqual(len(added[0]), 2 * len(real[0]))


if __name__ == "__main__":
    unittest.main()
