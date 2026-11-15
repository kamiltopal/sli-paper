"""Small deterministic checks without a GPU. Run with unittest discovery."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.asoc_budget_matrix import planned_cells, completed_conditions, REQUIRED_CONDITIONS
from scripts.asoc_budget_report import paired_cells, series_summary


class BudgetMatrixReportTests(unittest.TestCase):
    def test_plan_distinguishes_series_and_training_seeds(self):
        cells = list(planned_cells(
            levels=(.2, .5, .8), series_seeds=(0, 1, 2), variants=("plain",),
            regimes=(.1, 1.), reps=(0, 1, 2), arch="dlinear", epochs=10,
            vae_epochs=30, batch_size=64, lr=.001, input_mode="target"))
        self.assertEqual(len(cells), 54)
        self.assertEqual(len({c['file'] for c in cells}), 9)
        self.assertEqual(len({c['rep'] for c in cells}), 3)

    def test_result_metrics_and_incomplete(self):
        base = dict(file="om0.5_plain_s0.npy", regime=.1, arch="dlinear", rep=0,
                    pool_seed=0, input_mode="target", epochs=10, batch_size=64,
                    lr=.001, vae_epochs=30)
        rows = []
        for cond, mse, best, steps in (
                ("epoch_baseline", 0.136, 210, 210),
                ("step_baseline", 0.124, 410, 410),
                ("duplicate_real", 0.120, 410, 410),
                ("vae_augmented", 0.123, 400, 410)):
            rows.append({**base, "condition": cond, "test_mse": mse,
                         "val_mse": mse, "best_step": best, "optimizer_steps": steps})
        rows.append({**base, "rep": 1, "condition": "epoch_baseline", "test_mse": .13,
                     "val_mse": .13, "best_step": 210, "optimizer_steps": 210})
        df = pd.DataFrame(rows)
        paired, incomplete = paired_cells(df)
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(incomplete), 1)
        self.assertAlmostEqual(paired.iloc[0].vae_vs_step_pct, 100 * (.123 - .124) / .124)
        self.assertFalse(bool(paired.iloc[0].vae_beats_both_matched))
        self.assertFalse(bool(paired.iloc[0].best_at_last_vae_augmented))
        summary = series_summary(paired)
        self.assertEqual(int(summary.iloc[0].n_series_configurations), 1)
        self.assertEqual(int(summary.iloc[0].n_training_cells), 1)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "partial.csv"
            df.to_csv(out, index=False)
            checked = completed_conditions(out)
            self.assertIn("vae_augmented", checked[next(k for k in checked if k[3] == 0)])
            self.assertEqual(set(checked[next(k for k in checked if k[3] == 0)]), REQUIRED_CONDITIONS)

    def test_duplicate_check(self):
        r = dict(file="x", regime=.1, arch="dlinear", rep=0,
                 pool_seed=0, input_mode="target", epochs=10, batch_size=64,
                 lr=.001, vae_epochs=30, condition="vae_augmented",
                 test_mse=1, val_mse=1, best_step=1, optimizer_steps=1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_cells(pd.DataFrame([r, r]))


if __name__ == "__main__":
    unittest.main()
