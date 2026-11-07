"""Unit checks for phase-3 pairing and balanced-input contrasts (CPU only)."""
import unittest

import numpy as np
import pandas as pd

from scripts.asoc_channel_controls import (
    CONDITIONS, KEY_COLS, context_features, nearest_context_aux,
    permute_duplicate_aux, correlation_gap,
)
from scripts.asoc_channel_report import paired_table, summary_table


class ChannelControlTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = rng.standard_normal((24, 12, 3)).astype(np.float32)
        self.x[:, :, 1] = 0.75 * self.x[:, :, 0] + 0.2 * rng.standard_normal((24, 12))
        self.x[:, :, 2] = -0.3 * self.x[:, :, 0] + 0.1 * rng.standard_normal((24, 12))
        self.y = rng.standard_normal((24, 3)).astype(np.float32)

    def test_context_pairing_preserves_synthetic_target_and_uses_real_aux(self):
        syn_x = self.x[[4, 13, 1]].copy()
        syn_y = self.y[[4, 13, 1]].copy()
        syn_x[:, :, 1:] = 777.0   # deliberately unrelated original aux
        result, metrics = nearest_context_aux(self.x, (syn_x, syn_y), bins=3)
        np.testing.assert_array_equal(result[0][:, :, 0], syn_x[:, :, 0])
        np.testing.assert_array_equal(result[1], syn_y)
        np.testing.assert_array_equal(result[0][:, :, 1:], self.x[[4, 13, 1], :, 1:])
        self.assertLess(metrics["context_distance_median"], 1e-8)

    def test_permutation_is_target_preserving_and_deterministic(self):
        a = permute_duplicate_aux((self.x, self.y), pool_seed=11)
        b = permute_duplicate_aux((self.x, self.y), pool_seed=11)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[0][:len(self.x)], self.x)
        np.testing.assert_array_equal(a[0][:, :, 0], np.concatenate([self.x[:, :, 0]] * 2))
        np.testing.assert_array_equal(a[1], np.concatenate([self.y] * 2))
        self.assertFalse(np.array_equal(a[0][len(self.x):, :, 1:], self.x[:, :, 1:]))
        self.assertLess(correlation_gap(self.x, self.x)["mean_abs_corr_gap"], 1e-12)

    def test_validates_binning(self):
        with self.assertRaises(ValueError):
            context_features(self.x, bins=5)

    def test_complete_paired_table_and_step_budget(self):
        base = dict(file="om0.5_plain_s0.npy", regime=.1, rep=0,
                    pool_seed=0, epochs=10, vae_epochs=30, batch_size=64,
                    lr=.001, context_bins=12, optimizer_steps=410, val_mse=.2)
        mse = dict(zip(CONDITIONS, [.13, .12, .115, .10, .09, .14, .12, .105]))
        df = pd.DataFrame([{**base, "condition": c, "test_mse": mse[c]} for c in CONDITIONS])
        result, incomplete = paired_table(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(incomplete), 0)
        self.assertGreater(result.iloc[0]["all_vae_random_vs_duplicate_pct"], 0)
        self.assertLess(result.iloc[0]["all_vae_context_vs_random_pct"], 0)
        summary = summary_table(result)
        self.assertEqual(summary.iloc[0]["n_training_cells"], 1)
        changed = df.copy()
        changed.loc[changed.condition == "itr_all_vae_context", "optimizer_steps"] = 450
        with self.assertRaises(ValueError):
            paired_table(changed)
        missing = df.iloc[:-1]
        good, bad = paired_table(missing)
        self.assertEqual(len(good), 0)
        self.assertEqual(len(bad), 1)


if __name__ == "__main__":
    unittest.main()
