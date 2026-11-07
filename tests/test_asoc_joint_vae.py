"""CPU validation for Phase 3B; no external datasets or CUDA required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.asoc_joint_vae import (JointWindowVAE, gen_joint_vae,
                                    shuffle_synthetic_aux,
                                    cross_channel_diagnostics)
from scripts.asoc_joint_controls import CONDITIONS, KEY_COLS, EXPERIMENT_VERSION
from scripts.asoc_joint_report import paired_table, summary_table


class JointVaeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(902)
        self.x = rng.standard_normal((18, 12, 3)).astype(np.float32)
        self.x[:, :, 1] = 0.85 * self.x[:, :, 0] + 0.12 * rng.standard_normal((18, 12))
        self.x[:, :, 2] = -0.4 * self.x[:, :, 0] + 0.12 * rng.standard_normal((18, 12))
        self.y = rng.standard_normal((18, 3)).astype(np.float32)

    def test_joint_generator_shapes_finite_and_repeatable(self):
        pair = (self.x, self.y)
        a, stats_a = gen_joint_vae(pair, seed=6, epochs=2, device=torch.device("cpu"),
                                   latent=4, hidden=24, batch_size=8)
        b, stats_b = gen_joint_vae(pair, seed=6, epochs=2, device=torch.device("cpu"),
                                   latent=4, hidden=24, batch_size=8)
        self.assertEqual(a[0].shape, self.x.shape)
        self.assertEqual(a[1].shape, self.y.shape)
        self.assertTrue(np.isfinite(a[0]).all())
        self.assertTrue(np.isfinite(a[1]).all())
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(stats_a, stats_b)

    def test_joint_decoder_y_is_future_target_only(self):
        model = JointWindowVAE(12, 3, 3, latent=4, hidden=24)
        x, y = model.decode(torch.randn(7, 4))
        self.assertEqual(x.shape, (7, 12, 3))
        self.assertEqual(y.shape, (7, 3))

    def test_shuffled_control_preserves_target_and_label(self):
        a = shuffle_synthetic_aux((self.x, self.y), seed=14)
        b = shuffle_synthetic_aux((self.x, self.y), seed=14)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[0][:, :, 0], self.x[:, :, 0])
        np.testing.assert_array_equal(a[1], self.y)
        self.assertFalse(np.array_equal(a[0][:, :, 1:], self.x[:, :, 1:]))
        self.assertLess(cross_channel_diagnostics((self.x, self.y), (self.x, self.y), max_lag=3)["lag_corr_gap"], 1e-12)
        self.assertLess(cross_channel_diagnostics((self.x, self.y), (self.x, self.y), max_lag=3)["future_assoc_gap"], 1e-12)

    def test_incompatible_inputs_rejected(self):
        with self.assertRaises(ValueError):
            gen_joint_vae((self.x[:, :, :1], self.y), seed=1, epochs=1, device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            shuffle_synthetic_aux((self.x[:, :, :1], self.y), seed=2)
        with self.assertRaises(ValueError):
            cross_channel_diagnostics((self.x, self.y), (self.x[:, :10], self.y), max_lag=2)

    def test_report_complete_steps_incomplete_and_summary(self):
        common = dict(file="om0.5_plain_s0.npy", regime=0.1, rep=0, pool_seed=0,
                      epochs=10, vae_epochs=30, batch_size=64, lr=.001,
                      latent=16, hidden=128, horizon_weight=.5,
                      experiment_version=EXPERIMENT_VERSION,
                      optimizer_steps=410, best_step=380, val_mse=.08,
                      lag_corr_gap=.2, future_assoc_gap=.3, zero_lag_corr_gap=.1)
        mse = [0.12, 0.10, 0.11, 0.09, 0.115]
        raw = pd.DataFrame([{**common, "condition": c, "test_mse": m}
                            for c, m in zip(CONDITIONS, mse)])
        paired, incomplete = paired_table(raw)
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(incomplete), 0)
        self.assertLess(paired.iloc[0]["joint_vs_original_pct"], 0)
        self.assertGreater(paired.iloc[0]["joint_shuffled_vs_joint_pct"], 0)
        self.assertEqual(summary_table(paired).iloc[0]["n_series_configurations"], 1)
        bad = raw.copy()
        bad.loc[bad.condition == CONDITIONS[-1], "optimizer_steps"] = 411
        with self.assertRaisesRegex(ValueError, "Unequal optimizer"):
            paired_table(bad)
        complete, incomplete = paired_table(raw.iloc[:-1])
        self.assertEqual(len(complete), 0)
        self.assertEqual(len(incomplete), 1)

    def test_resume_does_not_regenerate_finished_pilot(self):
        from scripts.asoc_joint_controls import run_one
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            data_path = folder / "tiny_s0.npy"
            rng = np.random.default_rng(9)
            t = np.arange(900, dtype=np.float32)
            array = np.column_stack((np.sin(t/20), np.sin(t/20+.4), np.cos(t/17))).astype(np.float32)
            np.save(data_path, array)
            out = folder / "results.csv"
            def fast_fit(*, train_pair, max_steps, validation_interval, **kwargs):
                return dict(test_mse=.07, test_mae=.2, val_mse=.08, best_step=max_steps,
                            optimizer_steps=max_steps, samples_seen=123,
                            train_windows=len(train_pair[0]), validation_checkpoints=1,
                            validation_curve="[]", train_seconds=0.01)
            with patch("scripts.asoc_joint_controls.fit_condition", side_effect=fast_fit):
                run_one(filename=str(data_path), regime=.25, rep=0,
                        pool_seed=0, epochs=1, vae_epochs=1,
                        batch_size=64, lr=.001, latent=4, hidden=24,
                        horizon_weight=.5, gpu_index=0, allow_cpu=True, out=out)
            rows = pd.read_csv(out)
            self.assertEqual(set(rows.condition), set(CONDITIONS))
            self.assertEqual(rows.optimizer_steps.nunique(), 1)
            with patch("scripts.asoc_joint_controls.gen_joint_vae", side_effect=AssertionError("should skip")):
                run_one(filename=str(data_path), regime=.25, rep=0,
                        pool_seed=0, epochs=1, vae_epochs=1,
                        batch_size=64, lr=.001, latent=4, hidden=24,
                        horizon_weight=.5, gpu_index=0, allow_cpu=True, out=out)


if __name__ == "__main__":
    unittest.main()
