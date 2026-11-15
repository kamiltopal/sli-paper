"""Fast tests for budget matching, initializer pairing, and validation selection.

Run: python -m unittest discover -s tests -p 'test_asoc_budget_controls.py'
"""
import json
import math
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts.asoc_budget_controls import count_steps, fit_condition, only_channels, select_device


class TestControls(unittest.TestCase):
    def test_budget_math(self):
        self.assertEqual(count_steps(101, 101, 64, 10), (20, 40))
        with self.assertRaises(ValueError):
            count_steps(0, 1, 64, 10)

    def test_device_selection_is_gpu_first_and_cpu_fallback_explicit(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU not available"):
                select_device()
            self.assertEqual(select_device(allow_cpu=True).type, "cpu")

    def test_target_only(self):
        p = (np.ones((5, 6, 3), dtype="float32"), np.zeros((5, 2), dtype="float32"))
        self.assertEqual(only_channels(p, "target")[0].shape, (5, 6, 1))
        self.assertEqual(only_channels(p, "all")[0].shape, (5, 6, 3))

    def test_seeded_repeat(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(40, 6, 1)).astype("float32")
        y = x[:, -2:, 0].copy()
        p = (x, y)
        def model_factory(_arch, lookback, horizon):
            return nn.Sequential(nn.Flatten(), nn.Linear(lookback, horizon))
        kw = dict(arch="dlinear", train_pair=p, val_pair=p, test_pair=p,
                  init_seed=7, max_steps=4, validation_interval=2, batch_size=16,
                  lr=1e-3, device=torch.device("cpu"), model_factory=model_factory)
        a = fit_condition(**kw)
        b = fit_condition(**kw)
        self.assertEqual(a["optimizer_steps"], 4)
        self.assertEqual(a["validation_checkpoints"], 2)
        self.assertEqual(json.loads(a["validation_curve"])[-1][0], 4)
        self.assertEqual(a["best_step"], b["best_step"])
        self.assertAlmostEqual(a["test_mse"], b["test_mse"], places=7)


if __name__ == "__main__":
    unittest.main()
