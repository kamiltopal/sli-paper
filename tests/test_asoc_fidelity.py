"""Phase13 R1.7: diagnostic mathematics, leakage/provenance, report pairing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.asoc_fidelity import (compare_windows, diagnostic_metrics,
                                   load_training_group, parse_regimes,
                                   select_rows, read_previous,
                                   GROUP_FIELDS, VERSION, target_windows,
                                   digest_file, run)
from scripts.asoc_fidelity_report import combine, summarize, exploratory_correlations
from scripts.asoc_nonfourier import simulate, array_sha256
from src.data import WindowConfig, prepare


class FidelityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.windows = rng.normal(size=(20, 32))
        self.dirs = rng.normal(size=(12, 32))

    def test_identical_windows_zero_distance(self):
        r = compare_windows(self.windows, self.windows, self.dirs, lags=(1, 2, 6))
        for key, val in r.items():
            self.assertAlmostEqual(val, 0.0, places=10, msg=key)

    def test_translation_increases_wasserstein(self):
        a = self.windows
        b = a + 2
        r = compare_windows(a, b, self.dirs, lags=(1, 2, 6))
        self.assertAlmostEqual(r["marginal_w1"], 2.0, places=8)
        self.assertGreater(r["sliced_w1"], 0.01)
        self.assertAlmostEqual(r["spectral_js"], 0.0, places=10)

    def test_shuffled_order_preserves_marginal_not_structure(self):
        rng = np.random.default_rng(6)
        x = np.array([np.sin(np.arange(48) / 7) for _ in range(64)])
        y = np.array([r[rng.permutation(48)] for r in x])
        directions = rng.normal(size=(16, 48))
        r = compare_windows(x, y, directions, lags=(1, 2, 12))
        self.assertAlmostEqual(r["marginal_w1"], 0.0, places=9)
        self.assertGreater(r["acf_mae"], 0.1)
        self.assertGreater(r["spectral_tv"], 0.05)

    def test_diagnostics_seeded_and_three_references(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(120, 10, 1)).astype("float32")
        y = rng.normal(size=(120, 4)).astype("float32")
        pair = (X, y)
        r = diagnostic_metrics(pair, pair, sample_windows=30, projections=8,
                               seed=123, lags=(1, 2, 4))
        r2 = diagnostic_metrics(pair, pair, sample_windows=30, projections=8,
                               seed=123, lags=(1, 2, 4))
        self.assertEqual(r, r2)
        self.assertEqual(r["sample_windows_per_group"], 30)
        self.assertIn("shuffled_spectral_js", r)
        self.assertIn("real_real_sliced_w1", r)

    def test_invalid_projection_lag_and_nonfinite(self):
        with self.assertRaises(ValueError):
            compare_windows(self.windows, self.windows, self.dirs, lags=(32,))
        with self.assertRaises(ValueError):
            compare_windows(self.windows, np.full_like(self.windows, np.nan), self.dirs)
        with self.assertRaises(ValueError):
            compare_windows(self.windows, self.windows[:9], self.dirs)

    def test_parse_regimes(self):
        self.assertEqual(parse_regimes("0.10,1.00"), (0.1, 1.0))
        for invalid in ("0.10,0.1", "0.12", "0.10,"):
            with self.assertRaises(ValueError):
                parse_regimes(invalid)

    def _fake_run_rows(self, family="nonlinear_ar", sid=0, pool=0, rep=0):
        array = simulate(family, sid)
        d = prepare(array, 0.1, WindowConfig(96, 24))
        digest = array_sha256(array)
        meta = dict(experiment_version="phase12_nonfourier_fixed_v1",
                    family=family, series_id=sid, data_sha256=digest,
                    regime=.1, arch="itransformer", rep=rep, pool_seed=pool,
                    condition="vae_augmented", vae_epochs=30, latent=16,
                    lookback=96, horizon=24, test_mse=.8,
                    epochs=10, batch_size=64, lr=.001,
                    train_start=d["train_start"], train_end=int(len(array)*.7),
                    real_windows=len(d["train"][0]), target_sd=float(d["sd"][0,0]),
                    optimizer_steps=230)
        return meta, array

    def test_select_once_for_model_replications(self):
        a, _ = self._fake_run_rows(rep=0)
        b = dict(a, rep=1)
        groups = select_rows(pd.DataFrame([a, b]), "nonfourier",
                 families=("nonlinear_ar",), ids=(0,), domains=(), regimes=(.1,))
        self.assertEqual(len(groups), 1)

    def test_reject_mismatched_sha(self):
        a, _ = self._fake_run_rows()
        a["data_sha256"] = "wrong"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_training_group(a, "nonfourier", Path(tmp))

    def test_reject_mismatched_partition(self):
        a, _ = self._fake_run_rows()
        a["train_start"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "partition"):
                load_training_group(a, "nonfourier", Path(tmp))

    def test_read_existing_duplicate_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.csv"
            r, _ = self._fake_run_rows()
            r = {"source": "nonfourier", "diagnostic_version": VERSION,
                 "experiment_version": r["experiment_version"],
                 **{k: r[k] for k in GROUP_FIELDS["nonfourier"]},
                 "sample_windows": 512, "projections": 32}
            pd.DataFrame([r, r]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_previous(path, "nonfourier")

    def test_combine_reports_only_matching_controls(self):
        vae, _ = self._fake_run_rows()
        step = dict(vae, condition="step_baseline", pool_seed=-1, test_mse=1.)
        dup = dict(vae, condition="duplicate_real", pool_seed=-1, test_mse=1.1)
        diag = {"diagnostic_version": VERSION,
                **{k: vae[k] for k in GROUP_FIELDS["nonfourier"]},
                **{f"vae_{m}": 0.1 for m in ("marginal_w1", "sliced_w1",
                                                    "spectral_js", "spectral_tv",
                                                    "acf_mae", "tail_p2_absdiff")}}
        result = combine(pd.DataFrame([vae, step, dup]), pd.DataFrame([diag]), "nonfourier")
        self.assertAlmostEqual(result.iloc[0].vae_vs_step_baseline_mse_pct, -20.)
        self.assertTrue(result.iloc[0].vae_beats_both)
        dup["optimizer_steps"] = 229
        with self.assertRaisesRegex(ValueError, "Optimizer"):
            combine(pd.DataFrame([vae, step, dup]), pd.DataFrame([diag]), "nonfourier")

    def test_report_aggregates_by_independent_series(self):
        r = pd.DataFrame([dict(family="nonlinear_ar", series_id=0, regime=.1,
                   pool_seed=ps, vae_beats_both=(ps == 0),
                   vae_vs_duplicate_real_mse_pct=pct,
                   vae_vs_step_baseline_mse_pct=pct,
                   **{f"vae_{m}": .1 + ps for m in ("marginal_w1", "sliced_w1", "spectral_js",
                                                       "spectral_tv", "acf_mae", "tail_p2_absdiff")})
                   for ps, pct in [(0,-4), (1,2)]] +
                   [dict(family="nonlinear_ar", series_id=1, regime=.1,
                   pool_seed=0, vae_beats_both=False,
                   vae_vs_duplicate_real_mse_pct=1.,
                   vae_vs_step_baseline_mse_pct=1.,
                   **{f"vae_{m}": .5 for m in ("marginal_w1", "sliced_w1", "spectral_js",
                                              "spectral_tv", "acf_mae", "tail_p2_absdiff")})])
        x = summarize(r, "nonfourier")
        self.assertEqual(int(x.iloc[0].n_independent_series), 2)
        self.assertEqual(int(x.iloc[0].n_pool_rows), 3)
        self.assertAlmostEqual(x.iloc[0].fraction_series_beating_both, 0.5)
        cors = exploratory_correlations(r, "nonfourier")
        self.assertTrue(cors.spearman_rho.isna().all())

    def test_real_domain_source_validates_file_hash_and_training_slice(self):
        rng = np.random.default_rng(13)
        a = rng.normal(size=(1800, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            datadir = Path(tmp)
            path = datadir / "weather.npy"
            np.save(path, a)
            hash_before = digest_file(path)
            prep = prepare(a, regime=.1, cfg=WindowConfig(96, 24))
            row = dict(domain="weather", regime=.1, arch="itransformer",
                       input_mode="target", pool_seed=0, data_sha256=hash_before,
                       vae_epochs=30, latent=16, lookback=96, horizon=24,
                       train_start=prep["train_start"], train_end=int(.7 * len(a)),
                       real_windows=len(prep["train"][0]),
                       target_sd=float(prep["sd"][0, 0]))
            real, segment, cfg = load_training_group(row, "real", datadir)
            self.assertEqual(len(real[0]), row["real_windows"])
            self.assertEqual(segment.shape[1], 3)
            self.assertEqual(digest_file(path), hash_before)
            row["data_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_training_group(row, "real", datadir)

    def test_real_report_pairs_by_rep_and_parameters(self):
        r, _ = self._fake_run_rows()
        meta = dict(domain="weather", regime=.1, arch="itransformer",
                    input_mode="target", data_sha256="h1",
                    vae_epochs=30, latent=16, lookback=96, horizon=24,
                    epochs=10, batch_size=64, lr=.001,
                    rep=0, train_start=10, train_end=70,
                    real_windows=20, target_sd=1., optimizer_steps=20)
        vae = dict(meta, condition="vae_augmented", pool_seed=0, test_mse=.8)
        step = dict(meta, condition="step_baseline", pool_seed=-1, test_mse=1.)
        dup = dict(meta, condition="duplicate_real", pool_seed=-1, test_mse=1.1)
        diag = {"diagnostic_version": VERSION,
                **{k: vae[k] for k in GROUP_FIELDS["real"]},
                **{f"vae_{m}": .1 for m in ("marginal_w1", "sliced_w1",
                                               "spectral_js", "spectral_tv",
                                               "acf_mae", "tail_p2_absdiff")}}
        out = combine(pd.DataFrame([vae, step, dup]), pd.DataFrame([diag]), "real")
        self.assertEqual(len(out), 1)
        self.assertTrue(out.iloc[0].vae_beats_both)
        summary = summarize(out, "real")
        self.assertEqual(int(summary.iloc[0].n_pool_rows), 1)

    def test_dry_run_does_not_write(self):
        a, _ = self._fake_run_rows()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "runs.csv"
            outdir = Path(tmp) / "out"
            pd.DataFrame([a]).to_csv(input_path, index=False)
            run(["--source", "nonfourier", "--in", str(input_path),
                 "--outdir", str(outdir), "--dry-run"])
            self.assertFalse(outdir.exists())


if __name__ == "__main__":
    unittest.main()
