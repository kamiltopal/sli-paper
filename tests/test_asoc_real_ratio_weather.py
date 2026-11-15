"""CPU-only safety tests for the Weather ratio sensitivity add-on."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_real_ratio_weather import (
    check_active_time, main,
    EXPERIMENT_VERSION, KEY, METRICS, REFERENCE_VERSION,
    load_existing, load_reference, make_duplicate, parse_seeds, planned,
    ratio_values, report, result_key,
)

SHA = "abc" * 21 + "0"
N = 3570


def sample_reference(reps=(0, 1, 2), pools=(0, 1)):
    rows = []
    for rep in reps:
        for condition, ps in (("epoch_baseline", -1), ("step_baseline", -1),
                              ("duplicate_real", -1)) + tuple(("vae_augmented", p) for p in pools):
            steps = 560 if condition == "epoch_baseline" else 1120
            loss = dict(epoch_baseline=1.2, step_baseline=1.1,
                        duplicate_real=1.0, vae_augmented=0.6)[condition]
            if condition == "vae_augmented":
                loss += rep*0.02 + ps*0.02
            rows.append(dict(experiment_version=REFERENCE_VERSION, domain="weather",
                             data_sha256=SHA, regime=0.1, arch="itransformer",
                             input_mode="target", rep=rep, epochs=10, batch_size=64,
                             lr=0.001, vae_epochs=30, latent=16, hidden=128,
                             horizon_weight=0.5, condition=condition, pool_seed=ps,
                             optimizer_steps=steps, matched_steps=1120, original_steps=560,
                             best_step=1000 if steps == 1120 else 500, target_sd=2.0,
                             train_start=33198, train_end=36887,
                             test_mse=loss, test_mae=loss/2,
                             mse_raw=loss*4, mae_raw=loss))
    return pd.DataFrame(rows)


def sample_ratios(reps=(0, 1, 2), pools=(0, 1)):
    rows = []
    for ratio in (0.5, 2.0):
        for rep in reps:
            for cond, pool in (("duplicate_real", -1),) + tuple(("vae_augmented", ps) for ps in pools):
                loss = 1.05 if cond == "duplicate_real" else 0.7 + 0.07*(ratio-0.5) + .01*rep + .01*pool
                rows.append(dict(experiment_version=EXPERIMENT_VERSION,
                                 domain="weather", data_sha256=SHA, regime=.1,
                                 arch="itransformer", input_mode="target", ratio=ratio,
                                 rep=rep, condition=cond, pool_seed=pool,
                                 fixed_steps=1120, matched_steps=1120, optimizer_steps=1120,
                                 epochs=10, batch_size=64, lr=.001, vae_epochs=30,
                                 latent=16, hidden=128, horizon_weight=.5,
                                 lookback=96, horizon=24, pool_file_sha256="test",
                                 target_sd=2.0, best_step=1000,
                                 test_mse=loss, test_mae=loss/2,
                                 mse_raw=loss*4, mae_raw=loss))
    return pd.DataFrame(rows)


class WeatherRatioSafetyTests(unittest.TestCase):
    def test_seed_ratio_validation(self):
        self.assertEqual(parse_seeds("0,1,2"), (0,1,2))
        self.assertEqual(ratio_values("0.5,2.0"), (0.5,2.0))
        for bad in ("", "0,0", "-1", "a"):
            with self.assertRaises(ValueError):
                parse_seeds(bad)
        for bad in ("1.0", "0.5,0.5", "3.0"):
            with self.assertRaises(ValueError):
                ratio_values(bad)

    def test_ratio_matched_real_duplicates(self):
        x = np.arange(120, dtype=np.float32).reshape(10,12,1)
        y = np.arange(20, dtype=np.float32).reshape(10,2)
        for ratio, expected in ((.5,15), (1,20), (2,30)):
            rx, ry = make_duplicate((x,y),ratio)
            self.assertEqual(len(rx),expected)
            self.assertEqual(len(ry),expected)
            np.testing.assert_array_equal(rx[:10], x)
            np.testing.assert_array_equal(ry[:10], y)
            self.assertTrue(np.isin(ry[10:,0],y[:,0]).all())
            ax, ay = make_duplicate((x,y),ratio)
            np.testing.assert_array_equal(rx,ax)
            np.testing.assert_array_equal(ry,ay)

    def test_reference_budget_and_no_1x_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"base.csv"
            sample_reference().to_csv(path,index=False)
            ref, original, fixed = load_reference(path, SHA, (0,1,2), (0,1), N)
            self.assertEqual((original,fixed),(560,1120))
            self.assertEqual(len(ref),15)
            pending = planned((0,1,2),(0,1),(.5,2.0),pd.DataFrame(columns=KEY),SHA)
            self.assertEqual(len(pending),18)
            self.assertFalse(any(r[0]==1.0 for r in pending))
            old = sample_reference()
            old.loc[(old.condition=="step_baseline") & (old.rep==1),"optimizer_steps"] = 999
            old.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"budget mismatch"):
                load_reference(path,SHA,(0,1,2),(0,1),N)

    def test_report_paired_and_pool_hierarchy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/"base.csv"
            sample_reference().to_csv(base,index=False)
            ref, _, steps = load_reference(base,SHA,(0,1,2),(0,1),N)
            new = sample_ratios()
            paired, per_pool, summary = report(ref,new,(0,1,2),(0,1),(.5,2.0),steps,Path(temp)/"report")
            self.assertEqual(len(paired),18)  # 3 ratios x 3 model seeds x 2 pools
            self.assertEqual(len(per_pool),6)
            self.assertEqual(len(summary),3)
            self.assertEqual(summary.n_pairs.tolist(),[6,6,6])
            self.assertEqual(summary.beats_both_n_pairs.tolist(),[6,6,6])
            self.assertTrue((paired[paired.ratio==1.0].vae_vs_one_x_vae_test_mse_pct==0).all())
            self.assertTrue((Path(temp)/"report/ratio_summary.csv").is_file())
            self.assertTrue((Path(temp)/"report/ratio_incomplete.csv").is_file())

    def test_resume_and_digest_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"r.csv"
            frame = sample_ratios(reps=(0,),pools=(0,))
            frame.to_csv(out,index=False)
            previous = load_existing(out,SHA)
            self.assertEqual(len(planned((0,),(0,),(.5,2.0),previous,SHA)),0)
            with self.assertRaisesRegex(ValueError,"digest changed"):
                load_existing(out,"different")
            pd.concat([frame,frame.iloc[[0]]]).to_csv(out,index=False)
            with self.assertRaisesRegex(ValueError,"duplicate experiment IDs"):
                load_existing(out,SHA)

    def test_active_timestamps_ignore_disjoint_early_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp)/"weather.csv"
            dates = pd.date_range("2020-01-01", periods=300, freq="10min")
            dates = dates.to_series().reset_index(drop=True)
            # One duplicate and one gap, both earlier than active start 150.
            dates.iloc[20] = dates.iloc[19]
            dates.iloc[50:150] += pd.Timedelta(minutes=10)
            # Active timestamps must be regular, but the edit above only creates
            # a gap and a reverse boundary outside the active span.
            pd.DataFrame({"date": dates.astype(str)}).to_csv(filename,index=False)
            first, last = check_active_time(filename, 300, 150)
            self.assertTrue(first.startswith("2020-01-02"))
            self.assertTrue(last.startswith("2020-01-03"))

    def test_realistic_no_gpu_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base/"weather.npy"
            X = np.tile(np.arange(52696,dtype=np.float32)[:,None]/100000.0,(1,3))
            np.save(data,X)
            from scripts.asoc_real_ratio_weather import sha256_file
            digest = sha256_file(data)
            csv = base/"weather.csv"
            dates = pd.date_range("2020-01-01", periods=len(X), freq="10min")
            pd.DataFrame({"date":dates.astype(str)}).to_csv(csv,index=False)
            ref = base/"reference.csv"
            rows = sample_reference(reps=(0,),pools=(0,))
            rows["data_sha256"] = digest
            rows.to_csv(ref,index=False)
            from contextlib import redirect_stdout
            import io
            with redirect_stdout(io.StringIO()) as out:
                main(["--reps","0","--pool-seeds","0","--dry-run",
                      "--data",str(data),"--raw-csv",str(csv),
                      "--reference",str(ref),"--out",str(base/"new.csv")])
            self.assertIn("4 pending",out.getvalue())
            self.assertFalse((base/"new.csv").exists())


if __name__ == "__main__":
    unittest.main()
