"""Phase 7 protocol tests: keys, strict step matching, raw-scale metrics, no leakage."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.asoc_real_controls import (KEY, VERSION, build_key, parse_ints,
                                       read_completed, run_group)
from scripts.asoc_real_matrix import expected_rows
from scripts.asoc_real_report import paired_table, pool_table, summary_table


class RealPhase7Tests(unittest.TestCase):
    def row(self, *, condition, pool_seed, rep=0, mse=None, steps=None):
        common = dict(experiment_version=VERSION, domain="electricity",
                      data_sha256="abc123", regime=0.25, arch="dlinear",
                      input_mode="target", rep=rep, epochs=10,
                      batch_size=64, lr=0.001, vae_epochs=30, latent=16,
                      hidden=128, horizon_weight=0.5)
        if mse is None:
            mse = {"epoch_baseline": 1.3, "step_baseline": 1.0,
                   "duplicate_real": 1.1, "vae_augmented": 0.9}[condition]
        if steps is None:
            steps = 50 if condition == "epoch_baseline" else 100
        return {**common, "condition": condition, "pool_seed": pool_seed,
                "test_mse": mse, "test_mae": mse/2, "mse_raw": mse * 9,
                "mae_raw": mse/2 * 3, "optimizer_steps": steps,
                "matched_steps": 100, "original_steps": 50,
                "best_step": steps-5, "target_sd": 3.0,
                "real_windows": 130, "train_start": 300, "train_end": 600}

    def complete(self, pool_seeds=(0,1), reps=(0,1)):
        rows = []
        for rep in reps:
            for c in ("epoch_baseline", "step_baseline", "duplicate_real"):
                rows.append(self.row(condition=c, pool_seed=-1, rep=rep))
        for ps in pool_seeds:
            for rep in reps:
                rows.append(self.row(condition="vae_augmented", pool_seed=ps, rep=rep,
                                     mse=0.9 + ps*0.1 + rep*0.01))
        return pd.DataFrame(rows)

    def test_seed_parse_and_key(self):
        self.assertEqual(parse_ints("0,2"), (0,2))
        for bad in ("", "0,0", "0,-1", "x"):
            with self.assertRaises(ValueError):
                parse_ints(bad)
        row = self.row(condition="vae_augmented", pool_seed=3)
        self.assertEqual(len(build_key(row, "vae_augmented", 0, 3)), len(KEY))

    def test_matrix_controls_only_once_per_rep(self):
        rows = list(expected_rows(domain="electricity", regime=0.25,
            arch="dlinear", input_mode="target", reps=(0,1), pool_seeds=(0,1,2),
            epochs=10, batch_size=64, lr=0.001, vae_epochs=30, latent=16,
            hidden=128, horizon_weight=0.5))
        self.assertEqual(len(rows), 12)  # 2*(3 controls + 3 VAE pools)
        self.assertEqual(sum(x["condition"] != "vae_augmented" for x in rows),6)
        self.assertEqual(sum(x["condition"] == "vae_augmented" for x in rows),6)

    def test_report_pool_aggregation_and_absolute_scale(self):
        df = self.complete()
        paired, incomplete = paired_table(df)
        self.assertEqual(len(paired),4)
        self.assertTrue(incomplete.empty)
        self.assertAlmostEqual(float(paired.iloc[0]["vae_vs_duplicate_real_mse_raw_abs"]),
                               (0.9-1.1)*9)
        self.assertAlmostEqual(float(paired.iloc[0]["vae_vs_step_baseline_test_mse_pct"]),-10)
        pools = pool_table(paired)
        self.assertEqual(len(pools),2)
        summ = summary_table(pools)
        self.assertEqual(int(summ.n_pool_seeds.iloc[0]),2)
        self.assertEqual(int(summ.n_training_reps_min.iloc[0]),2)
        self.assertEqual(int(summ.n_pairs.iloc[0]),4)

    def test_missing_controls_are_reported_not_assumed(self):
        df = self.complete(pool_seeds=(0,), reps=(0,))
        df = df[df.condition != "step_baseline"]
        with self.assertRaises(ValueError):
            paired_table(df)  # No complete comparison

    def test_detect_bad_steps_scale_and_duplicate_ids(self):
        df = self.complete(pool_seeds=(0,), reps=(0,))
        bad = df.copy()
        bad.loc[bad.condition == "duplicate_real", "optimizer_steps"] = 99
        with self.assertRaisesRegex(ValueError,"Unmatched"):
            paired_table(bad)
        bad = df.copy()
        bad.loc[bad.condition == "vae_augmented", "mse_raw"] += 1
        with self.assertRaisesRegex(ValueError,"scale"):
            paired_table(bad)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            paired_table(pd.concat([df, df.iloc[[0]]], ignore_index=True))

    def test_read_completed_and_resumption_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "runs.csv"
            df = self.complete(pool_seeds=(0,), reps=(0,))
            df.to_csv(path,index=False)
            keys, loaded = read_completed(path)
            self.assertEqual(len(keys),4)
            self.assertEqual(len(loaded),4)
            df.iloc[[0]].to_csv(path,mode="a",header=False,index=False)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                read_completed(path)

    def test_reject_full_span_covariates_under_scarcity(self):
        with self.assertRaisesRegex(ValueError, "full 70% train span"):
            run_group(domain="electricity",regime=.25,arch="itransformer",
                      input_mode="all",reps=(0,),pool_seeds=(0,),
                      epochs=1,vae_epochs=1,batch_size=64,lr=.001,
                      latent=16,hidden=128,horizon_weight=.5,
                      gpu_index=0,allow_cpu=True,out=Path("unused.csv"))

    def test_cpu_integration_fake_training_real_window_pipeline(self):
        """Exercise real prepare(), seed isolation and generator call integration.

        Training is patched to deterministic pseudo-metrics, to keep this
        suite fast and independent of a CUDA device. Separate manual smoke
        checks use the real neural trainer below in the verification script.
        """
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); src=d/"real";src.mkdir();out=d/"out.csv"
            rng=np.random.default_rng(3)
            t=np.arange(1600)
            x=np.stack([np.sin(t/18)+0.05*rng.standard_normal(len(t)),
                        np.cos(t/25)],axis=1).astype("float32")
            np.save(src/"electricity.npy",x)
            fitted=[]
            def fake_fit(**kwargs):
                fitted.append(kwargs["max_steps"])
                return dict(test_mse=1.0+len(fitted)/10,
                            test_mae=.5, val_mse=.4, best_step=kwargs["max_steps"],
                            optimizer_steps=kwargs["max_steps"],
                            samples_seen=64*kwargs["max_steps"],
                            train_windows=len(kwargs["train_pair"][0]),
                            validation_checkpoints=1,validation_curve="[]",
                            train_seconds=.01)
            def fake_gen(segment,cfg,ratio,seed,epochs,latent):
                from src.data import make_windows
                return make_windows(segment,0,len(segment),cfg)
            kw=dict(domain="electricity",regime=0.25,arch="dlinear",
                    input_mode="target",reps=(0,1),pool_seeds=(0,1),
                    epochs=1,vae_epochs=1,batch_size=64,lr=.001,
                    latent=16,hidden=128,horizon_weight=.5,gpu_index=0,
                    allow_cpu=True,out=out,data_dir=src)
            with patch("scripts.asoc_real_controls.select_device",return_value=torch.device("cpu")), \
                 patch("scripts.asoc_real_controls.fit_condition",side_effect=fake_fit), \
                 patch("src.generators.gen_vae",side_effect=fake_gen):
                run_group(**kw)
                self.assertEqual(len(pd.read_csv(out)),10)
                self.assertEqual(len(fitted),10)
                run_group(**kw)
                self.assertEqual(len(fitted),10) # resume must skip all
                paired, missing = paired_table(pd.read_csv(out))
                self.assertEqual(len(paired),4)
                self.assertTrue(missing.empty)
                np.save(src/"electricity.npy",x+1)
                with self.assertRaisesRegex(ValueError,"Data changed"):
                    run_group(**kw)


if __name__ == "__main__":
    unittest.main()
