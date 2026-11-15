"""Phase 12 fixed-simulation, leakage guards, resume, pairing, hierarchy tests."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.asoc_nonfourier import (
    CFG, VERSION, array_sha256, ensure_series, expected_rows, fixed_row, key_of,
    parse_csv_choices, parse_ids, read_runs, run_group, series_seed, simulate,
    _combine, simple_references, main,
)
from scripts.asoc_nonfourier_report import audit_and_pair, aggregate
from src.data import prepare


class NonFourierTests(unittest.TestCase):
    def fixture_row(self, c, rep=0, ps=-1, sid=0, family="nonlinear_ar"):
        kw = fixed_row(family, sid, "sha0" if sid == 0 else "sha1", 0.1,
                       "itransformer", rep, ps, c)
        mse = {"epoch_baseline": 1.2, "step_baseline": 1.1,
               "duplicate_real": 1.0, "vae_augmented": .8}[c]
        steps = 120 if c == "epoch_baseline" else 230
        return {**kw, "test_mse": mse, "test_mae": mse/2,
                "mse_raw": mse*4, "mae_raw": mse,
                "target_sd": 2.0, "optimizer_steps": steps,
                "original_steps": 120, "matched_steps": 230,
                "best_step": steps-10, "real_windows": 721,
                "train_start": 7560, "train_end": 8400,
                "persistence_mse": 1.8, "trainmean_mse": 2.0,
                "direct_ar2_mse": 1.1, "generator_seconds": 0.4}

    def fixture(self, *, ids=(0, 1, 2), reps=(0, 1), pools=(0, 1), family="nonlinear_ar"):
        rows=[]
        for sid in ids:
            for rep in reps:
                for cond in ("epoch_baseline", "step_baseline", "duplicate_real"):
                    rows.append(self.fixture_row(cond, rep=rep, sid=sid, family=family))
                for ps in pools:
                    rows.append(self.fixture_row("vae_augmented", rep=rep, ps=ps,
                                                 sid=sid, family=family))
        return pd.DataFrame(rows)

    def test_deterministic_finite_distinct_stochastic_series(self):
        for family in ("nonlinear_ar", "markov_ar"):
            a = simulate(family, 0, 3000)
            self.assertEqual(a.shape, (3000, 1))
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, simulate(family, 0, 3000))
            self.assertNotEqual(array_sha256(a), array_sha256(simulate(family, 1, 3000)))
        self.assertNotEqual(array_sha256(simulate("nonlinear_ar", 0, 3000)),
                            array_sha256(simulate("markov_ar", 0, 3000)))

    def test_generator_params_frozen_and_bad_inputs(self):
        with self.assertRaises(ValueError): simulate("wrong", 0)
        with self.assertRaises(ValueError): simulate("nonlinear_ar", 3)
        with self.assertRaises(ValueError): simulate("nonlinear_ar", 0, 1500)
        self.assertNotEqual(series_seed("nonlinear_ar", 0), series_seed("markov_ar", 0))
        self.assertEqual(parse_ids("0,2"), (0,2))
        for bad in ("0,0", "3", "", "x", "-1"):
            with self.assertRaises(ValueError): parse_ids(bad)
        self.assertEqual(parse_csv_choices("nonlinear_ar,markov_ar",
                         ("nonlinear_ar", "markov_ar")), ("nonlinear_ar", "markov_ar"))
        with self.assertRaises(ValueError): parse_csv_choices("nonlinear_ar,nonlinear_ar",
                                             ("nonlinear_ar", "markov_ar"))

    def test_stable_saved_arrays_read_only_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            arr, digest = ensure_series(path, "nonlinear_ar", 0, write=False)
            self.assertFalse(list(path.glob("*.npy")))
            arr2, digest2 = ensure_series(path, "nonlinear_ar", 0, write=True)
            self.assertEqual(digest, digest2)
            np.testing.assert_array_equal(arr, arr2)
            path_file = path / "nonlinear_ar_00.npy"
            modified = np.load(path_file)
            modified[0] += 2
            np.save(path_file, modified)
            with self.assertRaisesRegex(ValueError, "differs"):
                ensure_series(path, "nonlinear_ar", 0, write=False)

    def test_train_only_standardization_and_window_boundaries(self):
        a = simulate("markov_ar", 0, 3000)
        d = prepare(a, .1, CFG)
        self.assertEqual(len(d["train"][0]), 91)  # t1=2100, 210 selected rows, 91 windows
        self.assertTrue(np.isfinite(d["test"][1]).all())
        self.assertAlmostEqual(float(d["train"][0].shape[2]), 1.)
        b = a.copy(); b[2100:] += 999
        bd = prepare(b, .1, CFG)
        np.testing.assert_array_equal(d["train"][0], bd["train"][0])
        np.testing.assert_array_equal(d["train"][1], bd["train"][1])
        np.testing.assert_array_equal(d["mu"], bd["mu"])
        np.testing.assert_array_equal(d["sd"], bd["sd"])

    def test_expected_group_counts_no_control_repetition_for_pool(self):
        rows = list(expected_rows(family="nonlinear_ar", series_id=0, digest="abc",
                                   regime=.1, arch="itransformer", reps=(0,1,2), pool_seeds=(0,1)))
        self.assertEqual(len(rows),15)
        self.assertEqual(sum(r["condition"] == "vae_augmented" for r in rows),6)
        self.assertEqual(len({key_of(r) for r in rows}),15)

    def test_resume_and_duplication_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.csv"
            raw=self.fixture(ids=(0,), reps=(0,), pools=(0,))
            raw.to_csv(path,index=False)
            keys, loaded=read_runs(path)
            self.assertEqual(len(keys),4)
            self.assertEqual(len(loaded),4)
            raw.iloc[[0]].to_csv(path,mode="a",header=False,index=False)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                read_runs(path)

    def test_synthetic_shapes_and_finiteness(self):
        a=(np.zeros((2,3,1),dtype="float32"),np.zeros((2,1),dtype="float32"))
        good=_combine(a,a)
        self.assertEqual(good[0].shape,(4,3,1))
        b=(np.zeros((3,3,1),dtype="float32"),np.zeros((3,1),dtype="float32"))
        with self.assertRaises(ValueError): _combine(a,b)
        c=(np.full((2,3,1),np.nan,dtype="float32"),np.zeros((2,1),dtype="float32"))
        with self.assertRaises(ValueError): _combine(a,c)

    def test_simple_references_train_only_and_finite(self):
        d=prepare(simulate("nonlinear_ar",0,3000),.1,CFG)
        ref=simple_references(d["train"],d["test"])
        self.assertEqual(set(ref), {"persistence_mse","trainmean_mse","direct_ar2_mse"})
        self.assertTrue(all(np.isfinite(x) and x>0 for x in ref.values()))

    def test_paired_and_series_unit_not_seed_unit(self):
        p, incomplete=audit_and_pair(self.fixture())
        self.assertEqual(len(p), 12)
        self.assertTrue(incomplete.empty)
        self.assertEqual(int(p.vae_beats_both.sum()),12)
        self.assertAlmostEqual(float(p.iloc[0].vae_vs_duplicate_real_test_mse_pct),-20.)
        pools, series, family=aggregate(p)
        self.assertEqual(len(pools),6)
        self.assertEqual(len(series),3)
        self.assertEqual(len(family),1)
        self.assertEqual(int(family.n_independent_series.iloc[0]),3)
        self.assertEqual(int(family.n_pairs.iloc[0]),12)

    def test_incomplete_pair_report_and_budget_guard(self):
        df=self.fixture(ids=(0,), reps=(0,), pools=(0,))
        out, missing=audit_and_pair(df[df.condition!="step_baseline"])
        self.assertTrue(out.empty)
        self.assertEqual(len(missing),1)
        bad=df.copy()
        bad.loc[bad.condition=="vae_augmented","optimizer_steps"]=999
        with self.assertRaisesRegex(ValueError,"Unmatched"):
            audit_and_pair(bad)

    def test_raw_scale_and_partition_guard(self):
        df=self.fixture(ids=(0,), reps=(0,), pools=(0,))
        bad=df.copy()
        bad.loc[bad.condition=="vae_augmented","mse_raw"] += .5
        with self.assertRaisesRegex(ValueError,"scale"):
            audit_and_pair(bad)
        bad=df.copy()
        bad.loc[bad.condition=="vae_augmented","train_end"] += 1
        with self.assertRaisesRegex(ValueError,"partition"):
            audit_and_pair(bad)

    def test_integrated_fake_fit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);out=tmp/"runs.csv";arr=simulate("nonlinear_ar",0,3000)
            digest=array_sha256(arr)
            counter=[]
            def fake_fit(**kwargs):
                steps=kwargs["max_steps"];counter.append(steps)
                return dict(test_mse=1+len(counter)/10,test_mae=.7,val_mse=.5,
                            best_step=steps-1, optimizer_steps=steps,
                            samples_seen=steps*64,train_windows=len(kwargs["train_pair"][0]),
                            validation_checkpoints=1,validation_curve="[]",train_seconds=.01)
            def fake_gen(segment,cfg,ratio,seed,epochs,latent):
                from src.data import make_windows
                return make_windows(segment,0,len(segment),cfg)
            kw=dict(family="nonlinear_ar",series_id=0,digest=digest,array=arr,regime=.1,
                    arch="itransformer",reps=(0,1),pool_seeds=(0,1),epochs=1,
                    batch_size=64,lr=.001,vae_epochs=1,latent=16,
                    device=torch.device("cpu"),out=out,done=set())
            with patch("scripts.asoc_nonfourier.fit_condition",side_effect=fake_fit), \
                 patch("src.generators.gen_vae",side_effect=fake_gen):
                with redirect_stdout(io.StringIO()):
                    run_group(**kw)
                self.assertEqual(len(pd.read_csv(out)),10)
                self.assertEqual(len(counter),10)
                kw["done"], _=read_runs(out)
                with redirect_stdout(io.StringIO()):
                    run_group(**kw)
                self.assertEqual(len(counter),10)
                pair, incomplete = audit_and_pair(pd.read_csv(out))
                self.assertEqual(len(pair),4)
                self.assertTrue(incomplete.empty)

    def test_dry_run_no_data_or_csv_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)
            with redirect_stdout(io.StringIO()) as stream:
                main(["--families","nonlinear_ar","--ids","0","--regimes","0.10",
                      "--reps","0","--pool-seeds","0","--dry-run",
                      "--data-dir",str(t/"data"),"--out",str(t/"r.csv")])
            self.assertIn("4 new runs",stream.getvalue())
            self.assertFalse((t/"r.csv").exists())
            self.assertFalse((t/"data").exists())


if __name__ == "__main__":
    unittest.main()
