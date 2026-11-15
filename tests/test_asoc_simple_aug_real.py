"""Small CPU-only integrity tests; no full GPU training and no original edits."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asoc_simple_aug_real import (
    REF_VERSION, VERSION, KEY, REF_KEY, METRICS,
    augmented_training_pair, audit_timestamps, parse_methods, parse_seeds,
    validate_reference, pending_runs, load_new, add_result, summarize, main,
)

SHA = "a" * 64


def reference(domain="weather", reps=(0,1,2), seeds=(0,1), n=3570, start=33198, end=36887):
    original = 10 * int(np.ceil(n / 64))
    matched = 10 * int(np.ceil(2*n/64))
    rows = []
    for rep in reps:
        for condition, ps in [("epoch_baseline", -1), ("step_baseline", -1),
                              ("duplicate_real", -1)] + [("vae_augmented", s) for s in seeds]:
            loss = {"epoch_baseline":1.1,"step_baseline":1.0,"duplicate_real":1.05,"vae_augmented":.6}[condition]
            if condition=="vae_augmented":loss += .02*rep+.01*ps
            rows.append(dict(experiment_version=REF_VERSION, domain=domain, data_sha256=SHA,
                regime=.1, arch="itransformer", input_mode="target", rep=rep,
                epochs=10, batch_size=64, lr=.001, vae_epochs=30,
                latent=16, hidden=128, horizon_weight=.5,
                condition=condition, pool_seed=ps,
                optimizer_steps=original if condition=="epoch_baseline" else matched,
                original_steps=original, matched_steps=matched,
                target_sd=2., train_start=start, train_end=end, best_step=matched-5,
                test_mse=loss, test_mae=loss/2, mse_raw=loss*4, mae_raw=loss))
    return pd.DataFrame(rows)


def new_record(method="jitter", rep=0, aug_seed=0, domain="weather", loss=.62, matched=1120):
    return dict(experiment_version=VERSION, domain=domain, data_sha256=SHA,
        regime=.1, arch="itransformer", input_mode="target", method=method,
        rep=rep, aug_seed=aug_seed, epochs=10, batch_size=64, lr=.001,
        vae_epochs=30, latent=16, hidden=128, horizon_weight=.5,
        jitter_sigma=.03, scale_sigma=.1, lookback=96, horizon=24,
        original_steps=560, matched_steps=matched, optimizer_steps=matched,
        target_sd=2., best_step=matched-100,
        test_mse=loss, test_mae=loss/2, mse_raw=loss*4, mae_raw=loss,
        pool_file_sha256="digest")


class SimpleAugSafetyTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_seeds("0,1,2"),(0,1,2))
        self.assertEqual(parse_methods("jitter,scaling"),("jitter","scaling"))
        for s in ("", "0,0", "-1", "bad"):
            with self.assertRaises(ValueError):parse_seeds(s)
        for s in ("", "jitter,jitter", "warp"):
            with self.assertRaises(ValueError):parse_methods(s)

    def test_jitter_scaling_shared_input_and_label_perturbation(self):
        x = np.full((15,96,1),2.,np.float32)
        y = np.full((15,24),2.,np.float32)
        jx,jy = augmented_training_pair((x,y),"jitter",0)
        sx,sy = augmented_training_pair((x,y),"scaling",0)
        self.assertEqual(jx.shape,x.shape)
        self.assertEqual(jy.shape,y.shape)
        self.assertTrue(np.any(jy != y))
        self.assertTrue(np.any(jx != x))
        np.testing.assert_allclose(sx[:,0,0],sy[:,0])
        np.testing.assert_allclose(sx[:,0,0],sx[:,-1,0])
        np.testing.assert_array_equal(x,np.full_like(x,2.))
        np.testing.assert_array_equal(y,np.full_like(y,2.))
        jx2,jy2 = augmented_training_pair((x,y),"jitter",0)
        np.testing.assert_array_equal(jx,jx2)
        np.testing.assert_array_equal(jy,jy2)

    def test_reference_prevents_budget_errors_and_missing_vae(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"ref.csv"
            df=reference(reps=(0,),seeds=(0,))
            df.to_csv(path,index=False)
            got,old,matched = validate_reference(path,"weather",SHA,(0,),(0,),3570,33198,36887)
            self.assertEqual((old,matched),(560,1120))
            self.assertEqual(len(got),4)
            df.loc[df.condition=="vae_augmented","optimizer_steps"] = 1200
            df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"budget/split mismatch"):
                validate_reference(path,"weather",SHA,(0,),(0,),3570,33198,36887)
            df=df[df.condition!="vae_augmented"]
            df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"Missing Phase7"):
                validate_reference(path,"weather",SHA,(0,),(0,),3570,33198,36887)

    def test_resume_and_digests(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"runs.csv"
            blank=pd.DataFrame(columns=KEY)
            todo=pending_runs("weather",SHA,("jitter","scaling"),(0,1,2),(0,1),blank)
            self.assertEqual(len(todo),12)
            row=new_record()
            saved=add_result(p,blank,row)
            self.assertEqual(len(pending_runs("weather",SHA,("jitter",),(0,),(0,),saved)),0)
            self.assertEqual(len(load_new(p,"weather",SHA,1120)),1)
            with self.assertRaisesRegex(ValueError,"Data hash changed"):
                load_new(p,"weather","b"*64,1120)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                add_result(p,saved,row)

    def test_active_time_disjoint_anomalies(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"weather.csv"
            dates=pd.date_range("2020-01-01",periods=3000,freq="10min").to_series().reset_index(drop=True)
            dates.iloc[100] = dates.iloc[99] # old duplicate
            dates.iloc[200:201] = dates.iloc[200]+pd.Timedelta(minutes=10) # old gap
            pd.DataFrame({"date":dates.astype(str)}).to_csv(path,index=False)
            self.assertEqual(len(audit_timestamps(path,3000,400,"weather")),2)
            with self.assertRaisesRegex(ValueError,"Irregular weather"):
                audit_timestamps(path,3000,99,"weather")

    def test_hierarchy_report_and_comparators(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t)/"base.csv"
            reference(reps=(0,1,2),seeds=(0,1)).to_csv(base,index=False)
            refs,_,matched=validate_reference(base,"weather",SHA,(0,1,2),(0,1),3570,33198,36887)
            rows=[new_record(method=m,rep=rep,aug_seed=seed,
                             loss=.62+.02*rep+.015*seed,matched=matched)
                  for m in ("jitter","scaling") for seed in (0,1) for rep in (0,1,2)]
            paired,pools,summary=summarize(refs,pd.DataFrame(rows),"weather",
                ("jitter","scaling"),(0,1,2),(0,1),matched,Path(t)/"report")
            self.assertEqual((len(paired),len(pools),len(summary)),(12,4,2))
            self.assertTrue((summary.n_pairs==6).all())
            self.assertTrue((paired.simple_vs_step_test_mse_pct<0).all())
            self.assertTrue((Path(t)/"report/weather_summary.csv").is_file())

    def test_partial_report_not_claimed_complete(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t)/"base.csv"
            reference(reps=(0,),seeds=(0,)).to_csv(base,index=False)
            refs,_,matched=validate_reference(base,"weather",SHA,(0,),(0,),3570,33198,36887)
            new=pd.DataFrame([new_record()])
            paired,_,summary=summarize(refs,new,"weather",("jitter","scaling"),(0,),(0,),matched,Path(t)/"report")
            self.assertEqual(len(paired),1)
            self.assertEqual(len(summary),1)
            self.assertEqual(len(pd.read_csv(Path(t)/"report/weather_incomplete.csv")),1)

    def test_dry_run_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"weather").mkdir()
            np.save(root/"weather.npy",np.arange(3000*3,dtype=np.float32).reshape(3000,3))
            dates=pd.date_range("2020-01-01",periods=3000,freq="10min")
            pd.DataFrame({"date":dates.astype(str)}).to_csv(root/"weather/weather.csv",index=False)
            from scripts.asoc_simple_aug_real import file_sha
            t1=int(3000*.7); start=int(t1-t1*.1)
            n=t1-start-96-24+1
            self.assertGreater(n, 0)
            df=reference(reps=(0,),seeds=(0,),n=n,start=start,end=t1)
            df["data_sha256"]=file_sha(root/"weather.npy")
            ref=root/"ref.csv";df.to_csv(ref,index=False)
            out=root/"output.csv"
            main(["--domain","weather","--reps","0","--aug-seeds","0",
                 "--reference",str(ref),"--data-dir",str(root),"--out",str(out),"--dry-run"])
            self.assertFalse(out.exists())


if __name__=="__main__":
    unittest.main()
