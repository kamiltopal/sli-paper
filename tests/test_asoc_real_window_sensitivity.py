"""Protocol contract tests for Weather window sensitivity.

Fast CPU tests mock forecaster fitting and VAE; actual GPU results must be
collected on the user's machine and checked separately.
"""
import tempfile
import types
import sys
from unittest.mock import patch
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import WindowConfig, make_windows, chrono_split, apply_regime
from scripts import asoc_real_window_sensitivity as w


class TestPhase9(unittest.TestCase):
    def test_configs_fail_closed(self):
        self.assertEqual(w.parse_configs('48:24,96:48'), ((48,24),(96,48)))
        for invalid in ('', '96:24', '48:24,48:24', '48:48', 'abc', '48', '96:24,96:48'):
            with self.subTest(spec=invalid), self.assertRaises(ValueError):
                w.parse_configs(invalid)

    def test_expected_counts_and_versioned_keys(self):
        kw=w.settings(epochs=10,vae_epochs=30,batch_size=64,lr=.001,latent=16)
        rows=list(w.expected(lookback=48,horizon=24,reps=(0,1,2),
                             pool_seeds=(0,1),digest='abc',**kw))
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({w.rowkey(r) for r in rows}),15)
        self.assertEqual(sum(r['pool_seed']==-1 for r in rows), 9)
        self.assertEqual(sum(r['condition']=='vae_augmented' for r in rows),6)

    def test_weather_expected_step_budget(self):
        t1=int(52696*.7)
        for (L,H),n,short,long in [((48,24),3618,570,1140),
                                   ((96,48),3546,560,1110),
                                   ((96,24),3570,560,1120)]:
            start=apply_regime(t1, .1, L)
            self.assertEqual(start,33198)
            self.assertEqual(t1-start-L-H+1,n)
            from scripts.asoc_budget_controls import count_steps
            self.assertEqual(count_steps(n,n,64,10), (short,long))

    def test_timeline_earlier_anomaly_acceptable_but_active_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'weather.csv'
            series=pd.date_range('2020-01-01',periods=500,freq='10min')
            series=pd.Series(series)
            series.iloc[11]=series.iloc[10]
            series.to_frame('date').to_csv(f,index=False)
            self.assertEqual(w.active_timeline_check(f,500,250),'active_10min_regular')
            with self.assertRaisesRegex(ValueError,'irregular'):
                w.active_timeline_check(f,500,0)
            series.iloc[300]=series.iloc[299]
            series.to_frame('date').to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError,'irregular'):
                w.active_timeline_check(f,500,250)
            self.assertEqual(w.active_timeline_check(f.parent/'missing.csv',500,250),
                             'raw_csv_unavailable')

    def test_cached_pool_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            L,H=48,24;n=6;seg=np.random.default_rng(2).normal(size=(200,3)).astype('float32')
            params=w.pool_metadata(digest='aaa',lookback=L,horizon=H,train_start=1,
                                   train_end=201,vae_epochs=1,latent=4)
            path=w.pool_path(Path(d),params,0)
            calls=[]
            def gen(segment, cfg, ratio, seed, epochs, latent):
                calls.append(1)
                return (np.ones((n,L,3),dtype='float32'), np.ones((n,H),dtype='float32'))
            with patch('src.generators.gen_vae', side_effect=gen):
                p,sha,_=w.load_or_generate_pool(path,params,seg,WindowConfig(L,H),0,n,torch.device('cpu'))
                self.assertEqual(len(calls),1)
                p2,sha2,s=w.load_or_generate_pool(path,params,seg,WindowConfig(L,H),0,n,torch.device('cpu'))
                self.assertEqual(len(calls),1)
                self.assertEqual(sha,sha2)
                self.assertEqual(s,0)
                self.assertTrue(np.array_equal(p[0],p2[0]))
            with np.load(path,allow_pickle=False) as obj:
                saved={k:obj[k] for k in obj.files}
            saved['x']=saved['x']+1
            with path.open('wb') as fd:np.savez_compressed(fd,**saved)
            with self.assertRaisesRegex(ValueError,'Corrupted'):
                w.load_or_generate_pool(path,params,seg,WindowConfig(L,H),0,n,torch.device('cpu'))

    def test_fake_end_to_end_resume_and_paired(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);src=base/'input';src.mkdir()
            t=np.arange(2400)
            data=np.column_stack([np.sin(t/20),np.cos(t/30),t*.002]).astype('float32')
            path=src/'weather.npy';np.save(path,data)
            out=base/'run.csv';pool_dir=base/'pools';fitted=[];generated=[]
            kw=w.settings(epochs=1,vae_epochs=1,batch_size=32,lr=.001,latent=4)
            def fake_fit(**kwargs):
                fitted.append(kwargs)
                steps=kwargs['max_steps']
                self.assertEqual(kwargs['train_pair'][0].shape[2],1)
                self.assertEqual(kwargs['train_pair'][1].shape[1],kwargs['test_pair'][1].shape[1])
                return dict(test_mse=1+len(fitted)*.01,test_mae=.4,
                            val_mse=.25,best_step=steps-1,optimizer_steps=steps,
                            samples_seen=steps*32,train_windows=len(kwargs['train_pair'][0]),
                            validation_checkpoints=1,validation_curve='[]',train_seconds=.01)
            def fake_gen(segment,cfg,ratio,seed,epochs,latent):
                generated.append((cfg.lookback,cfg.horizon,seed))
                x,y=make_windows(segment,0,len(segment),cfg)
                return x+seed*.02,y+seed*.02
            args=dict(cfg=(48,24),reps=(0,1),pool_seeds=(0,1),
                      digest=w.sha256(path),data_path=path,
                      raw_csv=base/'missing.csv',out=out,pool_dir=pool_dir,
                      gpu_index=0,allow_cpu=True,dry_run=False,**kw)
            with patch.object(w,'select_device',return_value=torch.device('cpu')),\
                 patch.object(w,'fit_condition',side_effect=fake_fit),\
                 patch('src.generators.gen_vae',side_effect=fake_gen):
                w.run_config(**args)
                self.assertEqual(len(fitted),10) # 3 controls*2 reps+2 pools*2 reps
                self.assertEqual(len(generated),2)
                w.run_config(**args)
                self.assertEqual(len(fitted),10)
                self.assertEqual(len(generated),2)
                df=pd.read_csv(out)
                self.assertEqual(df.pool_sha256.nunique(),2) # two independent pools; controls blank
                pair,missing=w.paired_table(df)
                self.assertEqual(len(pair),4)
                self.assertTrue(missing.empty)
                self.assertTrue((pair.matched_steps==df.matched_steps.iloc[0]).all())
                self.assertEqual(set(df.lookback),{48})
                self.assertEqual(set(df.horizon),{24})
                self.assertTrue(set(df.cadence_check)=={'raw_csv_unavailable'})
                wrong=df.copy();wrong.loc[wrong.condition=='duplicate_real','optimizer_steps']-=1
                with self.assertRaisesRegex(ValueError,'budget'):
                    w.paired_table(wrong)
                wrong=df.copy();wrong.loc[wrong.condition=='vae_augmented','mse_raw']+=1
                with self.assertRaisesRegex(ValueError,'conversion'):
                    w.paired_table(wrong)
                wrong=df.copy();wrong.loc[(wrong.condition=='vae_augmented') & (wrong.rep==0) & (wrong.pool_seed==0),'pool_sha256']='x'
                with self.assertRaisesRegex(ValueError,'differing'):
                    w.read_runs(self._write(base/'invalid.csv',wrong),w.sha256(path))
                np.save(path,data+10)
                with self.assertRaisesRegex(ValueError,'SHA changed'):
                    w.run_config(**{**args,'digest':w.sha256(path)})

    @staticmethod
    def _write(path,data):
        data.to_csv(path,index=False)
        return path

    def test_pool_median_and_partial_scope(self):
        def frame(L,H,source):
            return pd.DataFrame([
                dict(domain='weather',regime=.1,lookback=L,horizon=H,rep=rep,
                     pool_seed=ps,source=source,matched_steps=100,vae_best_at_last=False,
                     beats_both=True,vae_vs_step_baseline_test_mse_pct=v,
                     vae_vs_duplicate_real_test_mse_pct=v-3)
                for ps,v in ((0,-5),(1,-9)) for rep in (0,1,2)])
        old=frame(96,24,'phase7_read_only')
        new=frame(48,24,'phase9_new')
        # Only relative differences are summarized; no raw cross-horizon ranking.
        paired,pools,summary=w.make_reports(new,old,((48,24),(96,48)),(0,1,2),(0,1))
        self.assertEqual(len(paired),12)
        self.assertEqual(len(pools),4)
        self.assertEqual(len(summary),2)
        self.assertAlmostEqual(float(summary.iloc[0]['median_vae_vs_step_baseline_test_mse_pct']),-7)
        self.assertTrue(summary.complete_for_requested_seeds.all())


    def test_dry_run_does_not_create_results(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            data=np.random.default_rng(1).normal(size=(2400,3)).astype('float32')
            np.save(base/'weather.npy',data)
            out=base/'never_created.csv'
            args=['program','--data-dir',str(base),'--out',str(out),
                  '--configs','48:24,96:48','--reps','0','--pool-seeds','0',
                  '--dry-run']
            with patch.object(sys,'argv',args):
                w.main()
            self.assertFalse(out.exists())
            self.assertFalse((base/'window_sensitivity_pools_v1').exists())

    def test_report_with_only_read_only_baseline_marks_new_configs_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);np.save(root/'weather.npy', np.ones((2400,3),dtype='float32'))
            outdir=root/'reports'
            record=dict(domain='weather',regime=.1,lookback=96,horizon=24,
                        rep=0,pool_seed=0,source='phase7_read_only',
                        matched_steps=1120,vae_best_at_last=False,beats_both=True,
                        vae_vs_step_baseline_test_mse_pct=-10.0,
                        vae_vs_duplicate_real_test_mse_pct=-9.0)
            args=['program','--data-dir',str(root),'--out',str(root/'no_phase9_yet.csv'),
                  '--outdir',str(outdir),'--report','--reps','0','--pool-seeds','0']
            with patch.object(sys,'argv',args), patch.object(w,'phase7_reference',
                    return_value=pd.DataFrame([record])):
                w.main()
            summary=pd.read_csv(outdir/'window_summary.csv')
            self.assertEqual(summary[['lookback','horizon']].values.tolist(),[[96,24]])
            self.assertTrue(bool(summary.complete_for_requested_seeds.iloc[0]))

    def test_phase7_reference_read_only_and_data_sha_guard(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'reference.csv'
            kw=w.settings(epochs=10,vae_epochs=30,batch_size=64,lr=.001,latent=16)
            rows=[]
            for c,ps in [('epoch_baseline',-1),('step_baseline',-1),
                         ('duplicate_real',-1),('vae_augmented',0)]:
                rows.append(dict(experiment_version='phase7_real_equal_step_v1',
                                 hidden=128,horizon_weight=.5,domain='weather',
                                 data_sha256='correct',regime=.1,arch='itransformer',
                                 input_mode='target',lookback=96,horizon=24,
                                 rep=0,condition=c,pool_seed=ps,**kw))
            rows.append({**rows[-1], 'domain':'traffic','data_sha256':'another'})
            pd.DataFrame(rows).to_csv(file,index=False)
            before=file.read_bytes()
            def fake_phase7(frame):
                self.assertEqual(len(frame),4)
                self.assertTrue((frame.data_sha256=='correct').all())
                entry=dict(rep=0,pool_seed=0,matched_steps=1120,
                           vae_best_at_last=False,vae_beats_both_matched=True)
                for m in ('test_mse','test_mae','mse_raw','mae_raw'):
                    entry['vae_'+m]=.75
                    for c in w.CONDITIONS[:3]:
                        entry[c+'_'+m]=1.0
                        entry['vae_vs_'+c+'_'+m+'_pct']=-25.0
                        entry['vae_vs_'+c+'_'+m+'_abs']=-.25
                return pd.DataFrame([entry]), pd.DataFrame()
            stub=types.ModuleType('scripts.asoc_real_report')
            stub.paired_table=fake_phase7
            with patch.dict(sys.modules,{'scripts.asoc_real_report':stub}):
                baseline=w.phase7_reference(file,'correct',(0,),(0,),**kw)
                self.assertEqual(len(baseline),1)
                self.assertEqual(set(baseline.source),{'phase7_read_only'})
                self.assertEqual(file.read_bytes(),before)
                with self.assertRaisesRegex(ValueError,'SHA differs'):
                    w.phase7_reference(file,'wrong',(0,),(0,),**kw)
                with self.assertRaisesRegex(ValueError,'missing model/pool'):
                    w.phase7_reference(file,'correct',(0,),(0,1),**kw)


if __name__=='__main__':unittest.main()
