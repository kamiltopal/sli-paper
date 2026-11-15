"""Phase 10 protocol tests; small synthetic data, no Weather GPU result claims."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts import asoc_real_diffusion_weather as m
from scripts.asoc_budget_controls import count_steps
from scripts.asoc_real_controls import sha256
from src.data import WindowConfig, apply_regime, chrono_split, prepare


class Phase10Tests(unittest.TestCase):
    def fixture(self, folder, *, reps=(0,1), pools=(0,1)):
        d=Path(folder)
        x=np.stack([np.sin(np.arange(2500)/13)+.2*np.sin(np.arange(2500)/67),
                    np.cos(np.arange(2500)/25)],axis=1).astype('float32')
        array=d/'weather.npy';np.save(array,x)
        cfg=WindowConfig(96,24);p=prepare(x,.10,cfg);n=len(p['train'][0])
        t1,_=chrono_split(len(x));st=apply_regime(t1,.10,96)
        orig,matched=count_steps(n,n,32,1)
        sd=float(p['sd'][0,0])
        rows=[]
        for rep in reps:
            for c,ps in ([(name,-1) for name in m.CONTROLS]+[('vae_augmented',ps) for ps in pools]):
                mse=0.7 if c=='epoch_baseline' else .9 if c=='step_baseline' else 1.0 if c=='duplicate_real' else .8 + .01*ps + .02*rep
                mae=mse/2
                rows.append(dict(experiment_version=m.REF_VERSION,domain='weather',
                    data_sha256=sha256(array),regime=.10,arch='itransformer',
                    input_mode='target',rep=rep,condition=c,pool_seed=ps,epochs=1,
                    batch_size=32,lr=.001,vae_epochs=1,latent=16,hidden=128,
                    horizon_weight=.5,lookback=96,horizon=24,
                    optimizer_steps=orig if c=='epoch_baseline' else matched,
                    original_steps=orig,matched_steps=matched,test_mse=mse,
                    test_mae=mae,mse_raw=mse*sd*sd,mae_raw=mae*sd,target_sd=sd,
                    train_start=st,train_end=t1,real_windows=n))
        reference=d/'real_budget_controls.csv';pd.DataFrame(rows).to_csv(reference,index=False)
        return dict(data=array,reference=reference,out=d/'diffusion.csv',
                    cache=d/'cache',report=d/'reports', n=n,orig=orig,matched=matched,
                    start=st,end=t1)

    @staticmethod
    def argv(f, *, reps='0', pools='0', extra=()):
        return ['test','--data',str(f['data']),'--reference',str(f['reference']),
                '--out',str(f['out']),'--cache-dir',str(f['cache']),
                '--report-dir',str(f['report']),
                '--raw-timestamps',str(f['data'].parent/'unavailable.csv'),
                '--epochs','1','--batch-size','32','--vae-epochs','1',
                '--reps',reps,'--pool-seeds',pools,
                '--gen-epochs','1','--diffusion-steps','32',
                '--denoiser-width','16','--denoiser-batch','64',
                '--sampling-batch','32','--allow-cpu',*extra]

    def cli(self,f,*,reps='0',pools='0',extra=()):
        with patch.object(sys,'argv',self.argv(f,reps=reps,pools=pools,extra=extra)),\
             contextlib.redirect_stdout(io.StringIO()) as output:
            m.main()
        return output.getvalue()

    def fail_cli(self, f, *, reps='0', pools='0', extra=()):
        with patch.object(sys,'argv',self.argv(f,reps=reps,pools=pools,extra=extra)), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as errors, \
             self.assertRaises(SystemExit) as exit_code:
            m.main()
        self.assertEqual(exit_code.exception.code,2)
        return errors.getvalue()

    def test_diffusion_schedule_low_terminal_snr(self):
        for T in (32,64,128):
            beta, alpha, cum, var=m.diffusion_schedule(T,torch.device('cpu'))
            self.assertTrue(bool((beta>0).all()))
            self.assertTrue(bool((var>=0).all()))
            self.assertLess(float(cum[-1]),.015)
            self.assertEqual(tuple(cum.shape),(T,))
        with self.assertRaises(ValueError):
            m.diffusion_schedule(8,torch.device('cpu'))

    def test_model_shape_and_single_pool_generation(self):
        net=m.WindowDenoiser(120,16)
        self.assertEqual(tuple(net(torch.randn(3,120),torch.tensor([0,8,31])).shape),(3,120))
        rng=np.random.default_rng(4)
        real=(rng.normal(size=(13,96,1)).astype('float32'),
              rng.normal(size=(13,24)).astype('float32'))
        (x,y),meta=m.generate_pool(real,seed=0,epochs=1,steps=32,width=16,
                                    batch_size=8,lr=.001,sample_batch=7,
                                    device=torch.device('cpu'))
        self.assertEqual(x.shape,real[0].shape)
        self.assertEqual(y.shape,real[1].shape)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        self.assertEqual(meta['train_updates'],2)
        self.assertEqual(meta['total_synthetic_windows'],13)
        self.assertTrue(0 <= meta['spectral_fidelity'] <= 1)

    def test_cadence_audits_only_active_span(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'weather.csv'
            dt=pd.date_range('2020-01-01',periods=150,freq='10min')
            vals=dt.tolist()
            vals[10]=vals[9]
            pd.DataFrame({'date':vals}).to_csv(p,index=False)
            self.assertEqual(m.check_weather_active_timestamps(path=p,start=20),
                             'active_train_val_test_10min_regular')
            vals[90]=vals[89]
            pd.DataFrame({'date':vals}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'timestamp anomalies'):
                m.check_weather_active_timestamps(path=p,start=20)

    def test_reference_budget_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            d=pd.read_csv(f['reference'])
            d.loc[d.condition=='step_baseline','optimizer_steps']=2
            d.to_csv(f['reference'],index=False)
            self.assertIn('budget/partition mismatch',
                          self.fail_cli(f,extra=('--dry-run',)))

    def test_reference_requires_pool_and_model_seed(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td,pools=(0,))
            self.assertIn('Expected one Phase 7 row',
                          self.fail_cli(f,reps='0,1',pools='0,1',extra=('--dry-run',)))

    def test_dry_run_does_not_create_output_or_cache(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            with patch.object(m,'select_device',side_effect=AssertionError('GPU must not be touched')):
                output=self.cli(f,reps='0,1',pools='0,1',extra=('--dry-run',))
            self.assertIn('4 pending',output)
            self.assertFalse(f['out'].exists())
            self.assertFalse(f['cache'].exists())

    def test_cache_reuse_and_metadata_guard(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            p=prepare(np.load(f['data']),.10,WindowConfig(96,24))
            r=(p['train'][0][:,:,:1],p['train'][1])
            meta=m.pool_meta(data_sha256=sha256(f['data']),pool_seed=0,
                             gen_epochs=1,diffusion_steps=32,denoiser_width=16,
                             denoiser_batch=64,gen_lr=.001,sampling_batch=32,real_windows=f['n'])
            a,stats,path=m.cache_pool(f['cache'],r,meta,sample_batch=32,
                                      device=torch.device('cpu'))
            with patch.object(m,'generate_pool',side_effect=AssertionError('regenerated')):
                b,other,p2=m.cache_pool(f['cache'],r,meta,sample_batch=32,
                                         device=torch.device('cpu'))
            self.assertEqual(path,p2)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(stats,other)
            with np.load(path,allow_pickle=False) as x:
                first=x['x'].copy();last=x['y'].copy();oldstats=str(x['stats'].item())
            with path.open('wb') as handle:
                np.savez_compressed(handle,x=first,y=last,metadata=json.dumps({'wrong':True}),
                                    stats=oldstats)
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                m.cache_pool(f['cache'],r,meta,sample_batch=32,device=torch.device('cpu'))

    def test_end_to_end_resume_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            seen=[]
            def fake_fit(**kwargs):
                seen.append(kwargs['init_seed'])
                s=kwargs['max_steps']
                return dict(test_mse=.76+.05*kwargs['init_seed'],test_mae=.46,
                            val_mse=.55,best_step=s-1,optimizer_steps=s,
                            samples_seen=s*32,train_windows=len(kwargs['train_pair'][0]),
                            validation_checkpoints=2,validation_curve='[]',train_seconds=.01)
            with patch.object(m,'fit_condition',side_effect=fake_fit):
                self.cli(f,reps='0,1',pools='0,1',extra=('--limit','1'))
                self.assertEqual(len(pd.read_csv(f['out'])),1)
                self.assertEqual(len(list(f['cache'].glob('*.npz'))),1)
                self.cli(f,reps='0,1',pools='0,1')
                self.assertEqual(len(pd.read_csv(f['out'])),4)
                self.cli(f,reps='0,1',pools='0,1')
                self.assertEqual(len(seen),4)
            self.cli(f,reps='0,1',pools='0,1',extra=('--report',))
            paired=pd.read_csv(f['report']/'diffusion_paired.csv')
            pools=pd.read_csv(f['report']/'diffusion_per_pool.csv')
            summary=pd.read_csv(f['report']/'diffusion_summary.csv')
            self.assertEqual((len(paired),len(pools),int(summary.n_pairs.iloc[0])),(4,2,4))
            self.assertTrue((paired.matched_steps==f['matched']).all())
            self.assertTrue((paired.diffusion_vs_vae_test_mse_pct<0).all())

    def test_diffusion_output_detects_data_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            x=np.load(f['data']);np.save(f['data'],x+3)
            self.assertIn('hash mismatch',
                          self.fail_cli(f,extra=('--dry-run',)))

    def test_existing_duplicate_runs_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            row=dict(experiment_version=m.VERSION,data_sha256=sha256(f['data']),
                     rep=0,pool_seed=0,gen_epochs=1,diffusion_steps=32,
                     denoiser_width=16,denoiser_batch=64,gen_lr=.001,sampling_batch=32,
                     epochs=1,batch_size=32,lr=.001)
            pd.DataFrame([row,row]).to_csv(f['out'],index=False)
            self.assertIn('Duplicated diffusion',
                          self.fail_cli(f,extra=('--dry-run',)))

if __name__=='__main__':unittest.main()
