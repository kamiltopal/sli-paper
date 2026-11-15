import unittest,tempfile,math,json,zipfile,sys,types,random,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from scripts import asoc_timegan as g
from scripts import asoc_real_timegan as m

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(4);self.real=(rng.normal(size=(12,8,1)).astype('float32'),rng.normal(size=(12,4)).astype('float32'))
        self.gen=dict(width=4,layers=1,batch_size=4,lr=.001,ae_steps=2,sup_steps=2,joint_steps=2,sample_batch=5)
        self.dev=torch.device('cpu')
    def test_training_and_shape(self):
        (x,y),s=g.generate_pool(self.real,seed=1,device=self.dev,**self.gen)
        self.assertEqual(x.shape,self.real[0].shape);self.assertEqual(y.shape,self.real[1].shape)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        self.assertEqual((s['ae_updates'],s['supervisor_pretrain_updates'],s['joint_generator_updates'],s['joint_embedder_updates'],s['joint_discriminator_updates']),(2,2,4,4,2))
        self.assertGreater(s['synthetic_std'],0)
    def test_repeat_cpu_seed(self):
        a,_=g.generate_pool(self.real,seed=1,device=self.dev,**self.gen);b,_=g.generate_pool(self.real,seed=1,device=self.dev,**self.gen)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):g.generate_pool((np.zeros_like(self.real[0]),np.zeros_like(self.real[1])),seed=0,device=self.dev,**self.gen)
    def test_cache(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);md=dict(domain='weather',pool_seed=0,generator=self.gen)
            a,s,f=m.pool_for(p,self.real,md,self.dev)
            with patch.object(m,'generate_pool',side_effect=AssertionError('regenerated')):b,t,h=m.pool_for(p,self.real,md,self.dev)
            self.assertEqual(f,h);self.assertEqual(s,t);np.testing.assert_array_equal(a[0],b[0])
            with np.load(f) as z:x=z['x'];y=z['y'];stats=z['stats']
            m.atomic_npz(f,x=x,y=y,stats=stats,metadata=json.dumps({'wrong':True}))
            with self.assertRaisesRegex(ValueError,'metadata'):m.pool_for(p,self.real,md,self.dev)
    def refs(self):
        meta=dict(data_length=2500,train_start=1575,train_end=1750,real_windows=56,val_windows=227,test_windows=477,target_mean=0.,target_sd=2.,original_steps=10,matched_steps=20)
        rows=[]
        for rep in (0,1):
            for c,ps in [(c,-1) for c in m.CONTROLS]+[('vae_augmented',p) for p in (0,1)]:
                rows.append(dict(**m.CONFIG,**meta,experiment_version='phase7_real_equal_step_v1',domain='weather',data_sha256='a',rep=rep,condition=c,pool_seed=ps,optimizer_steps=10 if c=='epoch_baseline' else 20,test_mse=1.,test_mae=.5,mse_raw=4.,mae_raw=1.,normalization='per_regime_train_segment_only',val_checkpoint_selection='minimum_validation_mse'))
        return pd.DataFrame(rows),meta
    def test_reference_fail_closed(self):
        f,meta=self.refs();self.assertEqual(len(m.select_refs(f,'weather','a',meta,(0,1),(0,1))),10)
        f.loc[f.condition=='step_baseline','optimizer_steps']=19
        with self.assertRaisesRegex(ValueError,'optimizer'):m.select_refs(f,'weather','a',meta,(0,1),(0,1))
    def test_report(self):
        f,meta=self.refs();refs={'weather':m.select_refs(f,'weather','a',meta,(0,1),(0,1))}
        d=pd.DataFrame([dict(domain='weather',rep=r,pool_seed=p,optimizer_steps=20,test_mse=.8,test_mae=.4,mse_raw=3.2,mae_raw=.8) for p in (0,1) for r in (0,1)])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m.report(d,refs,root,('weather',),(0,1),(0,1));s=pd.read_csv(root/'report/summary.csv');self.assertEqual(int(s.n_pairs.iloc[0]),4);self.assertAlmostEqual(s.timegan_vs_duplicate_real_test_mse_pct.iloc[0],-20)
            with self.assertRaisesRegex(ValueError,'incomplete'):m.report(d.iloc[:-1],refs,root,('weather',),(0,1),(0,1))
    def test_origin_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);pool=p/'pool.npz';origin=p/'origin.npz';m.atomic_npz(pool,x=np.zeros(1));m.atomic_npz(origin,origin=np.arange(2),mse=np.array([1.,3.]),mae=np.array([1.,1.]))
            d=pd.DataFrame([dict(domain='weather',rep=0,pool_seed=0,data_sha256='a',config_id='c',experiment_version=m.VERSION,optimizer_steps=20,pool_file=pool.name,pool_sha256=m.sha(pool),origin_file=origin.name,origin_sha256=m.sha(origin),test_mse=2.,test_mae=1.)]);m.validate_existing(d,p,'weather','a',dict(matched_steps=20,test_windows=2),'c');origin.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):m.validate_existing(d,p,'weather','a',dict(matched_steps=20,test_windows=2),'c')

    def test_forecaster_origin_metrics(self):
        # Small local fixture for the runner's repository helper interface.
        helper=types.ModuleType('scripts.asoc_budget_controls')
        def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
        def loader(pair,batch_size,shuffle):
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*[torch.tensor(x) for x in pair]),batch_size=batch_size,shuffle=shuffle)
        def evaluate(model,pair,batch_size,device):
            model.eval()
            with torch.no_grad():
                r=model(torch.tensor(pair[0]))-torch.tensor(pair[1]);return float(r.square().mean()),float(r.abs().mean())
        helper.seed_everything=seed;helper._loader=loader;helper._evaluate=evaluate
        factory=lambda arch,L,H:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(L,H))
        with patch.dict(sys.modules,{'scripts.asoc_budget_controls':helper}):
            met,ms,ma=m.fit_with_origins(self.real,self.real,self.real,0,3,1,self.dev,batch_size=4,model_factory=factory)
        self.assertEqual(met['optimizer_steps'],3);self.assertEqual(len(ms),12);self.assertAlmostEqual(ms.mean(),met['test_mse'],places=6);self.assertAlmostEqual(ma.mean(),met['test_mae'],places=6)

    def test_repo_fit_parity(self):
        # Executed in the user's repository; standalone distribution has no old helper.
        try:from scripts.asoc_budget_controls import fit_condition
        except ModuleNotFoundError:self.skipTest('Phase 7 repository helper not installed in standalone test folder')
        factory=lambda arch,L,H:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(L,H))
        ref=fit_condition(arch='itransformer',train_pair=self.real,val_pair=self.real,test_pair=self.real,init_seed=1,max_steps=4,validation_interval=2,batch_size=4,lr=.001,device=self.dev,model_factory=factory)
        got,ms,ma=m.fit_with_origins(self.real,self.real,self.real,1,4,2,self.dev,batch_size=4,model_factory=factory)
        for k in ('test_mse','test_mae','val_mse','best_step','optimizer_steps','samples_seen','validation_curve'):
            self.assertEqual(got[k],ref[k],k)

if __name__=='__main__':unittest.main()
