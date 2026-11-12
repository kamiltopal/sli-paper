"""CPU protocol tests; test doubles for repository loaders/model, not a CUDA benchmark."""
import json, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scripts import asoc_phase16_compute as m

class Tiny(nn.Module):
    def __init__(self,L,H):
        super().__init__();self.linear=nn.Linear(L,H)
    def forward(self,x):return self.linear(x[:,:,0])

def loader(pair,batch_size,shuffle):
    return DataLoader(TensorDataset(*(torch.as_tensor(a,dtype=torch.float32) for a in pair)),batch_size=batch_size,shuffle=shuffle)

@torch.no_grad()
def evaluate(net,pair,batch_size,device):
    net.eval();se=ae=0.;n=0
    for x,y in loader(pair,batch_size,False):
        r=net(x.to(device))-y.to(device);se+=float(r.square().sum());ae+=float(r.abs().sum());n+=r.numel()
    return se/n,ae/n

class FakeClock:
    def __init__(self):self.t=0.
    def __call__(self):self.t+=.1;return self.t

class Phase16Tests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(len(m.ARMS)*len(m.p14.DOMAINS),8);self.assertEqual(m.ARMS[0],'timegan')
    def test_stopping(self):
        self.assertFalse(m.checkpoint_stop(5,10,6,None));self.assertTrue(m.checkpoint_stop(6,1,6,None));self.assertFalse(m.checkpoint_stop(100,1.9,None,2));self.assertTrue(m.checkpoint_stop(100,2,None,2))
    def test_tolerance(self):
        self.assertTrue(m.budget_status(104,100)[1]);self.assertFalse(m.budget_status(106,100)[1]);self.assertFalse(m.budget_status(94,100)[1]);self.assertAlmostEqual(m.budget_status(103,100)[0],3)
        for e,t in [(0,1),(1,0),(np.nan,1)]:
            with self.assertRaises(ValueError):m.budget_status(e,t)
    def run_fit(self,**kwargs):
        fake=types.ModuleType('scripts.asoc_budget_controls');fake.seed_everything=torch.manual_seed;fake._loader=loader;fake._evaluate=evaluate
        rng=np.random.default_rng(3);pair=(rng.normal(size=(12,4,1)).astype('float32'),rng.normal(size=(12,2)).astype('float32'));clock=FakeClock()
        key='scripts.asoc_budget_controls';old=sys.modules.get(key);sys.modules[key]=fake
        try:
            return m.fit_budget(pair,pair,pair,torch.device('cpu'),start=0.,interval=2,model_factory=lambda arch,L,H:Tiny(L,H),batch_size=4,clock=clock,**kwargs)
        finally:
            if old is None:sys.modules.pop(key,None)
            else:sys.modules[key]=old
    def test_fixed_fit_and_origin_aggregate(self):
        row,ms,ma=self.run_fit(fixed_steps=5);self.assertEqual(row['optimizer_steps'],5);self.assertEqual(row['validation_checkpoints'],3);self.assertEqual([x[0] for x in json.loads(row['validation_curve'])],[2,4,5]);self.assertAlmostEqual(ms.mean(),row['test_mse'],places=6);self.assertAlmostEqual(ma.mean(),row['test_mae'],places=6);self.assertEqual(len(ms),12);self.assertGreater(row['test_seconds'],0)
    def test_timed_fit(self):
        row,_,_=self.run_fit(budget=.25);self.assertEqual(row['optimizer_steps'],4);self.assertGreaterEqual(row['budgeted_seconds'],.25);self.assertEqual(row['validation_checkpoints'],2);self.assertIn(row['best_step'],(2,4))
    def test_invalid_rule(self):
        for kw in ({},{'budget':1,'fixed_steps':1},{'budget':0}):
            with self.assertRaises(ValueError):self.run_fit(**kw)
    def test_group_commit_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pair=(np.ones((8,96,1),dtype='float32'),np.ones((8,24),dtype='float32'));idx=np.arange(12,20)
            plan=dict(data={'train':pair,'val':pair,'test':pair},idx=idx,digest='d',rawsha='r',meta={'real_windows':8,'matched_steps':10,'target_sd':2.},tg=pd.DataFrame([dict(rep=0,pool_seed=0,test_mse=.5)]),refs={(0,c,p):pd.Series(dict(test_mse=1.)) for c,p in [('step_baseline',-1),('duplicate_real',-1),('vae_augmented',0)]})
            def fit(*args,**kwargs):
                n=10 if kwargs['fixed_steps'] is not None else 20;seconds=100. if n==10 else 103.
                return dict(optimizer_steps=n,samples_seen=n*8,train_windows=8,best_step=n,val_mse=1.,test_mse=1.,test_mae=1.,validation_checkpoints=1,validation_curve=json.dumps([[n,1.,seconds]]),budgeted_seconds=seconds,forecaster_loop_seconds=seconds-1,test_seconds=.1),np.ones(8),np.ones(8)
            with patch.object(m,'warmup'),patch.object(m,'generate',side_effect=lambda a,p,d:(pair,{}) if a in ('timegan','vae') else (None,{})),patch.object(m,'fit_budget',side_effect=fit):
                m.run_group(root,'weather',plan,'cid',torch.device('cpu'))
            self.assertFalse((root/'weather.inprogress').exists());self.assertEqual(len(m.validate_group(root,plan,'weather','cid')),4)
            f=root/'weather/origins/timegan.npz';f.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Origin missing or changed'):m.validate_group(root,plan,'weather','cid')
    def test_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);df=pd.DataFrame([dict(domain='weather',arm=a,test_mse=1+i,test_mae=1+i,budgeted_seconds=100+i,optimizer_steps=10+i,budget_delta_pct=i,within_tolerance=True) for i,a in enumerate(m.ARMS)])
            m.report(root,[df]);f=pd.read_csv(root/'report/comparisons.csv');self.assertEqual(len(f),3);self.assertTrue((f.delta_mse<0).all());self.assertEqual(f.comparator.tolist(),list(m.ARMS[1:]))

if __name__=='__main__':unittest.main()
