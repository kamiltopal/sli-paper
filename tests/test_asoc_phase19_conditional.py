import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scripts import asoc_phase19_conditional as m

class Phase19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        r=np.random.default_rng(6);self.real=(r.normal(size=(16,96,3)).astype('float32'),r.normal(size=(16,24)).astype('float32'));self.dev=torch.device('cpu')
    def test_matrix(self):
        self.assertEqual(len(m.expected()),24);self.assertEqual(len(set(m.expected())),24)
    def test_parameter_parity_and_mask(self):
        torch.manual_seed(0);a=m.MatchedForecaster('independent_fusion',16,4,1).eval();b=m.MatchedForecaster('channel_mixing',16,4,1).eval();b.load_state_dict(a.state_dict())
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x=torch.tensor(self.real[0][:3]);y=x.clone();y[:,:,1]+=3
        with torch.no_grad():
            torch.testing.assert_close(a.encode(x)[:,0],a.encode(y)[:,0]);self.assertFalse(torch.allclose(a(x),a(y)));self.assertFalse(torch.allclose(b.encode(x)[:,0],b.encode(y)[:,0]))
    def test_target_only_ignores_aux(self):
        a=m.MatchedForecaster('target_only',16,4,1).eval();x=torch.tensor(self.real[0][:3]);y=x.clone();y[:,:,1:]+=99
        with torch.no_grad():torch.testing.assert_close(a(x),a(y),rtol=0,atol=0)
    def test_generator_shape_same_aux(self):
        a,i,s=m.generate_pool(self.real,False,self.dev,epochs=1,batch_size=8,width=16);b,j,t=m.generate_pool(self.real,True,self.dev,epochs=1,batch_size=8,width=16)
        np.testing.assert_array_equal(i,j);np.testing.assert_array_equal(a[0][:,:,1:],b[0][:,:,1:]);np.testing.assert_array_equal(a[0][:,:,1:],self.real[0][i,:,1:]);self.assertEqual(s['generator_parameters'],t['generator_parameters']);self.assertEqual(s['generator_updates'],2);self.assertFalse(np.array_equal(a[1],b[1]))
    def test_generator_reproducibility(self):
        a,i,_=m.generate_pool(self.real,True,self.dev,epochs=1,batch_size=8,width=16);b,j,_=m.generate_pool(self.real,True,self.dev,epochs=1,batch_size=8,width=16)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(i,j)
    def test_pool_cache_and_tamper(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(m.CONFIG,{'gen_epochs':1}):
            r=Path(td);a,i,_,h=m.ensure_pool(r,'conditional_vae',self.real,'test',self.dev)
            with patch.object(m,'generate_pool',side_effect=AssertionError('regeneration')):b,j,_,h2=m.ensure_pool(r,'conditional_vae',self.real,'test',self.dev)
            self.assertEqual(h,h2);np.testing.assert_array_equal(a[1],b[1]);self.assertRaisesRegex(ValueError,'metadata',m.load_pool,r,'conditional_vae',self.real,'wrong')
    def test_forecaster_origin_and_checkpoint(self):
        for mode in m.MODES:
            metrics,(ms,ma)=m.fit(self.real,self.real,self.real,mode,0,2,1,self.dev,batch=4,width=16,heads=4,layers=1)
            self.assertEqual(metrics['optimizer_steps'],2);self.assertEqual(len(ms),16);self.assertAlmostEqual(float(ms.mean()),metrics['test_mse'],places=6);self.assertAlmostEqual(float(ma.mean()),metrics['test_mae'],places=6);self.assertIn(metrics['best_step'],(1,2))
    def test_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);df=pd.DataFrame([dict(mode=mode,arm=arm,rep=rep,test_mse=1.+m.ARMS.index(arm)/10,test_mae=.5+m.ARMS.index(arm)/20) for mode,arm,rep in m.expected()]);m.report(root,df)
            self.assertEqual(len(pd.read_csv(root/'report/paired.csv')),30);self.assertEqual(len(pd.read_csv(root/'report/interaction.csv')),8)
            self.assertRaisesRegex(ValueError,'24',m.report,root,df.iloc[:-1])
if __name__=='__main__':unittest.main()
