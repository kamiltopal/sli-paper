import tempfile, unittest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import asoc_phase15_origins as m

class Phase15Tests(unittest.TestCase):
    def test_matrix(self):
        j=m.jobs(('weather','traffic'));self.assertEqual(len(j),24);self.assertEqual(len(set(j)),24);self.assertEqual(sum(c=='vae_replay' for _,_,c,_ in j),12)
    def test_constant(self):
        self.assertEqual(m.circular_ci(np.full(100,2.),24,100),(2.,2.,2.))
    def test_reproducible(self):
        x=np.arange(100.);self.assertEqual(m.circular_ci(x,24,100),m.circular_ci(x,24,100))
        a,l,h=m.circular_ci(x,24,100);self.assertEqual(a,49.5);self.assertLess(l,a);self.assertGreater(h,a)
    def test_invalid(self):
        for x,b in [(np.arange(10.),11),(np.array([np.nan,1]),1)]:
            with self.assertRaises(ValueError):m.circular_ci(x,b,100)
    def test_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'loss.npz';ix=np.arange(10,20);v=np.arange(10,dtype=float);np.savez(p,origin=ix,mse=v,mae=v)
            row=SimpleNamespace(origin_file=p.name,origin_sha256=m.p14.sha(p),test_mse=v.mean(),test_mae=v.mean());a,b=m.read_trace(root,row,ix);np.testing.assert_array_equal(a,v)
            with self.assertRaisesRegex(ValueError,'alignment'):m.read_trace(root,row,ix+1)
            row.origin_sha256='bad';self.assertRaises(ValueError,m.read_trace,root,row,ix)
    def test_report_shared_controls(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);idx=np.arange(400);rows=[]
            for d,r,c,ps in m.jobs(('weather',)):
                f=root/f'{c}_{r}_{ps}.npz';v=np.full(400,2.);np.savez(f,origin=idx,mse=v,mae=v)
                rows.append(dict(domain=d,rep=r,condition=c,pool_seed=ps,origin_file=f.name,origin_sha256=m.p14.sha(f),test_mse=2.,test_mae=2.))
            p={'idx':idx,'traces':{(r,s):(np.ones(400),np.ones(400)) for r in m.REPS for s in m.POOLS}}
            m.make_report(pd.DataFrame(rows),{'weather':p},root,draws=100)
            a=pd.read_csv(root/'report/block_bootstrap.csv');self.assertEqual(len(a),18);self.assertTrue((a.mean_delta==-1).all() and (a.ci_low==-1).all() and (a.ci_high==-1).all())
            self.assertEqual(len(pd.read_csv(root/'report/paired.csv')),36)
if __name__=='__main__':unittest.main()
