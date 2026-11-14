import unittest
import numpy as np
import torch
from torch import nn
from scripts.asoc_phase20_adequacy import LastValue,jobs,references

class Zero(nn.Module):
    def forward(self,x): return x.new_zeros((len(x),24))

class TestAdequacy(unittest.TestCase):
    def test_persistence(self):
        x=torch.randn(3,96,1);self.assertTrue(torch.equal(LastValue(Zero())(x),x[:,-1,0,None].expand(-1,24)))
    def test_translation(self):
        base=nn.Sequential(nn.Flatten(),nn.Linear(96,24));m=LastValue(base);x=torch.randn(4,96,1);torch.testing.assert_close(m(x+5),m(x)+5,atol=1e-5,rtol=1e-5)
    def test_fixed_jobs(self):
        j=jobs();self.assertEqual(len(j),30);self.assertEqual(len(set(j)),30);self.assertEqual(len(j[:7]),7);self.assertTrue(all(r==0 and p in (-1,0) for _,r,_,p in j[:7]));self.assertEqual(sum(d=='weather' for d,*_ in j),18)
    def test_reference(self):
        x=np.ones((10,96,1),dtype='float32');y=np.ones((10,24),dtype='float32');r=references((x,y),(x,y),(x,y));self.assertAlmostEqual(r['test_persistence_mse'],0);self.assertAlmostEqual(r['test_ridge96_mse'],0)

if __name__=='__main__':unittest.main()
