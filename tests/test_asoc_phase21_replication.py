"""CPU unit tests; no actual data or GPU needed."""
import random, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts import asoc_phase21_replication as p

# Small self-contained dependency doubles: explicitly test runner logic, not repo integration.
def helpers():
    m=types.ModuleType('scripts.asoc_budget_controls')
    def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
    def loader(pair,batch_size,shuffle):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*[torch.as_tensor(x) for x in pair]),batch_size=batch_size,shuffle=shuffle)
    @torch.no_grad()
    def evaluate(model,pair,batch_size,device):
        model.eval();e=torch.cat([model(x)-y for x,y in loader(pair,batch_size,False)])
        # Artificial RNG consumption proves isolation even for a future evaluator.
        random.random();np.random.rand();torch.rand(1)
        return float(e.square().mean()),float(e.abs().mean())
    m.seed_everything=seed;m._loader=loader;m._evaluate=evaluate
    return m

class Tests(unittest.TestCase):
    def test_locked_grid(self):
        j=p.jobs();self.assertEqual(len(j),360);self.assertEqual(len(set(j)),360)
        self.assertEqual(sum(k[0]=='replication' for k in j),280);self.assertEqual(sum(k[0]=='interaction' for k in j),80)
        self.assertEqual(sum(k[0]=='replication' and k[1]=='weather' for k in j),140)
        self.assertEqual(len([k for k in j if k[3]==0 and k[5] in (-1,0)]),10)
    def test_rng_restore(self):
        random.seed(2);np.random.seed(2);torch.manual_seed(2)
        with p.isolated_rng(): a=(random.random(),np.random.rand(),torch.rand(4));torch.rand(5)
        b=(random.random(),np.random.rand(),torch.rand(4));self.assertEqual(a[:2],b[:2]);torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
    def test_same_trajectory_nested_policies(self):
        rng=np.random.default_rng(3);pair=(rng.normal(size=(12,8,1)).astype('float32'),rng.normal(size=(12,3)).astype('float32'))
        class Toy(nn.Module):
            def __init__(self,L,H):super().__init__();self.net=nn.Sequential(nn.Linear(L,8),nn.Dropout(.3),nn.Linear(8,H))
            def forward(self,x):return self.net(x[:,:,0])
        with patch.dict(sys.modules,{'scripts.asoc_budget_controls':helpers()}):
            a,t,c,n,_=p.fit_dual(pair,pair,pair,0,8,4,1,torch.device('cpu'),batch_size=4,make_model=lambda mode,L,H:Toy(L,H))
            b,u,d,m,_=p.fit_dual(pair,pair,pair,0,8,4,4,torch.device('cpu'),batch_size=4,make_model=lambda mode,L,H:Toy(L,H))
        self.assertEqual([x for x in c if x[0]%4==0],d);self.assertEqual(a['coarse'],b['coarse']);np.testing.assert_array_equal(t['coarse'][0],u['coarse'][0]);self.assertEqual(n,m)
        self.assertLessEqual(a['fine']['val_mse'],a['coarse']['val_mse'])
    def test_report(self):
        mod=types.ModuleType('scripts.asoc_real_timegan')
        def atomic_csv(df,path):path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False)
        mod.atomic_csv=atomic_csv
        plans={'weather':{'meta':{'target_sd':2},'refs':{'test_persistence_mse':.9,'test_ridge96_mse':1.1}}};records=[]
        for k,v in [(('replication','weather','centered',0,'step_baseline',-1),1.2),(('replication','weather','centered',0,'duplicate_real',-1),1.0),(('replication','weather','centered',0,'vae',0),.8),(('interaction','weather','original',0,'vae',0),2.),(('interaction','weather','original',0,'timegan',0),1.),(('interaction','weather','centered',0,'vae',0),.8),(('interaction','weather','centered',0,'timegan',0),1.)]:
            records.append(dict(key=k,metrics={q:dict(test_mse=v,test_mae=v,val_mse=v,best_step=112) for q in p.POLICIES},optimizer_steps=1120,samples_seen=1,seconds=1.,data_sha256='x',pool_sha256='',pool_file='',origin_file='',origin_sha256='',config_id='x'))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'scripts.asoc_real_timegan':mod}):
            root=Path(tmp);p.summarize(records,root,plans);s=pd.read_csv(root/'report/summary.csv');a=s[(s.study=='replication')&(s.arm=='vae')&(s.reference=='duplicate_real')];np.testing.assert_allclose(a.delta_mse_pct,-20);self.assertEqual(len(pd.read_csv(root/'report/dual_wins.csv')),2);z=pd.read_csv(root/'report/representation_interaction.csv');np.testing.assert_allclose(z.centered_minus_original_pp,75)
if __name__=='__main__':unittest.main()
