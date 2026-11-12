"""Compact target-only TimeGAN adaptation (Yoon, Jarrett, van der Schaar, 2019).
Original PyTorch implementation; fixed-length temporal windows, no static features.
https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
"""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class SeqNet(nn.Module):
    def __init__(self, inp, width, out, layers, sigmoid=True):
        super().__init__()
        self.rnn=nn.GRU(inp,width,num_layers=layers,batch_first=True)
        self.head=nn.Linear(width,out)
        self.sigmoid=sigmoid
    def forward(self,x):
        y=self.head(self.rnn(x)[0]); return y.sigmoid() if self.sigmoid else y

class TimeGAN(nn.Module):
    def __init__(self,width=24,layers=3):
        super().__init__()
        self.e=SeqNet(1,width,width,layers)
        self.r=SeqNet(width,width,1,layers)
        self.g=SeqNet(1,width,width,layers)
        self.s=SeqNet(width,width,width,max(1,layers-1))
        self.d=SeqNet(width,width,1,layers,False)
    def sample(self,z): return self.r(self.s(self.g(z)))

def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def sup_loss(m,h): return F.mse_loss(m.s(h)[:,:-1],h[:,1:])
def bce(x,real): return F.binary_cross_entropy_with_logits(x,torch.ones_like(x) if real else torch.zeros_like(x))
def moments(x,y):
    return (x.mean(0)-y.mean(0)).abs().mean()+((x.var(0,unbiased=False)+1e-6).sqrt()-(y.var(0,unbiased=False)+1e-6).sqrt()).abs().mean()

def generate_pool(real, *, seed, device, width=24,layers=3,batch_size=64,lr=.001,
                  ae_steps=200,sup_steps=200,joint_steps=400,sample_batch=128):
    """One fixed pool; train-only min/max, full target windows, 1:1 sample count.
    Each joint iteration: 2 G/S and E/R updates, 1 D update (no threshold skip).
    All iterations are fixed in advance, not selected using forecasting outcomes.
    """
    if min(width,layers,batch_size,ae_steps,sup_steps,joint_steps,sample_batch)<1 or lr<=0: raise ValueError('Invalid TimeGAN configuration')
    x,y=real
    if x.ndim!=3 or x.shape[2]!=1 or y.ndim!=2 or len(x)!=len(y) or len(x)<2 or min(x.shape[1],y.shape[1])<1: raise ValueError('Expected target-only windows')
    full=np.concatenate((x[:,:,0],y),1).astype(np.float32)
    if not np.isfinite(full).all(): raise ValueError('Nonfinite train windows')
    lo,hi=float(full.min()),float(full.max())
    if hi<=lo: raise ValueError('Constant target; TimeGAN minmax undefined')
    torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    data=torch.from_numpy((full-lo)/(hi-lo)).unsqueeze(-1).to(device)
    n,T,_=data.shape; b=min(batch_size,n)
    m=TimeGAN(width,layers).to(device)
    opt_e=torch.optim.Adam(list(m.e.parameters())+list(m.r.parameters()),lr=lr)
    opt_s=torch.optim.Adam(m.s.parameters(),lr=lr)
    opt_g=torch.optim.Adam(list(m.g.parameters())+list(m.s.parameters()),lr=lr)
    opt_d=torch.optim.Adam(m.d.parameters(),lr=lr)
    def batch(): return data[torch.randint(n,(b,),device=device)]
    def noise(k): return torch.rand(k,T,1,device=device)
    def step(opt,loss):
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite TimeGAN loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();return float(loss.detach())
    def freeze(modules,on):
        for mod in modules:
            for p in mod.parameters(): p.requires_grad_(not on)
    sync(device);t=time.perf_counter()
    for _ in range(ae_steps):
        a=batch(); ae=step(opt_e,10*torch.sqrt(F.mse_loss(m.r(m.e(a)),a)+1e-8))
    sync(device);ae_sec=time.perf_counter()-t;t=time.perf_counter()
    for _ in range(sup_steps):
        with torch.no_grad(): h=m.e(batch())
        sup=step(opt_s,sup_loss(m,h))
    sync(device);sup_sec=time.perf_counter()-t;t=time.perf_counter()
    for _ in range(joint_steps):
        for __ in range(2):
            a=batch();z=noise(b)
            freeze([m.e,m.r,m.d],True)
            h=m.e(a).detach();eh=m.g(z);hh=m.s(eh);fake=m.r(hh)
            gl=step(opt_g,bce(m.d(hh),True)+bce(m.d(eh),True)+100*torch.sqrt(sup_loss(m,h)+1e-8)+100*moments(fake,a))
            freeze([m.e,m.r,m.d],False);freeze([m.s],True)
            h=m.e(a);el=step(opt_e,10*torch.sqrt(F.mse_loss(m.r(h),a)+1e-8)+.1*sup_loss(m,h))
            freeze([m.s],False)
        a=batch()
        with torch.no_grad(): h=m.e(a);eh=m.g(noise(b));hh=m.s(eh)
        dl=step(opt_d,bce(m.d(h),True)+bce(m.d(hh),False)+bce(m.d(eh),False))
    sync(device);joint_sec=time.perf_counter()-t;t=time.perf_counter()
    m.eval();parts=[]
    with torch.no_grad():
        for start in range(0,n,sample_batch):
            parts.append(m.sample(noise(min(sample_batch,n-start))).squeeze(-1).cpu().numpy())
    sync(device);sample_sec=time.perf_counter()-t
    pool=(np.concatenate(parts)*(hi-lo)+lo).astype(np.float32)
    if pool.shape!=full.shape or not np.isfinite(pool).all(): raise FloatingPointError('Invalid TimeGAN pool')
    L=x.shape[1]
    stats=dict(ae_seconds=ae_sec,supervisor_seconds=sup_sec,joint_seconds=joint_sec,
               generator_train_seconds=ae_sec+sup_sec+joint_sec,generator_sample_seconds=sample_sec,
               ae_updates=ae_steps,supervisor_pretrain_updates=sup_steps,joint_generator_updates=2*joint_steps,
               joint_embedder_updates=2*joint_steps,joint_discriminator_updates=joint_steps,
               last_ae_loss=ae,last_supervisor_loss=sup,last_generator_loss=gl,last_embedder_loss=el,last_discriminator_loss=dl,
               train_min=lo,train_max=hi,synthetic_std=float(pool.std()),real_std=float(full.std()),
               synthetic_mean=float(pool.mean()),real_mean=float(full.mean()),n_windows=n)
    return (pool[:,:L,None],pool[:,L:]),stats
