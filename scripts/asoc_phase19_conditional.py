"""Phase 19: EPİAŞ matched-information conditional VAE pilot (additive)."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
VERSION='phase19_epias_cvae_masked_attention_v1'
MODES=('target_only','independent_fusion','channel_mixing')
ARMS=('step_baseline','duplicate_real','unconditional_vae','conditional_vae')
REPS=(0,1)
CONFIG=dict(regime=.10,lookback=96,horizon=24,epochs=10,batch_size=64,lr=.001,
            gen_epochs=30,gen_batch=256,gen_lr=.001,latent=16,gen_width=128,
            beta=.001,pool_seed=0,width=64,heads=4,layers=2,dropout=0.)

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def digest(arrays):
    h=hashlib.sha256()
    for x in arrays:
        a=np.ascontiguousarray(x);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
def atomic_npz(path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**kw)
    os.replace(tmp,path)
def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);os.replace(tmp,path)
def resolve(p):return p if p.is_absolute() else ROOT/p

def checked_pair(pair, channels=3):
    x,y=pair;need(x.ndim==3 and x.shape[1:]==(96,channels) and y.shape==(len(x),24) and len(x)>0,'Bad window shape')
    need(x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all(),'Invalid windows')

class WindowVAE(nn.Module):
    """Same parameterization for conditional and unconditional (zero condition) models."""
    def __init__(self,latent=16,width=128):
        super().__init__();self.latent=latent
        self.enc=nn.Sequential(nn.Linear(120+192,width),nn.ReLU(),nn.Linear(width,2*latent))
        self.dec=nn.Sequential(nn.Linear(latent+192,width),nn.ReLU(),nn.Linear(width,120))
    def encode(self,w,c):
        q=self.enc(torch.cat([w,c],dim=1));return q[:,:self.latent],q[:,self.latent:].clamp(-12,12)
    def decode(self,z,c):return self.dec(torch.cat([z,c],dim=1))

def generate_pool(real,conditional,device,*,epochs=30,batch_size=256,seed=0,latent=16,width=128,lr=.001,beta=.001):
    checked_pair(real);need(epochs>0 and batch_size>0,'Invalid generator budget');seed_all(seed)
    x,y=real;n=len(x);full=np.concatenate([x[:,:,0],y],axis=1)
    c=x[:,:,1:].reshape(n,192).copy(); c_used=c if conditional else np.zeros_like(c)
    w=torch.as_tensor(full,device=device);aux=torch.as_tensor(c_used,device=device)
    sync(device);t=time.perf_counter();model=WindowVAE(latent,width).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr);updates=0
    for _ in range(epochs):
        for idx in torch.randperm(n,device=device).split(batch_size):
            mu,lv=model.encode(w[idx],aux[idx]);z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
            rec=model.decode(z,aux[idx]);loss=(rec-w[idx]).square().mean()+beta*(-.5*(1+lv-mu.square()-lv.exp())).mean()
            need(bool(torch.isfinite(loss)),'Nonfinite VAE loss');opt.zero_grad();loss.backward();opt.step();updates+=1
    sync(device);train_sec=time.perf_counter()-t;t=time.perf_counter()
    # Same permutation of *training* auxiliary histories for both generators.
    # The unconditional decoder never sees those histories; its pairing is random.
    indices=np.random.default_rng(seed).permutation(n).astype(np.int64)
    model.eval();out=[]
    with torch.no_grad():
        for ix in np.array_split(indices,math.ceil(n/batch_size)):
            ci=torch.as_tensor(ix,device=device);z=torch.randn(len(ix),latent,device=device)
            out.append(model.decode(z,aux[ci]).cpu().numpy())
    sync(device);sample_sec=time.perf_counter()-t
    synth=np.concatenate(out).astype(np.float32);sx=x[indices].copy();sx[:,:,0]=synth[:,:96];sy=synth[:,96:].copy();checked_pair((sx,sy))
    stats=dict(generator_train_seconds=train_sec,generator_sample_seconds=sample_sec,generator_updates=updates,last_loss=float(loss.detach().cpu()),synthetic_std=float(synth.std()),real_std=float(full.std()),generator_parameters=sum(p.numel() for p in model.parameters()))
    return (sx,sy),indices,stats

class MatchedForecaster(nn.Module):
    """Same 3-token shared encoder/readout; diagonal mask vs unrestricted attention.
    Target-only uses zero auxiliary inputs; no calendar information enters it.
    Independent encoding has no cross-token attention but a common final fusion.
    """
    def __init__(self,mode,width=64,heads=4,layers=2):
        super().__init__();need(mode in MODES,'Unknown mode');self.mode=mode
        self.embed=nn.Linear(96,width);self.channel_id=nn.Parameter(torch.zeros(3,width));nn.init.normal_(self.channel_id,std=.02)
        layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=2*width,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.readout=nn.Sequential(nn.LayerNorm(3*width),nn.Linear(3*width,24))
        self.register_buffer('diagonal_mask',~torch.eye(3,dtype=torch.bool),persistent=False)
    def encode(self,x):
        need(x.ndim==3 and x.shape[1:]==(96,3),'Expected three input channels')
        if self.mode=='target_only':x=torch.cat([x[:,:,:1],torch.zeros_like(x[:,:,1:])],dim=2)
        z=self.embed(x.transpose(1,2))+self.channel_id[None,:,:]
        return self.encoder(z,mask=None if self.mode=='channel_mixing' else self.diagonal_mask)
    def forward(self,x):return self.readout(self.encode(x).flatten(1))

def loader(pair,batch,shuffle):return DataLoader(TensorDataset(*(torch.as_tensor(x,dtype=torch.float32) for x in pair)),batch_size=batch,shuffle=shuffle,drop_last=False)
@torch.no_grad()
def evaluate(model,pair,batch,device,trace=False):
    model.eval();se=ae=0.;count=0;ms=[];ma=[]
    for x,y in loader(pair,batch,False):
        e=model(x.to(device))-y.to(device);se+=e.square().sum().item();ae+=e.abs().sum().item();count+=e.numel()
        if trace:ms.append(e.square().mean(1).cpu().numpy());ma.append(e.abs().mean(1).cpu().numpy())
    need(count>0 and np.isfinite([se,ae]).all(),'Invalid evaluation')
    return se/count,ae/count,(np.concatenate(ms),np.concatenate(ma)) if trace else None

def fit(train,val,test,mode,rep,steps,interval,device,*,batch=64,lr=.001,width=64,heads=4,layers=2):
    need(steps>0 and interval>0,'Bad optimizer budget');seed_all(rep);sync(device);tick=time.perf_counter()
    model=MatchedForecaster(mode,width,heads,layers).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
    dl=loader(train,batch,True);it=iter(dl);best=float('inf');state=None;curve=[];samples=0
    for s in range(1,steps+1):
        try:x,y=next(it)
        except StopIteration:it=iter(dl);x,y=next(it)
        samples+=len(x);model.train();opt.zero_grad();loss=(model(x.to(device))-y.to(device)).square().mean();need(bool(torch.isfinite(loss)),'Nonfinite forecaster loss');loss.backward();opt.step()
        if s%interval==0 or s==steps:
            v,_,_=evaluate(model,val,batch,device);curve.append([s,v])
            if v<best:best=v;best_step=s;state=copy.deepcopy(model.state_dict())
    need(state is not None,'No checkpoint');model.load_state_dict(state);mse,mae,tr=evaluate(model,test,batch,device,True);sync(device)
    metrics=dict(test_mse=mse,test_mae=mae,val_mse=best,best_step=best_step,optimizer_steps=steps,samples_seen=samples,train_windows=len(train[0]),validation_checkpoints=len(curve),validation_curve=json.dumps(curve),train_seconds=time.perf_counter()-tick,forecaster_parameters=sum(p.numel() for p in model.parameters()))
    return metrics,tr

def code_hash():
    files=['scripts/asoc_phase19_conditional.py','scripts/asoc_epias_preflight.py','src/data.py']
    return fingerprint({f:sha(ROOT/f) for f in files})
def preflight(data_path,raw,ref):
    from scripts.asoc_epias_preflight import audit
    from src.data import WindowConfig,prepare,chrono_split
    need(ref.is_file(),'Phase 7 reference required')
    gate=audit(csv_path=raw,npy_path=data_path,runs_path=ref,regime=.1)
    need(gate['pass'],'EPİAŞ preflight failed: '+json.dumps(gate['errors'],ensure_ascii=False))
    a=np.load(data_path,allow_pickle=False);d=prepare(a,.1,WindowConfig(96,24));t1,t2=chrono_split(len(a));n=len(d['train'][0]);steps=10*math.ceil(2*n/64);interval=math.ceil(2*n/64)
    for k in ('train','val','test'):checked_pair(d[k])
    need(steps==gate['split']['matched_steps_per_arm'],'Budget mismatch')
    # The full prepared array and raw file are pinned to the historical data hash.
    r=pd.read_csv(ref);sub=r[(r.domain=='epias') & (r.arch=='itransformer') & (r.input_mode=='target') & np.isclose(r.regime,.1)]
    need(len(sub)>0 and (sub.data_sha256==gate['prepared_sha256']).all(),'Missing/mismatched historical EPİAŞ data')
    need((sub.matched_steps==steps).all() and (sub.real_windows==n).all(),'Historical budget mismatch')
    meta=dict(version=VERSION,config=CONFIG,data_sha256=gate['prepared_sha256'],raw_sha256=gate['source_sha256'],reference_sha256=sha(ref),code_sha256=code_hash(),train_start=int(d['train_start']),train_end=t1,test_start=t2,data_length=len(a),real_windows=n,matched_steps=steps,target_sd=float(d['sd'][0,0]))
    return d,meta,np.arange(t2,len(a)-24+1,dtype=np.int64),gate

def env(device):return dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__)
def pool_path(root,arm):return root/'pools'/f'{arm}.npz'
def load_pool(root,arm,real,cid):
    path=pool_path(root,arm);need(path.is_file(),'Pool missing '+str(path))
    with np.load(path,allow_pickle=False) as q:
        pair=(q['x'],q['y']);idx=q['indices'];meta=json.loads(str(q['metadata'].item()));stats=json.loads(str(q['stats'].item()));checked_pair(pair)
        need(meta==dict(config_id=cid,arm=arm,pool_seed=0),'Pool metadata mismatch');need(digest((*pair,idx))==str(q['content_sha256'].item()),'Pool content mismatch')
    need(idx.dtype==np.int64 and np.array_equal(np.sort(idx),np.arange(len(real[0]))) and np.array_equal(pair[0][:,:,1:],real[0][idx,:,1:]),'Aux pairing mismatch')
    return pair,idx,stats,sha(path)
def ensure_pool(root,arm,real,cid,device):
    path=pool_path(root,arm)
    if not path.exists():
        pair,idx,stats=generate_pool(real,arm=='conditional_vae',device,epochs=CONFIG['gen_epochs'])
        atomic_npz(path,x=pair[0],y=pair[1],indices=idx,metadata=json.dumps(dict(config_id=cid,arm=arm,pool_seed=0)),stats=json.dumps(stats,allow_nan=False),content_sha256=digest((*pair,idx)))
    return load_pool(root,arm,real,cid)
def expected():return [(m,a,r) for m in MODES for a in ARMS for r in REPS]
def run_path(root):return root/'runs.csv'
def read_runs(root,meta,idx,real):
    path=run_path(root);df=pd.read_csv(path) if path.exists() else pd.DataFrame();cid=fingerprint(meta)
    if df.empty:return df
    need(not df.duplicated(['mode','arm','rep']).any() and set(df.config_id)=={cid},'Run key/config mismatch; use new output directory')
    need(set(zip(df["mode"],df["arm"],df["rep"])).issubset(set(expected())),'Unknown run key')
    for r in df.itertuples():
        need(r.optimizer_steps==meta['matched_steps'] and r.data_sha256==meta['data_sha256'] and r.raw_sha256==meta['raw_sha256'],'Budget/data mismatch')
        need(r.pool_seed==(0 if r.arm.endswith('vae') else -1),'Pool seed mismatch')
        curve=np.asarray(json.loads(r.validation_curve));need(curve.shape==(r.validation_checkpoints,2) and curve[-1,0]==r.optimizer_steps and curve[np.argmin(curve[:,1]),0]==r.best_step and np.isclose(curve[:,1].min(),r.val_mse),'Invalid validation checkpoint')
        path=root/'origins'/f'{r.mode}_{r.arm}_{r.rep}.npz';need(sha(path)==r.origin_sha256,'Origin hash mismatch')
        with np.load(path,allow_pickle=False) as q:
            need(np.array_equal(q['origin'],idx),'Origin index mismatch');need(q['mse'].shape==q['mae'].shape==idx.shape and np.isfinite(q['mse']).all() and np.isfinite(q['mae']).all(),'Invalid origin loss')
            need(np.isclose(q['mse'].mean(dtype=np.float64),r.test_mse,rtol=1e-6) and np.isclose(q['mae'].mean(dtype=np.float64),r.test_mae,rtol=1e-6),'Origin aggregate mismatch')
        if r.arm.endswith('vae'):
            _,_,_,h=load_pool(root,r.arm,real,cid);need(h==r.pool_sha256,'Pool SHA mismatch')
    return df

def report(root,df):
    need(len(df)==24,'Complete all 24 runs before reporting');rows=[]
    for (mode,rep),g in df.groupby(['mode','rep']):
        b=g.set_index('arm')
        for arm in ('unconditional_vae','conditional_vae'):
            for ref in ('step_baseline','duplicate_real')+ (('unconditional_vae',) if arm=='conditional_vae' else ()):
                rows.append(dict(mode=mode,rep=rep,arm=arm,reference=ref,mse=b.loc[arm,'test_mse'],ref_mse=b.loc[ref,'test_mse'],mae=b.loc[arm,'test_mae'],ref_mae=b.loc[ref,'test_mae'],delta_mse_pct=100*(b.loc[arm,'test_mse']/b.loc[ref,'test_mse']-1),delta_mae_pct=100*(b.loc[arm,'test_mae']/b.loc[ref,'test_mae']-1)))
    d=pd.DataFrame(rows);atomic_csv(d,root/'report/paired.csv');summary=d.groupby(['mode','arm','reference'],as_index=False).agg(n=('rep','size'),median_delta_mse_pct=('delta_mse_pct','median'),median_delta_mae_pct=('delta_mae_pct','median'));atomic_csv(summary,root/'report/summary.csv')
    # Signed difference-in-differences, in normalized MSE (not a causal generalization).
    contrast=[]
    for rep in REPS:
        p=df[df.rep==rep].pivot(index='mode',columns='arm',values='test_mse')
        for arm in ('unconditional_vae','conditional_vae'):
            for baseline in ('step_baseline','duplicate_real'):
                i=p.loc['independent_fusion',arm]-p.loc['independent_fusion',baseline];m=p.loc['channel_mixing',arm]-p.loc['channel_mixing',baseline]
                contrast.append(dict(rep=rep,arm=arm,baseline=baseline,independent_delta_mse=i,mixing_delta_mse=m,mixing_minus_independent=m-i))
    atomic_csv(pd.DataFrame(contrast),root/'report/interaction.csv')
    arch=[]
    for (rep,arm),g in df.groupby(['rep','arm']):
        q=g.set_index('mode');t=q.loc['target_only','test_mse'];i=q.loc['independent_fusion','test_mse'];m=q.loc['channel_mixing','test_mse']
        arch.append(dict(rep=rep,arm=arm,target_only_mse=t,independent_fusion_mse=i,channel_mixing_mse=m,aux_access_delta=i-t,mixing_delta=m-i))
    atomic_csv(pd.DataFrame(arch),root/'report/architecture.csv');print(summary.to_string(index=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/real/epias.npy'));p.add_argument('--raw',type=Path,default=Path('data/real/epias/ptf.csv'));p.add_argument('--reference',type=Path,default=Path('results/asoc_revision/real_budget_controls.csv'));p.add_argument('--outdir',type=Path,default=Path('results/asoc_revision/phase19_conditional'));p.add_argument('--gpu-index',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--dry-run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true');a=p.parse_args()
    def device():
        if torch.cuda.is_available():need(0<=a.gpu_index<torch.cuda.device_count(),'Invalid GPU');torch.cuda.set_device(a.gpu_index);return torch.device(f'cuda:{a.gpu_index}')
        need(a.allow_cpu,'CUDA unavailable (use --allow-cpu only for smoke)');return torch.device('cpu')
    if a.smoke:
        dev=device();rng=np.random.default_rng(7);real=(rng.normal(size=(16,96,3)).astype('float32'),rng.normal(size=(16,24)).astype('float32'))
        for arm in ('unconditional_vae','conditional_vae'):
            pair,_,stats=generate_pool(real,arm=='conditional_vae',dev,epochs=1,batch_size=8);print('SMOKE pool',arm,pair[0].shape,stats)
        for m in MODES:
            met,_=fit(real,real,real,m,0,2,1,dev,batch=4);print('SMOKE fit',m,met['test_mse'])
        print('SMOKE PASS');return
    d,meta,idx,gate=preflight(resolve(a.data),resolve(a.raw),resolve(a.reference));root=resolve(a.outdir);cid=fingerprint(meta);real=d['train'];df=read_runs(root,meta,idx,real);done=set(zip(df["mode"],df["arm"],df["rep"])) if len(df) else set();pending=[x for x in expected() if x not in done]
    print('EPİAŞ preflight PASS',json.dumps(gate['split'],ensure_ascii=False));print('config',cid,'pending fits',len(pending),'matched_steps',meta['matched_steps']);print('Raw SHA',meta['raw_sha256'],'prepared SHA',meta['data_sha256'])
    if a.dry_run:return
    if a.report:report(root,df);return
    need(a.limit is None or a.limit>0,'Limit must be positive');dev=device();e=json.dumps(env(dev),sort_keys=True);need(not len(df) or set(df.environment)=={e},'Environment changed; use new output directory')
    root.mkdir(parents=True,exist_ok=True);cfgfile=root/'protocol.json'
    if cfgfile.exists():need(json.loads(cfgfile.read_text(encoding='utf8'))==meta,'Protocol mismatch')
    else:cfgfile.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    runtime=root/'environment.json'
    if runtime.exists():need(json.loads(runtime.read_text(encoding='utf8'))==json.loads(e),'Environment changed; use new output directory')
    else:runtime.write_text(e,encoding='utf8')
    for mode,arm,rep in pending[:a.limit]:
        h='';stats={};pair=real
        if arm.endswith('vae'):pair,_,stats,h=ensure_pool(root,arm,real,cid,dev)
        train=real if arm=='step_baseline' else tuple(np.concatenate([x,y],axis=0) for x,y in zip(real,pair))
        met,(ms,ma)=fit(train,d['val'],d['test'],mode,rep,meta['matched_steps'],math.ceil(2*len(real[0])/64),dev)
        path=root/'origins'/f'{mode}_{arm}_{rep}.npz';atomic_npz(path,origin=idx,mse=ms,mae=ma)
        row=dict(version=VERSION,config_id=cid,data_sha256=meta['data_sha256'],raw_sha256=meta['raw_sha256'],mode=mode,arm=arm,rep=rep,pool_seed=0 if h else -1,environment=e,pool_sha256=h,origin_sha256=sha(path),generator_stats=json.dumps(stats,allow_nan=False),**met,mse_raw=met['test_mse']*meta['target_sd']**2,mae_raw=met['test_mae']*meta['target_sd'])
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True);atomic_csv(df,run_path(root));print(mode,arm,rep,'MSE',met['test_mse'],'MAE',met['test_mae'],flush=True)
if __name__=='__main__':main()
