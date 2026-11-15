"""Post-review last-value-centred forecaster stress test. Additive; no historical writes."""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('results/asoc_revision/phase20_adequacy')
VERSION = 'phase20_last_value_target_v1'
HASHES = {'weather':'bf5014da5724795b91ce862de5245018568064b86daa5ad49b207ecb76b81a03',
          'electricity':'c5fcd66ef85a284a7227fb4373a76e9c6d7469eed5f8ed6412598c8d981927ab'}
WEATHER_POOLS = {'vae': ('05b7f6c898c7eb2492448633bb876480d2da20305b808247c6462e32ba3df373','b2f3e314dd29932296872f41b26765fb37a2f0e5c03b0e4fb4594e586f989cdc'),
                 'timegan': ('b254bb606a5862292bab416dc3f3f4418dc322adc49bc5f56e5befb84f866f09','af01578131208f25ef5f1115e239aa896a5cc8381f121df5dbfeaf713dee6524')}
CONFIG = dict(regime=.1,lookback=96,horizon=24,batch_size=64,epochs=10,lr=.001,model='itransformer_lite_target_last_value',reps=[0,1,2],pool_seeds=[0,1],ridge_alpha=1.)

class LastValue(nn.Module):
    """Same forecaster parameters; only target history centering and output restoration."""
    def __init__(self, base):
        super().__init__(); self.base=base
    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=1: raise ValueError('Expected target-only B,L,1')
        last=x[:,-1:,:]
        return self.base(x-last)+last[:,0,:]

def factory(arch,L,H):
    from src.models import build
    return LastValue(build(arch,L,H))

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sha(p):
    with Path(p).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def fp(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def resolve(p): return p if p.is_absolute() else ROOT/p

def jobs():
    # Fixed pilot: rep0/pool0, then all remaining jobs irrespective of outcome.
    all_jobs=[]
    for rep in (0,1,2):
        for d in ('weather','electricity'):
            all_jobs += [(d,rep,a,-1) for a in ('step_baseline','duplicate_real')]
            all_jobs += [(d,rep,a,p) for p in (0,1) for a in (('vae','timegan') if d=='weather' else ('vae',))]
    pilot=[j for j in all_jobs if j[1]==0 and j[3] in (-1,0)]
    return pilot+[j for j in all_jobs if j not in pilot]

def valid_pair(pair,real):
    need(all(x.shape==y.shape and x.dtype==np.float32 and np.isfinite(x).all() for x,y in zip(pair,real)), 'Invalid pool shapes/dtype/values')

def references(train,val,test):
    # Direct 96-to-24 ridge with unpenalized intercept, train only; no search.
    x,y=train; X=np.column_stack([np.ones(len(x)),x[:,:,0]]).astype(np.float64)
    A=X.T@X; A[1:,1:]+=np.eye(X.shape[1]-1); coef=np.linalg.solve(A,X.T@y.astype(np.float64))
    out={}
    for label,(a,b) in [('validation',val),('test',test)]:
        for name,pred in [('persistence',np.repeat(a[:,-1,0,None],b.shape[1],axis=1)),('ridge96',np.column_stack([np.ones(len(a)),a[:,:,0]])@coef)]:
            e=pred.astype(np.float64)-b;out[label+'_'+name+'_mse']=float(np.mean(e**2));out[label+'_'+name+'_mae']=float(np.mean(abs(e)))
    return out

def preflight(datadir,ref):
    from src.data import WindowConfig,prepare,chrono_split
    from scripts.asoc_budget_controls import only_channels,count_steps
    from scripts.asoc_real_timegan import select_refs
    plans={}
    for d in HASHES:
        p=datadir/f'{d}.npy';need(sha(p)==HASHES[d],f'{d} prepared data SHA mismatch; do not replace source')
        a=np.load(p,allow_pickle=False);need(a.ndim==2 and a.shape[1]==3 and a.dtype==np.float32 and np.isfinite(a).all(),'Invalid source array')
        t1,t2=chrono_split(len(a)); data=prepare(a,.1,WindowConfig(96,24)); start=int(data['train_start']);n=len(data['train'][0]);orig,steps=count_steps(n,n,64,10)
        need(not np.any(a[start:,0]==-9999),f'{d}: active sentinel')
        raw=datadir/d/('weather.csv' if d=='weather' else 'ETTh1.csv'); rawdf=pd.read_csv(raw,usecols=['date','OT']);dates=pd.to_datetime(rawdf.date,errors='raise');active=dates.iloc[start:]
        need(len(rawdf)==len(a) and np.array_equal(rawdf.OT.to_numpy(dtype=np.float32),a[:,0]),f'{d} raw target differs from prepared data')
        need(not active.duplicated().any() and (active.diff().dropna()==pd.Timedelta(minutes=10 if d=='weather' else 60)).all(),f'{d} active timestamp cadence mismatch')
        meta=dict(data_length=len(a),train_start=start,train_end=t1,real_windows=n,val_windows=len(data['val'][0]),test_windows=len(data['test'][0]),target_mean=float(data['mu'][0,0]),target_sd=float(data['sd'][0,0]),original_steps=orig,matched_steps=steps)
        select_refs(ref,d,HASHES[d],meta,(0,1,2),(0,1))
        pairs={k:only_channels(data[k],'target') for k in ('train','val','test')}
        plans[d]=dict(meta=meta,pairs=pairs,raw_sha=sha(raw),test_start=t2,refs=references(**pairs),array=a)
    return plans

def weather_pool(results,arm,ps,plan):
    root=results/('phase15_origins' if arm=='vae' else 'phase14_timegan')
    rows=pd.read_csv(root/'runs.csv'); rows=rows[(rows.domain=='weather')&(rows.pool_seed==ps)]
    if arm=='vae': rows=rows[rows.condition=='vae_replay']
    need(len(rows)==3 and set(rows.rep)=={0,1,2},'Weather historical pool records incomplete')
    need(set(rows.data_sha256)=={HASHES['weather']} and set(rows.raw_sha256)=={plan['raw_sha']},'Weather pool data provenance mismatch')
    need(rows.pool_file.nunique()==1 and set(rows.pool_sha256)=={WEATHER_POOLS[arm][ps]},'Weather pool log mismatch')
    path=root/str(rows.pool_file.iloc[0]);need(sha(path)==WEATHER_POOLS[arm][ps],'Weather pool bytes changed')
    with np.load(path,allow_pickle=False) as z: pair=(z['x'],z['y'])
    valid_pair(pair,plan['pairs']['train']);return pair,path,sha(path)

def electricity_pool(root,ps,plan,cid):
    from src.data import WindowConfig,standardize
    from src.generators import gen_vae
    from scripts.asoc_budget_controls import only_channels
    from scripts.asoc_real_timegan import atomic_npz,array_digest
    path=root/'pools'/f'electricity_vae_p{ps}.npz'; real=plan['pairs']['train']; meta=dict(config_id=cid,pool_seed=ps,kind='new_vae_pool',data_sha256=HASHES['electricity'])
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            pair=(z['x'],z['y']);need(json.loads(str(z['metadata'].item()))==meta and str(z['content_sha256'].item())==array_digest(pair),'ETTh1 pool mismatch')
    else:
        m=plan['meta'];xs,_,_=standardize(plan['array'],slice(m['train_start'],m['train_end']));pair=only_channels(gen_vae(xs[m['train_start']:m['train_end']],WindowConfig(96,24),ratio=1.,seed=ps,epochs=30,latent=16),'target');valid_pair(pair,real)
        atomic_npz(path,x=pair[0],y=pair[1],metadata=json.dumps(meta,sort_keys=True),content_sha256=array_digest(pair))
    valid_pair(pair,real);return pair,path,sha(path)

def report(df,root):
    from scripts.asoc_real_timegan import atomic_csv
    rows=[]
    for r in df[df.arm.isin(['vae','timegan'])].itertuples():
        for c in ('step_baseline','duplicate_real')+(('vae',) if r.arm=='timegan' else ()):
            q=df[(df.domain==r.domain)&(df.rep==r.rep)&(df.arm==c)]
            if c=='vae': q=q[q.pool_seed==r.pool_seed]
            if len(q)!=1: continue
            b=q.iloc[0];rows.append(dict(domain=r.domain,rep=r.rep,pool_seed=r.pool_seed,arm=r.arm,reference=c,delta_mse_pct=100*(r.test_mse/b.test_mse-1),delta_mae_pct=100*(r.test_mae/b.test_mae-1),augmentation_mse=r.test_mse,reference_mse=b.test_mse,persistence_mse=r.test_persistence_mse,ridge_mse=r.test_ridge96_mse,control_beats_persistence=bool(b.test_mse<r.test_persistence_mse),augmentation_beats_persistence=bool(r.test_mse<r.test_persistence_mse),control_beats_ridge=bool(b.test_mse<r.test_ridge96_mse),augmentation_beats_ridge=bool(r.test_mse<r.test_ridge96_mse)))
    if not rows: return
    p=pd.DataFrame(rows);atomic_csv(p,root/'report/paired.csv');cols=['delta_mse_pct','delta_mae_pct'];g=p.groupby(['domain','arm','reference','pool_seed'])[cols].median().reset_index();atomic_csv(g,root/'report/per_pool.csv');s=g.groupby(['domain','arm','reference'])[cols].median().reset_index();atomic_csv(s,root/'report/summary.csv');print(s.to_string(index=False));print(f'Completed {len(df)}/30; partial summaries are pilot-only, not final.')

def main():
    from scripts.asoc_real_timegan import fit_with_origins,atomic_npz,atomic_csv
    from scripts.asoc_budget_controls import select_device
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data/real'));p.add_argument('--results-dir',type=Path,default=Path('results/asoc_revision'));p.add_argument('--gpu-index',type=int,default=0);p.add_argument('--limit',type=int,default=0);p.add_argument('--dry-run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true');a=p.parse_args();need(a.limit>=0,'Negative limit')
    if a.smoke:
        dev=select_device(allow_cpu=a.allow_cpu,gpu_index=a.gpu_index);rng=np.random.default_rng(20);x=rng.normal(size=(12,96,1)).astype('float32');y=rng.normal(size=(12,24)).astype('float32');m,ms,ma=fit_with_origins((x,y),(x,y),(x,y),0,2,1,dev,batch_size=4,model_factory=factory);need(np.isfinite(ms).all() and np.isfinite(ma).all(),'Smoke nonfinite');print('SMOKE PASS',m);return
    results=resolve(a.results_dir);root=results/'phase20_adequacy';refpath=results/'real_budget_controls.csv';plans=preflight(resolve(a.data_dir),pd.read_csv(refpath))
    # Read-only validation of ALL historical pools before any training.
    wp={(arm,ps):weather_pool(results,arm,ps,plans['weather']) for arm in ('vae','timegan') for ps in (0,1)}
    deps=['scripts/asoc_phase20_adequacy.py','scripts/asoc_real_timegan.py','scripts/asoc_budget_controls.py','src/data.py','src/models.py','src/generators.py']
    protocol=dict(version=VERSION,config=CONFIG,data_sha256=HASHES,raw_sha256={d:v['raw_sha'] for d,v in plans.items()},reference_sha256=sha(refpath),source_sha256={f:sha(ROOT/f) for f in deps},weather_pool_sha256=WEATHER_POOLS,jobs=jobs())
    # Serialize tuples as lists for stable JSON comparison.
    protocol=json.loads(json.dumps(protocol));cid=fp(protocol);out=root/'runs.csv';df=pd.read_csv(out,keep_default_na=False) if out.exists() else pd.DataFrame();done=set()
    if len(df):
        need(set(df.config_id)=={cid} and not df.duplicated(['domain','rep','arm','pool_seed']).any(),'Existing run config/key mismatch')
        for r in df.itertuples():
            key=(r.domain,int(r.rep),r.arm,int(r.pool_seed));need(key in jobs(),'Unknown job');done.add(key);plan=plans[r.domain];need(r.optimizer_steps==plan['meta']['matched_steps'] and r.data_sha256==HASHES[r.domain] and r.raw_sha256==plan['raw_sha'],'Stored budget/data mismatch');curve=np.asarray(json.loads(r.validation_curve));need(curve.shape==(10,2) and curve[-1,0]==r.optimizer_steps and curve[np.argmin(curve[:,1]),0]==r.best_step and np.isclose(curve[:,1].min(),r.val_mse),'Stored validation curve mismatch');path=root/r.origin_file;need(sha(path)==r.origin_sha256,'Origin SHA mismatch')
            with np.load(path,allow_pickle=False) as z:
                idx=np.arange(plan['test_start'],plan['test_start']+plan['meta']['test_windows']);need(np.array_equal(z['origin'],idx) and z['mse'].shape==z['mae'].shape==idx.shape and np.isfinite(z['mse']).all() and np.isfinite(z['mae']).all() and np.isclose(z['mse'].mean(),r.test_mse,rtol=1e-6) and np.isclose(z['mae'].mean(),r.test_mae,rtol=1e-6),'Origin mismatch')
            if r.arm in ('vae','timegan'): need(sha(results/r.pool_file)==r.pool_sha256,'Stored pool mismatch')
    pending=[j for j in jobs() if j not in done]; print('CONFIG',cid,'completed',len(done),'pending',len(pending));
    for d,plan in plans.items(): print(d,plan['meta'],plan['refs'])
    if a.dry_run: return
    if a.report: need(len(df)>0,'No completed runs');report(df,root);return
    if not pending: report(df,root);return
    dev=select_device(gpu_index=a.gpu_index);need(dev.type=='cuda','CUDA required for real runs');env=dict(gpu=torch.cuda.get_device_name(dev),torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__);envstr=json.dumps(env,sort_keys=True);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): need(json.loads((root/'protocol.json').read_text())==protocol,'Protocol changed')
    else: (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if (root/'environment.json').exists(): need(json.loads((root/'environment.json').read_text())==env,'Environment changed; use same GPU/software')
    else: (root/'environment.json').write_text(json.dumps(env,indent=2))
    for d,rep,arm,ps in pending[:a.limit or None]:
        plan=plans[d];m=plan['meta'];pairs=plan['pairs'];real=pairs['train'];path=None;ph=''
        if arm=='step_baseline':train=real
        else:
            if arm=='duplicate_real':extra=real
            elif d=='weather':extra,path,ph=wp[(arm,ps)]
            else:extra,path,ph=electricity_pool(root,ps,plan,cid)
            train=tuple(np.concatenate([x,y],axis=0) for x,y in zip(real,extra))
        met,ms,ma=fit_with_origins(train,pairs['val'],pairs['test'],rep,m['matched_steps'],m['matched_steps']//10,dev,model_factory=factory)
        idx=np.arange(plan['test_start'],plan['test_start']+m['test_windows']);need(ms.shape==ma.shape==idx.shape and np.isclose(ms.mean(),met['test_mse'],rtol=1e-6) and np.isclose(ma.mean(),met['test_mae'],rtol=1e-6),'Origin aggregation mismatch')
        op=root/'origins'/f'{d}_{arm}_r{rep}_p{ps}.npz';atomic_npz(op,origin=idx,mse=ms,mae=ma)
        row=dict(config_id=cid,version=VERSION,domain=d,rep=rep,arm=arm,pool_seed=ps,data_sha256=HASHES[d],raw_sha256=plan['raw_sha'],environment=envstr,pool_file=path.relative_to(results).as_posix() if path else '',pool_sha256=ph,origin_file=op.relative_to(root).as_posix(),origin_sha256=sha(op),**m,**met,**plan['refs'],mse_raw=met['test_mse']*m['target_sd']**2,mae_raw=met['test_mae']*m['target_sd'])
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True);atomic_csv(df,out);print(d,rep,arm,ps,'MSE',met['test_mse'],'persistence',plan['refs']['test_persistence_mse'],flush=True)
    report(df,root)

if __name__=='__main__': main()
