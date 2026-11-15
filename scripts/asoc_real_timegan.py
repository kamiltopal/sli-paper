"""Phase 14: fixed-protocol compact TimeGAN, read-only Phase 7 reference.
Run from repo root. All new files under a separate phase14_timegan directory.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.asoc_timegan import generate_pool,sync
ROOT=Path(__file__).resolve().parents[1]
VERSION='phase14_timegan_target_v1'
BASE=Path('results/asoc_revision/phase14_timegan')
REF=Path('results/asoc_revision/real_budget_controls.csv')
CONTROLS=('epoch_baseline','step_baseline','duplicate_real')
DOMAINS=('weather','traffic')
CONFIG=dict(regime=.1,arch='itransformer',input_mode='target',lookback=96,horizon=24,epochs=10,batch_size=64,lr=.001,vae_epochs=30,latent=16,hidden=128,horizon_weight=.5)
GEN=dict(width=24,layers=3,batch_size=64,lr=.001,ae_steps=200,sup_steps=200,joint_steps=400,sample_batch=128)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def fingerprint(d):return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()[:24]
def resolve(p):return p if p.is_absolute() else ROOT/p

def select_refs(df,domain,digest,meta,reps,pools):
    required=set(CONFIG)|set(meta)|{'experiment_version','domain','data_sha256','rep','pool_seed','condition','optimizer_steps','test_mse','test_mae','mse_raw','mae_raw','normalization','val_checkpoint_selection'}
    if required-set(df):raise ValueError('Missing reference columns: '+str(sorted(required-set(df))))
    f=df[(df.domain==domain)&(df.experiment_version=='phase7_real_equal_step_v1')].copy()
    for k,v in CONFIG.items(): f=f[np.isclose(f[k],v,rtol=0,atol=1e-10) if isinstance(v,float) else f[k]==v]
    if f.empty or not (f.data_sha256==digest).all():raise ValueError(f'{domain}: missing protocol or data hash mismatch')
    refs={}
    for rep in reps:
        for c,ps in [(c,-1) for c in CONTROLS]+[('vae_augmented',s) for s in pools]:
            a=f[(f.rep==rep)&(f.condition==c)&(f.pool_seed==ps)]
            if len(a)!=1:raise ValueError(f'{domain}: expected one reference for {rep}/{c}/{ps}, got {len(a)}')
            row=a.iloc[0]
            for k,v in meta.items():
                if not np.isclose(float(row[k]),float(v),rtol=1e-6,atol=1e-8):raise ValueError(f'{domain}: reference {k} mismatch')
            if row.optimizer_steps!=meta['original_steps' if c=='epoch_baseline' else 'matched_steps']:raise ValueError('Reference optimizer steps mismatch')
            if row.normalization!='per_regime_train_segment_only' or row.val_checkpoint_selection!='minimum_validation_mse':raise ValueError('Reference normalization/checkpoint mismatch')
            if not all(np.isfinite(row[k]) and row[k]>0 for k in ('test_mse','test_mae','mse_raw','mae_raw')):raise ValueError('Invalid reference metrics')
            if not np.isclose(row.mse_raw,row.test_mse*meta['target_sd']**2,rtol=1e-5) or not np.isclose(row.mae_raw,row.test_mae*meta['target_sd'],rtol=1e-5):raise ValueError('Reference raw scale mismatch')
            refs[(rep,c,ps)]=row
    return refs

def prepare_domain(domain,data_dir,ref,reps,pools):
    from src.data import WindowConfig,prepare,chrono_split,apply_regime
    from scripts.asoc_budget_controls import count_steps,only_channels
    p=data_dir/f'{domain}.npy';digest=sha(p);a=np.load(p,allow_pickle=False).astype('float32')
    if a.ndim!=2 or a.shape[1]<1 or not np.isfinite(a).all():raise ValueError('Invalid prepared array')
    cfg=WindowConfig(96,24);d=prepare(a,.1,cfg);t1,t2=chrono_split(len(a));n=len(d['train'][0]);start=apply_regime(t1,.1,96)
    orig,matched=count_steps(n,n,64,10)
    meta=dict(data_length=len(a),train_start=start,train_end=t1,real_windows=n,val_windows=len(d['val'][0]),test_windows=len(d['test'][0]),target_mean=float(d['mu'][0,0]),target_sd=float(d['sd'][0,0]),original_steps=orig,matched_steps=matched)
    raw=data_dir/domain/f'{domain}.csv'
    if not raw.is_file():raise FileNotFoundError(f'Raw timestamp CSV required: {raw}')
    dates=pd.to_datetime(pd.read_csv(raw,usecols=['date'])['date'],errors='coerce')
    active=dates.iloc[start:]
    if len(dates)!=len(a) or active.isna().any() or active.duplicated().any() or not (active.diff().dropna()==pd.Timedelta(minutes=10 if domain=='weather' else 60)).all():raise ValueError(f'{domain}: active timestamp cadence/length mismatch')
    refs=select_refs(ref,domain,digest,meta,reps,pools)
    return digest,meta,{k:only_channels(d[k],'target') for k in ('train','val','test')},refs,t2,sha(raw)

def atomic_npz(path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**kw)
    os.replace(tmp,path)
def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);os.replace(tmp,path)

def array_digest(pair):
    h=hashlib.sha256()
    for x in pair:
        a=np.ascontiguousarray(x);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def pool_for(root,real,metadata,device):
    path=root/'pools'/f"{metadata['domain']}_{fingerprint(metadata)}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['metadata'].item()))!=metadata:raise ValueError('Pool metadata mismatch')
            pair=(z['x'],z['y']);stats=json.loads(str(z['stats'].item()))
            if str(z['content_sha256'].item())!=array_digest(pair):raise ValueError('Cached pool content hash mismatch')
        if not all(a.shape==b.shape and np.isfinite(a).all() for a,b in zip(pair,real)):raise ValueError('Invalid cached pool')
    else:
        pair,stats=generate_pool(real,seed=metadata['pool_seed'],device=device,**metadata['generator'])
        stats.update(generator_device=str(device),generator_gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',generator_torch_version=str(torch.__version__),generator_cuda_version=torch.version.cuda or 'none')
        atomic_npz(path,x=pair[0],y=pair[1],metadata=json.dumps(metadata,sort_keys=True),stats=json.dumps(stats,allow_nan=False),content_sha256=array_digest(pair))
    return pair,stats,path

def fit_with_origins(train_pair,val_pair,test_pair,rep,steps,interval,device,*,batch_size=64,lr=.001,model_factory=None):
    """Same initialization, loader, optimizer, validation and checkpoint policy as Phase 7.
    The final test pass additionally stores the per-origin mean squared/absolute losses.
    """
    from scripts.asoc_budget_controls import seed_everything,_loader,_evaluate
    if model_factory is None:
        from src.models import build
        model_factory=build
    seed_everything(rep);model=model_factory('itransformer',train_pair[0].shape[1],train_pair[1].shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr);loader=_loader(train_pair,batch_size,True);it=iter(loader)
    best=float('inf');state=None;curve=[];samples=0;sync(device);tick=time.perf_counter()
    for s in range(1,steps+1):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        samples+=len(x);x,y=x.to(device),y.to(device);model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.mse_loss(model(x),y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forecaster loss')
        loss.backward();opt.step()
        if s%interval==0 or s==steps:
            v,_=_evaluate(model,val_pair,batch_size,device);curve.append([s,v])
            if v<best:best=v;best_step=s;state=copy.deepcopy(model.state_dict())
    if state is None:raise FloatingPointError('No valid validation checkpoint')
    model.load_state_dict(state);model.eval();squared=[];absolute=[];se=ae=0.;count=0
    with torch.no_grad():
        for x,y in _loader(test_pair,batch_size,False):
            residual=model(x.to(device))-y.to(device)
            se+=residual.square().sum().item();ae+=residual.abs().sum().item();count+=residual.numel()
            squared.append(residual.square().mean(1).cpu().numpy());absolute.append(residual.abs().mean(1).cpu().numpy())
    sync(device);seconds=time.perf_counter()-tick
    ms=np.concatenate(squared);ma=np.concatenate(absolute)
    if not np.isfinite(ms).all() or not np.isfinite(ma).all():raise FloatingPointError('Nonfinite test errors')
    return dict(test_mse=se/count,test_mae=ae/count,val_mse=best,best_step=best_step,optimizer_steps=steps,samples_seen=samples,train_windows=len(train_pair[0]),validation_checkpoints=len(curve),validation_curve=json.dumps(curve),train_seconds=seconds),ms,ma

def validate_existing(df,root,domain,digest,meta,config_id):
    if df.empty:return
    if df.duplicated(['domain','rep','pool_seed']).any():raise ValueError('Duplicate Phase 14 keys')
    for _,r in df[df.domain==domain].iterrows():
        if r.data_sha256!=digest or r.config_id!=config_id or r.experiment_version!=VERSION or r.optimizer_steps!=meta['matched_steps']:raise ValueError('Phase 14 protocol mismatch; use separate output directory')
        for name in ('pool','origin'):
            p=root/str(r[f'{name}_file'])
            if not p.is_file() or sha(p)!=r[f'{name}_sha256']:raise ValueError(f'{name} hash mismatch: {p}')
        with np.load(root/str(r.origin_file),allow_pickle=False) as z:
            if len(z['origin'])!=meta['test_windows'] or not np.array_equal(z['origin'],np.arange(meta.get('test_start',int(z['origin'][0])),meta.get('test_start',int(z['origin'][0]))+meta['test_windows'])) or len(z['mse'])!=meta['test_windows'] or len(z['mae'])!=meta['test_windows'] or not np.isclose(z['mse'].mean(),r.test_mse,rtol=1e-6) or not np.isclose(z['mae'].mean(),r.test_mae,rtol=1e-6):raise ValueError('Origin loss mismatch')

def report(df,refs,root,domains,reps,pools):
    pairs=[];missing=[]
    for domain in domains:
        for ps in pools:
            for rep in reps:
                f=df[(df.domain==domain)&(df.rep==rep)&(df.pool_seed==ps)] if not df.empty else df
                if len(f)!=1:missing.append(dict(domain=domain,rep=rep,pool_seed=ps));continue
                row=f.iloc[0];d=dict(domain=domain,rep=rep,pool_seed=ps,matched_steps=int(row.optimizer_steps))
                for c in ('epoch_baseline','step_baseline','duplicate_real','vae_augmented'):
                    r=refs[domain][(rep,c,ps if c=='vae_augmented' else -1)]
                    for metric in ('test_mse','test_mae','mse_raw','mae_raw'):
                        v=float(row[metric]);b=float(r[metric]);d[f'timegan_{metric}']=v;d[f'{c}_{metric}']=b;d[f'timegan_vs_{c}_{metric}_pct']=100*(v/b-1)
                d['beats_both']=d['timegan_vs_step_baseline_test_mse_pct']<0 and d['timegan_vs_duplicate_real_test_mse_pct']<0
                d['beats_vae']=d['timegan_vs_vae_augmented_test_mse_pct']<0;pairs.append(d)
    f=pd.DataFrame(pairs);atomic_csv(pd.DataFrame(missing,columns=['domain','rep','pool_seed']),root/'report/incomplete.csv')
    if f.empty:raise ValueError('No completed comparisons')
    atomic_csv(f,root/'report/paired.csv');cols=[c for c in f if c.endswith('_pct')]
    p=f.groupby(['domain','pool_seed'])[cols].median().reset_index();atomic_csv(p,root/'report/per_pool.csv')
    s=p.groupby('domain')[cols].median().reset_index();s=s.merge(f.groupby('domain').agg(n_pairs=('rep','size'),beats_both=('beats_both','sum'),beats_vae=('beats_vae','sum')).reset_index(),on='domain');atomic_csv(s,root/'report/summary.csv')
    print(s.to_string(index=False));print('Missing pairs:',len(missing),'(descriptive; no independent-test CI)')
    if missing:raise ValueError('Requested report incomplete; finish pending runs')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domains',default='weather,traffic');p.add_argument('--reps',default='0,1,2');p.add_argument('--pool-seeds',default='0,1');p.add_argument('--data-dir',type=Path,default=Path('data/real'));p.add_argument('--reference',type=Path,default=REF);p.add_argument('--outdir',type=Path,default=BASE);p.add_argument('--gpu-index',type=int,default=0);p.add_argument('--allow-cpu',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true',help='separate tiny synthetic CPU/GPU generator test; no real experiment outputs')
    a=p.parse_args()
    try:
        from scripts.asoc_budget_controls import select_device
        from scripts.asoc_real_controls import parse_ints
        if a.smoke:
            dev=select_device(allow_cpu=a.allow_cpu,gpu_index=a.gpu_index);r=np.random.default_rng(0);x=r.normal(size=(16,96,1)).astype('float32');y=r.normal(size=(16,24)).astype('float32');pair,stats=generate_pool((x,y),seed=0,device=dev,width=4,layers=1,batch_size=4,ae_steps=2,sup_steps=2,joint_steps=2,sample_batch=8);metrics,ms,ma=fit_with_origins(pair,(x,y),(x,y),0,2,1,dev,batch_size=4);print('SMOKE PASS',pair[0].shape,pair[1].shape,json.dumps(stats));print('Forecaster smoke:',json.dumps(metrics));return
        domains=tuple(a.domains.split(','));reps=parse_ints(a.reps);pools=parse_ints(a.pool_seeds)
        if not domains or set(domains)-set(DOMAINS) or len(set(domains))!=len(domains) or set(reps)-{0,1,2} or set(pools)-{0,1} or a.limit<0:raise ValueError('Outside locked Phase 14 protocol')
        root=resolve(a.outdir);ref_path=resolve(a.reference);ref=pd.read_csv(ref_path);ref_sha=sha(ref_path);out=root/'runs.csv';df=pd.read_csv(out) if out.exists() else pd.DataFrame();plans={};refs={};pending=[]
        for domain in domains:
            digest,meta,d,r,t2,raw_sha=prepare_domain(domain,resolve(a.data_dir),ref,reps,pools)
            config_id=fingerprint(dict(version=VERSION,config=CONFIG,generator=GEN,domain=domain,data_sha256=digest,raw_sha256=raw_sha,reference_sha256=ref_sha,runner_sha256=sha(Path(__file__)),generator_sha256=sha(Path(__file__).with_name('asoc_timegan.py'))))
            validate_existing(df,root,domain,digest,{**meta,'test_start':t2},config_id);plans[domain]=(digest,meta,d,t2,config_id,raw_sha);refs[domain]=r
            done={(int(x.rep),int(x.pool_seed)) for x in df[df.domain==domain].itertuples()} if not df.empty else set()
            jobs=[(domain,rep,ps) for ps in pools for rep in reps if (rep,ps) not in done];pending.extend(jobs);print(domain,'real_windows',meta['real_windows'],'matched_steps',meta['matched_steps'],'test_windows',meta['test_windows'],'pending',len(jobs))
        print('Total pending forecasters:',len(pending))
        if a.dry_run:return
        if a.report:report(df,refs,root,domains,reps,pools);return
        if not pending:return
        device=select_device(allow_cpu=a.allow_cpu,gpu_index=a.gpu_index)
        # CPU may be used only for --smoke; no mixing CPU/GPU campaign rows.
        if device.type!='cuda':raise ValueError('Real campaign requires CUDA; use --smoke --allow-cpu for CPU')
        for domain,rep,ps in pending[:a.limit or None]:
            digest,meta,d,t2,cid,raw_sha=plans[domain];md=dict(version=VERSION,domain=domain,data_sha256=digest,config_id=cid,pool_seed=ps,generator=GEN)
            pair,stats,path=pool_for(root,d['train'],md,device);pool_hash=sha(path)
            if not df.empty:
                old=df[(df.domain==domain)&(df.pool_seed==ps)]
                if not old.empty and not (old.pool_sha256==pool_hash).all():raise ValueError('Pool changed across model repetitions')
            train=tuple(np.concatenate([r,s],axis=0) for r,s in zip(d['train'],pair));torch.cuda.reset_peak_memory_stats(device)
            metrics,ms,ma=fit_with_origins(train,d['val'],d['test'],rep,meta['matched_steps'],math.ceil(2*meta['real_windows']/64),device)
            if len(ms)!=meta['test_windows'] or not np.isclose(ms.mean(),metrics['test_mse'],rtol=1e-6) or not np.isclose(ma.mean(),metrics['test_mae'],rtol=1e-6):raise ValueError('Origin error aggregate mismatch')
            origin=root/'origins'/f'{domain}_{cid}_r{rep}_p{ps}.npz';atomic_npz(origin,origin=np.arange(t2,t2+len(ms),dtype=np.int64),mse=ms,mae=ma)
            row=dict(experiment_version=VERSION,domain=domain,rep=rep,pool_seed=ps,config_id=cid,data_sha256=digest,raw_sha256=raw_sha,reference_sha256=ref_sha,**CONFIG,**meta,**metrics,**stats,pool_file=path.relative_to(root).as_posix(),pool_sha256=pool_hash,origin_file=origin.relative_to(root).as_posix(),origin_sha256=sha(origin),mse_raw=metrics['test_mse']*meta['target_sd']**2,mae_raw=metrics['test_mae']*meta['target_sd'],device=str(device),gpu_name=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,gpu_peak_memory_mb=torch.cuda.max_memory_allocated(device)/2**20)
            df=pd.concat([df,pd.DataFrame([row])],ignore_index=True);atomic_csv(df,out);print(domain,rep,ps,'MSE',row['test_mse'],'MAE',row['test_mae'],'pool',pool_hash[:12],flush=True)
    except (ValueError,FileNotFoundError,RuntimeError) as e:p.error(str(e))
if __name__=='__main__':main()
