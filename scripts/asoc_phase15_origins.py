"""Phase 15: independently versioned replay of Phase 7 matched controls.
Requires Phase 14, frozen Phase 7 CSV and exact original TimeGAN loss files.
No historical result is overwritten. One writer only. GPU training required.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts import asoc_real_timegan as p14

VERSION='phase15_matched_origin_replay_v1'
BASE=Path('results/asoc_revision/phase15_origins')
CONDS=('step_baseline','duplicate_real','vae_replay')
REPS=(0,1,2); POOLS=(0,1)
BLOCKS={'weather':(24,144,288),'traffic':(24,168,336)}

def need(ok,msg):
    if not ok: raise ValueError(msg)

def jobs(domains):
    return [(d,r,c,s) for d in domains for c in CONDS for s in (POOLS if c=='vae_replay' else (-1,)) for r in REPS]

def read_trace(root,row,origins):
    path=root/str(row.origin_file)
    need(path.is_file() and p14.sha(path)==row.origin_sha256,f'Origin missing or changed: {path}')
    with np.load(path,allow_pickle=False) as z:
        idx=z['origin'];m=z['mse'].astype('float64');a=z['mae'].astype('float64')
    need(np.array_equal(idx,origins) and m.shape==idx.shape and a.shape==idx.shape,'Origin alignment/shape mismatch')
    need(np.isfinite(m).all() and np.isfinite(a).all() and (m>=0).all() and (a>=0).all(),'Invalid loss trace')
    need(np.isclose(m.mean(),row.test_mse,rtol=1e-6) and np.isclose(a.mean(),row.test_mae,rtol=1e-6),'Origin aggregate mismatch')
    return m,a

def code_id():
    root=p14.ROOT
    paths=['scripts/asoc_phase15_origins.py','scripts/asoc_real_timegan.py','scripts/asoc_timegan.py','scripts/asoc_budget_controls.py','src/generators.py','src/models.py','src/data.py']
    return p14.fingerprint({x:p14.sha(root/x) for x in paths})

def load_plan(domain,datadir,ref,refsha,tg,tgroot):
    digest,meta,data,refs,t2,rawsha=p14.prepare_domain(domain,datadir,ref,REPS,POOLS)
    t=tg[tg.domain==domain].copy();need(len(t)==6 and not t.duplicated(['rep','pool_seed']).any(),'Need exactly six Phase 14 runs per domain')
    idx=np.arange(t2,t2+meta['test_windows'],dtype=np.int64)
    cfg=None; traces={}
    for r in t.itertuples():
        need((int(r.rep),int(r.pool_seed)) in [(i,j) for i in REPS for j in POOLS],'Unexpected TimeGAN seed')
        need(r.experiment_version==p14.VERSION and r.data_sha256==digest and r.raw_sha256==rawsha and r.reference_sha256==refsha,'TimeGAN provenance mismatch')
        for k,v in {**p14.CONFIG,**meta}.items():
            got=getattr(r,k);need(np.isclose(float(got),float(v),rtol=1e-6,atol=1e-8) if isinstance(v,float) else got==v,f'TimeGAN {k} mismatch')
        need(r.optimizer_steps==meta['matched_steps'],'TimeGAN budget mismatch')
        if cfg is None:cfg=r.config_id
        need(r.config_id==cfg,'Mixed TimeGAN configurations')
        pool=tgroot/str(r.pool_file);need(pool.is_file() and p14.sha(pool)==r.pool_sha256,f'TimeGAN pool missing/changed: {pool}')
        traces[(int(r.rep),int(r.pool_seed))]=read_trace(tgroot,r,idx)
    expected_cfg=p14.fingerprint(dict(version=p14.VERSION,config=p14.CONFIG,generator=p14.GEN,domain=domain,data_sha256=digest,raw_sha256=rawsha,reference_sha256=refsha,runner_sha256=p14.sha(p14.ROOT/'scripts/asoc_real_timegan.py'),generator_sha256=p14.sha(p14.ROOT/'scripts/asoc_timegan.py')))
    need(cfg==expected_cfg,'Phase 14 source/config hash differs from historical runs; preserve exact source bytes')
    return dict(digest=digest,meta=meta,data=data,refs=refs,idx=idx,rawsha=rawsha,tg=t,traces=traces,config_id=p14.fingerprint(dict(version=VERSION,code=code_id(),data=digest,raw=rawsha,ref=refsha,timegan_config=cfg,timegan_runs_sha=p14.sha(tgroot/'runs.csv'))))

def vae_pool(root,domain,ps,plan,device,existing):
    from src.data import WindowConfig, standardize
    from src.generators import gen_vae
    from scripts.asoc_budget_controls import only_channels
    meta=dict(version=VERSION,domain=domain,pool_seed=ps,config_id=plan['config_id'],data_sha256=plan['digest'],kind='new_vae_replay_not_original')
    path=root/'pools'/f'{domain}_{plan["config_id"]}_p{ps}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            need(json.loads(str(z['metadata'].item()))==meta,'VAE cache metadata mismatch');pair=(z['x'],z['y']);seconds=float(z['seconds'].item());env=json.loads(str(z['generator_environment'].item()));need(str(z['content_sha256'].item())==p14.array_digest(pair),'VAE content hash mismatch')
    else:
        need(existing.empty,'Referenced VAE pool missing; do not regenerate')
        arr=plan['array'];m=plan['meta'];xs,_,_=standardize(arr,slice(m['train_start'],m['train_end']))
        p14.sync(device);tick=time.perf_counter();pair=only_channels(gen_vae(xs[m['train_start']:m['train_end']],WindowConfig(96,24),ratio=1.,seed=ps,epochs=30,latent=16),'target');p14.sync(device);seconds=time.perf_counter()-tick
        env=dict(gpu_name=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,device=str(device))
        p14.atomic_npz(path,x=pair[0],y=pair[1],metadata=json.dumps(meta,sort_keys=True),seconds=seconds,generator_environment=json.dumps(env,sort_keys=True),content_sha256=p14.array_digest(pair))
    need(all(x.shape==y.shape and np.isfinite(x).all() for x,y in zip(pair,plan['data']['train'])),'Invalid VAE pool')
    digest=p14.sha(path);need(existing.empty or (existing.pool_sha256==digest).all(),'VAE pool hash mismatch')
    return pair,path,digest,seconds,env

def verify_existing(df,plans,root):
    if df.empty:return
    need(not df.duplicated(['domain','rep','condition','pool_seed']).any(),'Duplicate replay keys')
    need(set(df.experiment_version)=={VERSION},'Mixed replay versions')
    valid=set(jobs(tuple(plans)))
    for r in df.itertuples():
        if r.domain not in plans:continue
        k=(r.domain,int(r.rep),r.condition,int(r.pool_seed));need(k in valid,'Invalid replay key');p=plans[r.domain]
        need(r.config_id==p['config_id'] and r.data_sha256==p['digest'] and r.optimizer_steps==p['meta']['matched_steps'],'Replay configuration changed')
        read_trace(root,r,p['idx'])
        if r.condition=='vae_replay':
            f=root/str(r.pool_file);need(f.is_file() and p14.sha(f)==r.pool_sha256,'Replay VAE pool missing/changed')

def circular_ci(x,block,draws=2000,seed=1501):
    """Percentile circular moving-block bootstrap of a mean, same indices per paired loss difference.
    Conditional on the fitted models and one observed test interval; no seed/domain resampling.
    """
    x=np.asarray(x,dtype='float64');n=len(x);need(x.ndim==1 and np.isfinite(x).all() and 1<=block<=n and draws>=100,'Invalid bootstrap input')
    rng=np.random.default_rng(seed);k=math.ceil(n/block);offset=np.arange(block);means=np.empty(draws)
    for i in range(draws):
        ix=((rng.integers(0,n,k)[:,None]+offset)%n).ravel()[:n];means[i]=x[ix].mean()
    lo,hi=np.quantile(means,[.025,.975]);return float(x.mean()),float(lo),float(hi)

def make_report(df,plans,root,draws=2000):
    pair_rows=[];domain_rows=[];interval_rows=[]
    for d,p in plans.items():
        rows=df[df.domain==d];need(len(rows)==12,f'{d}: report requires all 12 replay rows')
        traces={(int(r.rep),r.condition,int(r.pool_seed)):read_trace(root,r,p['idx']) for r in rows.itertuples()}
        grouped={}
        for rep in REPS:
            for ps in POOLS:
                tg=p['traces'][(rep,ps)]
                for c in CONDS:
                    base=traces[(rep,c,ps if c=='vae_replay' else -1)]
                    for j,metric in enumerate(('mse','mae')):
                        delta=tg[j]-base[j];b=base[j].mean();need(b>0,'Zero comparator mean')
                        pair_rows.append(dict(domain=d,rep=rep,pool_seed=ps,comparator=c,metric=metric,timegan_mean=tg[j].mean(),replay_mean=b,delta=delta.mean(),delta_pct=100*delta.mean()/b))
                        grouped.setdefault((c,metric),[]).append(delta)
        for (c,metric),v in grouped.items():
            # all six aligned comparisons have equal weight; shared controls are not treated as independent observations
            x=np.mean(np.stack(v),axis=0);n=len(x)
            domain_rows.append(dict(domain=d,comparator=c,metric=metric,n_origins=n,n_model_pool_pairs=6,mean_delta=x.mean(),lag1_corr=np.corrcoef(x[:-1],x[1:])[0,1] if x.std()>0 else np.nan))
            for block in BLOCKS[d]:
                avg,lo,hi=circular_ci(x,block,draws)
                interval_rows.append(dict(domain=d,comparator=c,metric=metric,block=block,primary_block=block==BLOCKS[d][1],draws=draws,n_origins=n,mean_delta=avg,ci_low=lo,ci_high=hi,scope='conditional_on_fixed_models_single_test_period'))
    p14.atomic_csv(pd.DataFrame(pair_rows),root/'report/paired.csv');p14.atomic_csv(pd.DataFrame(domain_rows),root/'report/domain_means.csv');p14.atomic_csv(pd.DataFrame(interval_rows),root/'report/block_bootstrap.csv')
    print(pd.DataFrame(interval_rows).query('primary_block and metric == "mse"').to_string(index=False));print('Intervals describe conditional temporal uncertainty; not population or independent-seed inference.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--domains',default='weather,traffic');parser.add_argument('--data-dir',type=Path,default=Path('data/real'));parser.add_argument('--reference',type=Path,default=p14.REF);parser.add_argument('--timegan-dir',type=Path,default=p14.BASE);parser.add_argument('--outdir',type=Path,default=BASE);parser.add_argument('--gpu-index',type=int,default=0);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--report',action='store_true');parser.add_argument('--limit',type=int,default=0)
    a=parser.parse_args()
    try:
        from scripts.asoc_budget_controls import select_device
        domains=tuple(a.domains.split(','));need(domains and set(domains)<=set(p14.DOMAINS) and len(set(domains))==len(domains) and a.limit>=0,'Invalid protocol options')
        root=p14.resolve(a.outdir);tgroot=p14.resolve(a.timegan_dir);refpath=p14.resolve(a.reference);datadir=p14.resolve(a.data_dir)
        need(root.resolve()!=tgroot.resolve() and not tgroot.resolve() in root.resolve().parents,'Use separate replay output directory')
        ref=pd.read_csv(refpath);refsha=p14.sha(refpath);tg=pd.read_csv(tgroot/'runs.csv');plans={d:load_plan(d,datadir,ref,refsha,tg,tgroot) for d in domains}
        out=root/'runs.csv';df=pd.read_csv(out) if out.exists() else pd.DataFrame();verify_existing(df,plans,root)
        done=set(zip(df.domain,df.rep,df.condition,df.pool_seed)) if not df.empty else set();pending=[k for k in jobs(domains) if k not in done]
        for d in domains:print(d,'pending',sum(k[0]==d for k in pending),'matched_steps',plans[d]['meta']['matched_steps'])
        print('Total pending replay fits:',len(pending))
        if a.dry_run:return
        if a.report:need(not pending,'Incomplete requested campaign');make_report(df,plans,root);return
        if not pending:return
        device=select_device(gpu_index=a.gpu_index)
        for d,r,c,ps in pending[:a.limit or None]:
            p=plans[d];real=p['data']['train'];m=p['meta'];seconds=0.;poolfile='';poolhash='';genenv={}
            if c=='step_baseline':train=real
            elif c=='duplicate_real':train=tuple(np.concatenate([x,x]) for x in real)
            else:
                p['array']=np.load(datadir/f'{d}.npy',allow_pickle=False).astype('float32');old=df[(df.domain==d)&(df.condition==c)&(df.pool_seed==ps)] if not df.empty else df
                extra,path,poolhash,seconds,genenv=vae_pool(root,d,ps,p,device,old);poolfile=path.relative_to(root).as_posix();train=tuple(np.concatenate([x,y]) for x,y in zip(real,extra))
            metrics,ms,ma=p14.fit_with_origins(train,p['data']['val'],p['data']['test'],r,m['matched_steps'],math.ceil(2*m['real_windows']/64),device)
            need(len(ms)==len(p['idx']) and np.isclose(ms.mean(),metrics['test_mse'],rtol=1e-6) and np.isclose(ma.mean(),metrics['test_mae'],rtol=1e-6),'Replay origin/aggregate mismatch')
            origin=root/'origins'/f'{d}_{p["config_id"]}_{c}_r{r}_p{ps}.npz';p14.atomic_npz(origin,origin=p['idx'],mse=ms,mae=ma)
            old=p['refs'][(r,'vae_augmented' if c=='vae_replay' else c,ps)];row=dict(experiment_version=VERSION,domain=d,rep=r,condition=c,pool_seed=ps,config_id=p['config_id'],data_sha256=p['digest'],raw_sha256=p['rawsha'],reference_sha256=refsha,**p14.CONFIG,**m,**metrics,pool_file=poolfile,pool_sha256=poolhash,generator_seconds=seconds,generator_environment=json.dumps(genenv,sort_keys=True),origin_file=origin.relative_to(root).as_posix(),origin_sha256=p14.sha(origin),historical_test_mse=float(old.test_mse),historical_test_mae=float(old.test_mae),replay_minus_historical_mse=metrics['test_mse']-float(old.test_mse),gpu_name=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
            df=pd.concat([df,pd.DataFrame([row])],ignore_index=True);p14.atomic_csv(df,out);print(d,c,r,ps,'MSE',row['test_mse'],'historical',row['historical_test_mse'],flush=True)
    except (ValueError,RuntimeError,FileNotFoundError) as e:parser.error(str(e))
if __name__=='__main__':main()
