"""Phase 16: same-device, approximate wall-clock matched pilot; additive only."""
from __future__ import annotations
import argparse, copy, json, math, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts import asoc_real_timegan as p14
from scripts import asoc_phase15_origins as p15

VERSION='phase16_wallclock_pilot_v1'
BASE=Path('results/asoc_revision/phase16_compute')
ARMS=('timegan','real_only','duplicate_real','vae')
TOL=.05

def need(ok,msg):
    if not ok: raise ValueError(msg)

def environment(device):
    return dict(device=str(device),gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',torch_version=str(torch.__version__),cuda_version=torch.version.cuda or 'none',platform=platform.platform(),python=platform.python_version(),cudnn_version=str(torch.backends.cudnn.version()))

def code_hash():
    paths=['scripts/asoc_phase16_compute.py','scripts/asoc_phase15_origins.py','scripts/asoc_real_timegan.py','scripts/asoc_timegan.py','scripts/asoc_budget_controls.py','src/generators.py','src/models.py','src/data.py']
    return p14.fingerprint({s:p14.sha(p14.ROOT/s) for s in paths})

def checkpoint_stop(step, elapsed, fixed_steps, budget):
    return step>=fixed_steps if fixed_steps is not None else elapsed>=budget

def warmup(real,device):
    # Fixed, discarded training-only warmup before EACH arm, outside budget.
    from scripts.asoc_budget_controls import seed_everything
    from src.models import build
    seed_everything(2716)
    m=build('itransformer',96,24).to(device);opt=torch.optim.Adam(m.parameters(),lr=.001)
    x=torch.as_tensor(real[0][:64],device=device);y=torch.as_tensor(real[1][:64],device=device)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);nn.functional.mse_loss(m(x),y).backward();opt.step()
    p14.sync(device)
    del m,opt,x,y

def fit_budget(train,val,test,device,*,start,fixed_steps=None,budget=None,interval=112,model_factory=None,batch_size=64,clock=time.perf_counter):
    """Clock starts BEFORE generation. Stop only after full validation checkpoints.
    Test is evaluated once, after budget clock stops, on validation-selected state.
    """
    from scripts.asoc_budget_controls import seed_everything,_loader,_evaluate
    if model_factory is None:
        from src.models import build
        model_factory=build
    need((fixed_steps is None)!=(budget is None),'Exactly one stopping rule required')
    need(interval>=1 and (fixed_steps is None or fixed_steps>=1) and (budget is None or budget>0),'Invalid stop rule')
    seed_everything(0);m=model_factory('itransformer',train[0].shape[1],train[1].shape[1]).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=.001);loader=_loader(train,batch_size,True);it=iter(loader)
    best=float('inf');state=None;best_step=0;curve=[];step=0;samples=0
    p14.sync(device);fit_start=clock()
    while True:
        step+=1
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        samples+=len(x);m.train();x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
        loss=nn.functional.mse_loss(m(x),y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();opt.step()
        if step%interval==0 or (fixed_steps is not None and step==fixed_steps):
            v,_=_evaluate(m,val,batch_size,device)
            need(np.isfinite(v),'Nonfinite validation loss')
            if v<best:best=v;best_step=step;state=copy.deepcopy(m.state_dict())
            p14.sync(device);elapsed=clock()-start;curve.append([step,float(v),elapsed])
            if checkpoint_stop(step,elapsed,fixed_steps,budget):break
        if step>=1_000_000:raise RuntimeError('Safety step cap reached')
    end=clock();need(state is not None,'No validation checkpoint')
    # Test is never used to choose budget, steps or checkpoint.
    m.load_state_dict(state);m.eval();ms=[];ma=[];se=ae=0.;n=0
    p14.sync(device);test_start=clock()
    with torch.no_grad():
        for x,y in _loader(test,batch_size,False):
            r=m(x.to(device))-y.to(device);se+=r.square().sum().item();ae+=r.abs().sum().item();n+=r.numel()
            ms.append(r.square().mean(1).cpu().numpy());ma.append(r.abs().mean(1).cpu().numpy())
    p14.sync(device);test_seconds=clock()-test_start
    ms,ma=np.concatenate(ms),np.concatenate(ma);need(np.isfinite(ms).all() and np.isfinite(ma).all(),'Nonfinite test trace')
    return dict(optimizer_steps=step,samples_seen=samples,train_windows=len(train[0]),best_step=best_step,val_mse=best,test_mse=se/n,test_mae=ae/n,validation_checkpoints=len(curve),validation_curve=json.dumps(curve),budgeted_seconds=end-start,forecaster_loop_seconds=end-fit_start,test_seconds=test_seconds),ms,ma

def generate(arm,plan,device):
    real=plan['data']['train']
    if arm=='timegan':return p14.generate_pool(real,seed=0,device=device,**p14.GEN)
    if arm=='vae':
        from src.data import WindowConfig
        from src.generators import gen_vae
        from scripts.asoc_budget_controls import only_channels
        return only_channels(gen_vae(plan['vae_segment'],WindowConfig(96,24),ratio=1.,seed=0,epochs=30,latent=16),'target'),{}
    return None,{}

def budget_status(elapsed,target):
    need(np.isfinite(elapsed) and np.isfinite(target) and target>0 and elapsed>0,'Invalid budget')
    pct=100*(elapsed/target-1)
    return pct,abs(pct)<=100*TOL

def validate_group(root,plan,domain,cid):
    f=root/domain/'runs.csv'
    if not f.exists():return None
    df=pd.read_csv(f);need(len(df)==4 and set(df.arm)==set(ARMS) and not df.arm.duplicated().any(),'Invalid completed group')
    need(set(df.version)=={VERSION} and set(df.domain)=={domain} and set(df.config_id)=={cid},'Group provenance changed; new output directory required')
    need(df.environment.nunique()==1 and df.session_id.nunique()==1,'Mixed session/environment')
    need((df.rep==0).all() and (df.pool_seed==df.arm.map({'timegan':0,'vae':0,'real_only':-1,'duplicate_real':-1})).all(),'Unexpected seed')
    need(np.isfinite(df[['test_mse','test_mae','budgeted_seconds','target_seconds','generator_seconds','val_mse']].to_numpy()).all() and (df[['test_mse','test_mae','budgeted_seconds','target_seconds','val_mse']]>0).all().all(),'Invalid run metrics')
    anchor=df[df.arm=='timegan'].iloc[0];target=float(anchor.budgeted_seconds)
    need(int(anchor.optimizer_steps)==plan['meta']['matched_steps'],'Anchor step mismatch')
    for r in df.itertuples():
        need(np.isclose(r.target_seconds,target,rtol=1e-9) and r.data_sha256==plan['digest'],'Budget/data mismatch')
        pct,ok=budget_status(r.budgeted_seconds,target);need(np.isclose(pct,r.budget_delta_pct,atol=1e-7) and str(r.within_tolerance).lower()==str(ok).lower(),'Budget flag mismatch')
        need(r.optimizer_steps>=1 and 1<=r.best_step<=r.optimizer_steps and r.generator_seconds>=0 and r.generator_seconds<=r.budgeted_seconds,'Invalid training budget/checkpoint')
        curve=json.loads(r.validation_curve);need(len(curve)==r.validation_checkpoints and curve[-1][0]==r.optimizer_steps and all(np.isfinite(v).all() for v in curve),'Invalid validation trace')
        if r.arm!='timegan':need(r.budgeted_seconds>=target,'Comparator stopped before target')
        p15.read_trace(root/domain,r,plan['idx'])
        if r.arm in ('timegan','vae'):
            pool=root/domain/str(r.pool_file);need(pool.is_file() and p14.sha(pool)==r.pool_sha256,'Pool hash mismatch')
            with np.load(pool,allow_pickle=False) as z:
                pair=(z['x'],z['y']);md=json.loads(str(z['metadata'].item()));need(md==dict(version=VERSION,domain=domain,arm=r.arm,seed=0,config_id=cid),'Pool metadata mismatch');need(p14.array_digest(pair)==str(z['content_sha256'].item()),'Pool content hash mismatch');need(all(a.shape==b.shape and np.isfinite(a).all() for a,b in zip(pair,plan['data']['train'])),'Pool shape mismatch')
    return df

def run_group(root,domain,plan,cid,device):
    dest=root/domain;stage=root/(domain+'.inprogress')
    need(not dest.exists() and not stage.exists(),f'{domain}: existing/incomplete group; preserve it, use a new --outdir or rename incomplete folder')
    stage.mkdir(parents=True);env=environment(device);session=str(time.time_ns());rows=[];target=None
    real=plan['data']['train'];meta=plan['meta'];interval=math.ceil(2*meta['real_windows']/64)
    for arm in ARMS:
        print(domain,arm,'starting',flush=True);warmup(real,device)
        p14.sync(device);start=time.perf_counter()
        pool,stats=generate(arm,plan,device);p14.sync(device);gen_seconds=time.perf_counter()-start if pool is not None else 0.
        if pool is not None:need(all(a.shape==b.shape and np.isfinite(a).all() for a,b in zip(pool,real)),'Invalid fresh pool')
        train=tuple(np.concatenate([a,b],axis=0) for a,b in zip(real,pool if pool is not None else real)) if arm!='real_only' else real
        if target is not None:need(time.perf_counter()-start<target,'Generator consumed full budget; no valid comparator')
        metrics,ms,ma=fit_budget(train,plan['data']['val'],plan['data']['test'],device,start=start,fixed_steps=meta['matched_steps'] if arm=='timegan' else None,budget=target,interval=interval)
        if target is None:target=metrics['budgeted_seconds']
        pct,ok=budget_status(metrics['budgeted_seconds'],target)
        need(len(ms)==len(plan['idx']) and np.isclose(ms.mean(),metrics['test_mse'],rtol=1e-6) and np.isclose(ma.mean(),metrics['test_mae'],rtol=1e-6),'Origin aggregate mismatch')
        origin=stage/'origins'/f'{arm}.npz';p14.atomic_npz(origin,origin=plan['idx'],mse=ms,mae=ma)
        pf=ph=''
        if pool is not None:
            path=stage/'pools'/f'{arm}.npz';p14.atomic_npz(path,x=pool[0],y=pool[1],metadata=json.dumps(dict(version=VERSION,domain=domain,arm=arm,seed=0,config_id=cid),sort_keys=True),stats=json.dumps(stats,allow_nan=False),content_sha256=p14.array_digest(pool));pf=path.relative_to(stage).as_posix();ph=p14.sha(path)
        historical=plan['tg'].query('rep==0 and pool_seed==0').iloc[0].test_mse if arm=='timegan' else plan['refs'][(0,{'real_only':'step_baseline','duplicate_real':'duplicate_real','vae':'vae_augmented'}[arm],0 if arm=='vae' else -1)].test_mse
        row=dict(version=VERSION,domain=domain,arm=arm,rep=0,pool_seed=0 if pool is not None else -1,config_id=cid,session_id=session,environment=json.dumps(env,sort_keys=True),data_sha256=plan['digest'],raw_sha256=plan['rawsha'],target_seconds=target,generator_seconds=gen_seconds,budget_delta_pct=pct,within_tolerance=ok,validation_interval=interval,historical_equal_step_mse=historical,**metrics,origin_file=origin.relative_to(stage).as_posix(),origin_sha256=p14.sha(origin),pool_file=pf,pool_sha256=ph,mse_raw=metrics['test_mse']*meta['target_sd']**2,mae_raw=metrics['test_mae']*meta['target_sd'])
        rows.append(row);p14.atomic_csv(pd.DataFrame(rows),stage/'runs.csv');print(domain,arm,'seconds',round(metrics['budgeted_seconds'],3),'target',round(target,3),'steps',metrics['optimizer_steps'],'MSE',metrics['test_mse'],'budget_ok',ok,flush=True)
    os.replace(stage,dest)

def report(root,groups):
    df=pd.concat(groups,ignore_index=True);rows=[]
    for d,g in df.groupby('domain'):
        a=g[g.arm=='timegan'].iloc[0]
        for r in g.itertuples():
            if r.arm=='timegan':continue
            rows.append(dict(domain=d,comparator=r.arm,timegan_mse=a.test_mse,comparator_mse=r.test_mse,timegan_mae=a.test_mae,comparator_mae=r.test_mae,delta_mse=a.test_mse-r.test_mse,delta_mse_pct=100*(a.test_mse/r.test_mse-1),delta_mae=a.test_mae-r.test_mae,delta_mae_pct=100*(a.test_mae/r.test_mae-1),timegan_seconds=a.budgeted_seconds,comparator_seconds=r.budgeted_seconds,comparator_steps=r.optimizer_steps,budget_delta_pct=r.budget_delta_pct,within_tolerance=r.within_tolerance,scope='single_seed_single_device_wallclock_pilot'))
    p14.atomic_csv(df,root/'report/runs.csv');p14.atomic_csv(pd.DataFrame(rows),root/'report/comparisons.csv');print(pd.DataFrame(rows).to_string(index=False));print('Approximate wall-clock budget; not GPU FLOPs or general speed benchmark. Out-of-tolerance pairs are not matched.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domains',default='weather,traffic');p.add_argument('--data-dir',type=Path,default=Path('data/real'));p.add_argument('--reference',type=Path,default=p14.REF);p.add_argument('--timegan-dir',type=Path,default=p14.BASE);p.add_argument('--outdir',type=Path,default=BASE);p.add_argument('--gpu-index',type=int,default=0);p.add_argument('--dry-run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true');a=p.parse_args()
    try:
        from scripts.asoc_budget_controls import select_device
        if a.smoke:
            dev=select_device(allow_cpu=a.allow_cpu,gpu_index=a.gpu_index);rng=np.random.default_rng(16);x=rng.normal(size=(16,96,1)).astype('float32');y=rng.normal(size=(16,24)).astype('float32');pair=(x,y)
            warmup(pair,dev);p14.sync(dev);start=time.perf_counter();m,_,_=fit_budget(pair,pair,pair,dev,start=start,fixed_steps=2,interval=1,batch_size=8);print('SMOKE PASS fixed steps',m['optimizer_steps'],m['budgeted_seconds']);p14.sync(dev);start=time.perf_counter();m,_,_=fit_budget(pair,pair,pair,dev,start=start,budget=.05,interval=1,batch_size=8);print('SMOKE PASS timed steps',m['optimizer_steps'],m['budgeted_seconds']);return
        domains=tuple(a.domains.split(','));need(set(domains)<=set(p14.DOMAINS) and len(set(domains))==len(domains),'Invalid domains');need(not(a.dry_run and a.report),'Choose dry-run or report')
        root=p14.resolve(a.outdir);refpath=p14.resolve(a.reference);ref=pd.read_csv(refpath);refsha=p14.sha(refpath);tgroot=p14.resolve(a.timegan_dir);tg=pd.read_csv(tgroot/'runs.csv');datadir=p14.resolve(a.data_dir);plans={};done={}
        for d in domains:
            plan=p15.load_plan(d,datadir,ref,refsha,tg,tgroot);from src.data import standardize
            arr=np.load(datadir/f'{d}.npy',allow_pickle=False).astype('float32');m=plan['meta'];xs,_,_=standardize(arr,slice(m['train_start'],m['train_end']));plan['vae_segment']=xs[m['train_start']:m['train_end']];cid=p14.fingerprint(dict(version=VERSION,code=code_hash(),phase15_plan=plan['config_id'],arms=ARMS,tolerance=TOL));plans[d]=(plan,cid);done[d]=validate_group(root,plan,d,cid)
            need(not (root/(d+'.inprogress')).exists(),f'{d}: incomplete group exists; preserve it and use a new --outdir or rename incomplete folder')
            print(d,'matched_steps',plan['meta']['matched_steps'],'pending fits',0 if done[d] is not None else 4,flush=True)
        print('Total pending fits:',sum(4 for d in domains if done[d] is None))
        if a.dry_run:return
        if a.report:
            need(all(done[d] is not None for d in domains),'Incomplete requested domain groups');report(root,[done[d] for d in domains]);return
        device=select_device(gpu_index=a.gpu_index);need(device.type=='cuda','Real pilot requires CUDA')
        for d,(plan,cid) in plans.items():
            if done[d] is None:run_group(root,d,plan,cid,device)
    except (ValueError,RuntimeError,FileNotFoundError,FloatingPointError) as e:p.error(str(e))

if __name__=='__main__':main()
