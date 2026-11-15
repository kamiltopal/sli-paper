"""Fixed replication and checkpoint-sensitivity study; never modifies historical results."""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts import asoc_phase20_adequacy as p20

ROOT=p20.ROOT
VERSION='phase21_replication_dual_checkpoint_v1'
DOMAINS=('weather','electricity')
POLICIES=('coarse','fine')
FINE={'weather':28,'electricity':5}
COARSE={'weather':112,'electricity':35}
KEYS=['study','domain','representation','rep','arm','pool_seed']
CONFIG=dict(regime=.1,lookback=96,horizon=24,batch_size=64,lr=.001,epochs=10,
            reps=list(range(20)),pool_seeds=list(range(5)),interaction_reps=list(range(10)),
            interaction_pools=[0,1],fine=FINE,coarse=COARSE,vae_epochs=30,latent=16,
            rng_policy='restore_python_numpy_torch_cpu_cuda_around_evaluation',
            tie_policy='earliest_minimum_validation_mse',ridge_alpha=1.)

def jobs():
    j=[]
    for r in range(20):
        for d in DOMAINS:
            j += [('replication',d,'centered',r,a,-1) for a in ('step_baseline','duplicate_real')]
            j += [('replication',d,'centered',r,'vae',p) for p in range(5)]
    for r in range(10):
        for p in (0,1):
            for m in ('original','centered'):
                for a in ('vae','timegan'):
                    j.append(('interaction','weather',m,r,a,p))
    pilot=[k for k in j if k[3]==0 and k[5] in (-1,0)]
    return pilot+[k for k in j if k not in pilot]

@contextmanager
def isolated_rng():
    py=random.getstate(); npstate=np.random.get_state(); cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try: yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)

def factory(mode,L,H):
    from src.models import build
    base=build('itransformer',L,H)
    return p20.LastValue(base) if mode=='centered' else base

def fit_dual(train,val,test,rep,steps,coarse,fine,device,mode='centered',batch_size=64,lr=.001,make_model=factory):
    from scripts.asoc_budget_controls import seed_everything,_loader,_evaluate
    p20.need(steps%coarse==0 and coarse%fine==0,'Checkpoint grid must be nested')
    seed_everything(rep);model=make_model(mode,train[0].shape[1],train[1].shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr);loader=_loader(train,batch_size,True);it=iter(loader)
    best={p:(float('inf'),0,None) for p in POLICIES};curve=[];samples=0
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();tick=time.perf_counter()
    for s in range(1,steps+1):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        model.train();x,y=x.to(device),y.to(device);samples+=len(x);opt.zero_grad(set_to_none=True)
        loss=nn.functional.mse_loss(model(x),y);p20.need(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward();opt.step()
        if s%fine==0:
            with isolated_rng():
                v,_=_evaluate(model,val,batch_size,device)
            p20.need(math.isfinite(v),'Nonfinite validation');curve.append([s,v])
            for policy in POLICIES:
                if (policy=='fine' or s%coarse==0) and v<best[policy][0]:
                    best[policy]=(v,s,copy.deepcopy(model.state_dict()))
    # Testing is after the complete training path; both policies select on validation only.
    result={}; traces={}
    for policy in POLICIES:
        v,s,state=best[policy];p20.need(state is not None,'Missing checkpoint');model.load_state_dict(state);model.eval();ms=[];ma=[]
        with isolated_rng(),torch.no_grad():
            for x,y in _loader(test,batch_size,False):
                e=model(x.to(device))-y.to(device);ms.append(e.square().mean(1).cpu().numpy());ma.append(e.abs().mean(1).cpu().numpy())
        m=np.concatenate(ms);a=np.concatenate(ma);p20.need(np.isfinite(m).all() and np.isfinite(a).all(),'Nonfinite test')
        result[policy]=dict(test_mse=float(m.astype('float64').mean()),test_mae=float(a.astype('float64').mean()),val_mse=v,best_step=s)
        traces[policy]=(m,a)
    sync();return result,traces,curve,samples,time.perf_counter()-tick

def new_pool(root,d,ps,plan,cid,device,create):
    from src.data import WindowConfig,standardize
    from src.generators import gen_vae
    from scripts.asoc_budget_controls import only_channels,seed_everything
    from scripts.asoc_real_timegan import atomic_npz,array_digest
    path=root/'pools'/f'{d}_vae_p{ps}.npz';meta=dict(config_id=cid,domain=d,pool_seed=ps,kind='new_replication_vae',data_sha256=p20.HASHES[d]);real=plan['pairs']['train']
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            pair=(z['x'],z['y']);p20.need(json.loads(str(z['metadata'].item()))==meta and str(z['content_sha256'].item())==array_digest(pair),'New pool metadata/content mismatch')
    else:
        p20.need(create,'Referenced new pool missing: '+str(path));m=plan['meta'];xs,_,_=standardize(plan['array'],slice(m['train_start'],m['train_end']))
        # Independent seed-defined generator fits; same training data and settings as historical VAE.
        seed_everything(ps);pair=only_channels(gen_vae(xs[m['train_start']:m['train_end']],WindowConfig(96,24),ratio=1.,seed=ps,epochs=30,latent=16),'target');p20.valid_pair(pair,real)
        atomic_npz(path,x=pair[0],y=pair[1],metadata=json.dumps(meta,sort_keys=True),content_sha256=array_digest(pair))
    p20.valid_pair(pair,real);return pair,path,p20.sha(path)

def job_id(k):return '_'.join(map(str,k))

def validate_record(record,k,root,results,plans,cid):
    p20.need(tuple(record['key'])==k and record['config_id']==cid,'Run key/config mismatch');_,d,_,_,arm,ps=k;m=plans[d]['meta'];r=record
    p20.need(r['optimizer_steps']==m['matched_steps'] and r['data_sha256']==p20.HASHES[d] and r['raw_sha256']==plans[d]['raw_sha'],'Run budget/data mismatch')
    curve=np.asarray(r['curve'],dtype=float);p20.need(curve.shape==(m['matched_steps']//FINE[d],2) and np.array_equal(curve[:,0],np.arange(FINE[d],m['matched_steps']+1,FINE[d])) and np.isfinite(curve).all(),'Invalid validation curve')
    if arm in ('vae','timegan'):
        p20.need(p20.sha(results/r['pool_file'])==r['pool_sha256'],'Run pool changed')
        if k[0]=='interaction':p20.need(r['pool_sha256']==p20.WEATHER_POOLS[arm][ps],'Historical pool mismatch')
        else:p20.need((results/r['pool_file']).resolve()==(root/'pools'/f'{d}_vae_p{ps}.npz').resolve(),'New pool path mismatch')
    path=root/r['origin_file'];p20.need(p20.sha(path)==r['origin_sha256'],'Origin hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        idx=np.arange(plans[d]['test_start'],plans[d]['test_start']+m['test_windows']);p20.need(np.array_equal(z['origin'],idx),'Origin index mismatch')
        for policy in POLICIES:
            c=curve if policy=='fine' else curve[curve[:,0]%COARSE[d]==0];b=c[np.argmin(c[:,1])];q=r['metrics'][policy];ms=z[policy+'_mse'];ma=z[policy+'_mae']
            p20.need(q['best_step']==b[0] and np.isclose(q['val_mse'],b[1],rtol=1e-12) and ms.shape==ma.shape==idx.shape and np.isfinite(ms).all() and np.isfinite(ma).all() and (ms>=0).all() and (ma>=0).all() and np.isclose(ms.astype('float64').mean(),q['test_mse'],rtol=1e-10) and np.isclose(ma.astype('float64').mean(),q['test_mae'],rtol=1e-10),'Checkpoint or loss mismatch')

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def summarize(records,root,plans):
    from scripts.asoc_real_timegan import atomic_csv
    rows=[]
    for r in records:
        k=r['key'];d=k[1];m=plans[d]['meta'];refs=plans[d]['refs']
        for policy in POLICIES:
            q=r['metrics'][policy];rows.append(dict(zip(KEYS,k),policy=policy,**q,**refs,optimizer_steps=r['optimizer_steps'],samples_seen=r['samples_seen'],training_seconds=r['seconds'],data_sha256=r['data_sha256'],pool_sha256=r['pool_sha256'],pool_file=r['pool_file'],origin_file=r['origin_file'],origin_sha256=r['origin_sha256'],mse_raw=q['test_mse']*m['target_sd']**2,mae_raw=q['test_mae']*m['target_sd'],first_checkpoint=q['best_step']==(FINE[d] if policy=='fine' else COARSE[d]),config_id=r['config_id']))
    if not rows:return
    f=pd.DataFrame(rows);atomic_csv(f,root/'runs.csv');idx={tuple(r[k] for k in KEYS+['policy']):r for r in rows};pairs=[]
    def add(a,b,ref):
        pairs.append({**{k:a[k] for k in KEYS+['policy']},'reference':ref,'delta_mse_pct':100*(a['test_mse']/b['test_mse']-1),'delta_mae_pct':100*(a['test_mae']/b['test_mae']-1),'mse':a['test_mse'],'reference_mse':b['test_mse'],'beats_persistence_mse':a['test_mse']<a['test_persistence_mse'],'beats_ridge_mse':a['test_mse']<a['test_ridge96_mse'],'reference_beats_persistence_mse':b['test_mse']<a['test_persistence_mse'],'reference_beats_ridge_mse':b['test_mse']<a['test_ridge96_mse']})
    for a in rows:
        s,d,m,r,arm,p=[a[k] for k in KEYS];policy=a['policy']
        if s=='replication' and arm in ('vae','duplicate_real'):
            for c in (('step_baseline','duplicate_real') if arm=='vae' else ('step_baseline',)):
                b=idx.get((s,d,m,r,c,-1,policy));
                if b:add(a,b,c)
        if s=='interaction' and arm=='timegan':
            b=idx.get((s,d,m,r,'vae',p,policy));
            if b:add(a,b,'vae')
    if pairs:
        p=pd.DataFrame(pairs);atomic_csv(p,root/'report/paired.csv');gkeys=['study','domain','representation','policy','arm','reference'];cols=['delta_mse_pct','delta_mae_pct'];g=p.groupby(gkeys+['pool_seed'])[cols].median().reset_index();atomic_csv(g,root/'report/per_pool.csv');s=g.groupby(gkeys)[cols].median().reset_index();counts=p.groupby(gkeys).agg(n_pairs=('rep','size'),mse_wins=('delta_mse_pct',lambda x:int((x<0).sum())),mae_wins=('delta_mae_pct',lambda x:int((x<0).sum())),mse_q25=('delta_mse_pct',lambda x:x.quantile(.25)),mse_q75=('delta_mse_pct',lambda x:x.quantile(.75)),median_abs_mse_delta=('delta_mse_pct',lambda x:x.abs().median())).reset_index();atomic_csv(s.merge(counts,on=gkeys),root/'report/summary.csv')
        v=p[(p.study=='replication')&(p.arm=='vae')];w=v.pivot(index=['domain','policy','rep','pool_seed'],columns='reference',values=['delta_mse_pct','delta_mae_pct']).dropna() if len(v) else pd.DataFrame()
        if len(w) and {'step_baseline','duplicate_real'}<=set(w.columns.get_level_values(1)):
            dual=pd.DataFrame({'both_mse':(w[('delta_mse_pct','step_baseline')]<0)&(w[('delta_mse_pct','duplicate_real')]<0),'both_mae':(w[('delta_mae_pct','step_baseline')]<0)&(w[('delta_mae_pct','duplicate_real')]<0)}).reset_index();atomic_csv(dual,root/'report/dual_wins.csv')
        z=p[p.study=='interaction'];
        if len(z):
            w=z.pivot(index=['domain','policy','rep','pool_seed'],columns='representation',values='delta_mse_pct').dropna()
            if len(w) and {'original','centered'}<=set(w):w=w.assign(centered_minus_original_pp=w.centered-w.original).reset_index();atomic_csv(w,root/'report/representation_interaction.csv')
    f['beats_persistence_mse']=f.test_mse<f.test_persistence_mse;f['beats_ridge_mse']=f.test_mse<f.test_ridge96_mse
    g=f.groupby(['study','domain','representation','policy','arm']);a=g.agg(n=('rep','size'),median_mse=('test_mse','median'),median_mae=('test_mae','median'),median_mse_raw=('mse_raw','median'),median_mae_raw=('mae_raw','median'),first_checkpoint_count=('first_checkpoint','sum'),beats_persistence_mse=('beats_persistence_mse','sum'),beats_ridge_mse=('beats_ridge_mse','sum')).reset_index();atomic_csv(a,root/'report/absolute.csv')
    w=f.pivot(index=KEYS,columns='policy',values=['test_mse','test_mae','best_step','val_mse']).reset_index();w.columns=['_'.join(filter(None,map(str,c))) if isinstance(c,tuple) else c for c in w.columns];w['fine_vs_coarse_mse_pct']=100*(w.test_mse_fine/w.test_mse_coarse-1);w['fine_vs_coarse_mae_pct']=100*(w.test_mae_fine/w.test_mae_coarse-1);atomic_csv(w,root/'report/checkpoint_sensitivity.csv')
    print(f'Completed {len(records)}/360; {len(rows)} policy evaluations. Partial reports are descriptive only.',flush=True)

def main():
    from scripts.asoc_budget_controls import select_device
    from scripts.asoc_real_timegan import atomic_npz
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,default=Path('data/real'));ap.add_argument('--results-dir',type=Path,default=Path('results/asoc_revision'));ap.add_argument('--gpu-index',type=int,default=0);ap.add_argument('--cpu',action='store_true');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();p20.need(a.limit>=0,'Negative limit')
    if a.smoke:
        dev=torch.device('cpu') if a.cpu else select_device(gpu_index=a.gpu_index);rng=np.random.default_rng(21);x=rng.normal(size=(12,96,1)).astype('float32');y=rng.normal(size=(12,24)).astype('float32');q,t,c,_,_=fit_dual((x,y),(x,y),(x,y),0,4,2,1,dev,batch_size=4);print('SMOKE PASS',q);return
    results=p20.resolve(a.results_dir);root=results/'phase21_replication';refpath=results/'real_budget_controls.csv';plans=p20.preflight(p20.resolve(a.data_dir),pd.read_csv(refpath));old={(arm,ps):p20.weather_pool(results,arm,ps,plans['weather']) for arm in ('vae','timegan') for ps in (0,1)}
    deps=['scripts/asoc_phase21_replication.py','scripts/asoc_phase20_adequacy.py','scripts/asoc_real_timegan.py','scripts/asoc_budget_controls.py','src/data.py','src/models.py','src/generators.py']
    protocol=json.loads(json.dumps(dict(version=VERSION,config=CONFIG,jobs=jobs(),data_sha256=p20.HASHES,raw_sha256={d:p['raw_sha'] for d,p in plans.items()},reference_sha256=p20.sha(refpath),historical_pool_sha256=p20.WEATHER_POOLS,source_sha256={f:p20.sha(ROOT/f) for f in deps})));cid=p20.fp(protocol)
    records=[];done=set();pr=root/'protocol.json';
    if pr.exists():p20.need(json.loads(pr.read_text(encoding='utf-8'))==protocol,'Protocol changed; do not overwrite')
    for path in sorted((root/'records').glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'));k=tuple(r['key']);p20.need(k in jobs() and k not in done,'Unknown/duplicate key');validate_record(r,k,root,results,plans,cid);records.append(r);done.add(k)
    pending=[k for k in jobs() if k not in done];print('CONFIG',cid,'completed',len(done),'pending',len(pending),flush=True)
    for d,p in plans.items():print(d,p['meta'],'fine/coarse',FINE[d],COARSE[d],flush=True)
    if a.dry_run:return
    if a.report or not pending:
        p20.need(len(records)>0,'No completed runs');summarize(records,root,plans);return
    dev=torch.device('cpu') if a.cpu else select_device(gpu_index=a.gpu_index);p20.need(not a.cpu or not torch.cuda.is_available(),'CPU mode requires CUDA hidden (set CUDA_VISIBLE_DEVICES before launch); gen_vae chooses CUDA automatically')
    env=dict(device=str(dev),gpu=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__);ep=root/'environment.json'
    if ep.exists():p20.need(json.loads(ep.read_text(encoding='utf-8'))==env,'Environment changed; do not mix devices/software')
    else:write_json(ep,env)
    if not pr.exists():write_json(pr,protocol)
    # Validate and persist all ten new pools before forecasting. Never regenerate a referenced pool.
    pools={}
    for d in DOMAINS:
        for ps in range(5):
            used=any(r['key'][0]=='replication' and r['key'][1]==d and r['key'][5]==ps and r['key'][4]=='vae' for r in records)
            pools[d,ps]=new_pool(root,d,ps,plans[d],cid,dev,create=not used)
    for k in pending[:a.limit or None]:
        study,d,mode,rep,arm,ps=k;plan=plans[d];m=plan['meta'];p=plan['pairs'];real=p['train'];path=None;ph=''
        if arm=='step_baseline':train=real
        else:
            if arm=='duplicate_real':extra=real
            else:extra,path,ph=old[arm,ps] if study=='interaction' else pools[d,ps]
            train=tuple(np.concatenate([x,y],axis=0) for x,y in zip(real,extra))
        metrics,traces,curve,samples,seconds=fit_dual(train,p['val'],p['test'],rep,m['matched_steps'],COARSE[d],FINE[d],dev,mode=mode)
        op=root/'origins'/f'{job_id(k)}.npz';idx=np.arange(plan['test_start'],plan['test_start']+m['test_windows']);atomic_npz(op,origin=idx,**{f'{pol}_{metric}':traces[pol][i] for pol in POLICIES for i,metric in enumerate(('mse','mae'))})
        r=dict(key=list(k),config_id=cid,optimizer_steps=m['matched_steps'],data_sha256=p20.HASHES[d],raw_sha256=plan['raw_sha'],pool_file=path.relative_to(results).as_posix() if path else '',pool_sha256=ph,origin_file=op.relative_to(root).as_posix(),origin_sha256=p20.sha(op),curve=curve,metrics=metrics,samples_seen=samples,seconds=seconds)
        validate_record(r,k,root,results,plans,cid);write_json(root/'records'/f'{job_id(k)}.json',r);records.append(r);print(k,metrics,flush=True)
    summarize(records,root,plans)

if __name__=='__main__':main()
