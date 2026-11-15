"""Read-only ASOC baseline audit, no neural training. Run from repo root.
Only writes a fresh output directory. Original arrays and historical CSV are never modified.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch

DOMAINS=('weather','electricity','traffic','epias')
PERIODS={'weather':144,'electricity':24,'traffic':24,'epias':24}
SHA={
 'weather':'bf5014da5724795b91ce862de5245018568064b86daa5ad49b207ecb76b81a03',
 'electricity':'c5fcd66ef85a284a7227fb4373a76e9c6d7469eed5f8ed6412598c8d981927ab',
 'traffic':'7b31f4925ee97956fa3cd0ca190015b43fb1a9e7193282ea9da89b56286aa576',
 'epias':'95fcd634c07847d8fbdccc2804d96aa5c3aa1b9b2eec74bd0079997893145d69'}
L,H=96,24

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require(ok,msg):
    if not ok: raise ValueError(msg)

def omega(x):
    x=np.asarray(x,dtype=float); _,ps=welch(x-x.mean(),nperseg=1024)
    p=ps[ps>0];p/=p.sum();return float(1+(p*np.log(p)).sum()/np.log(len(p)))

def diagnostics(x,t1,t2):
    ix=np.flatnonzero(x==-9999); cleaned=x.astype('float64').copy()
    for i in ix:
        require(i>0,'Leading sentinel requires explicit handling')
        cleaned[i]=cleaned[i-1]
    return {'sentinel_count':len(ix),'sentinel_indices':ix.tolist(),
      'validation_count':int(((ix>=t1)&(ix<t2)).sum()),'test_count':int((ix>=t2).sum()),
      'train_sd_original':float(x[:t1].astype('float64').std()),
      'train_sd_causal_ffill_diagnostic':float(cleaned[:t1].std()),
      'omega_full_original':omega(x),'omega_full_causal_ffill_diagnostic':omega(cleaned),
      'warning':'Causal forward-fill is a sensitivity diagnostic only; not a replacement dataset or retrained experiment.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('data/real'))
    ap.add_argument('--runs',type=Path,default=Path('results/asoc_revision/real_budget_controls.csv'))
    ap.add_argument('--out',type=Path,default=Path('adequacy_audit_v51'))
    ap.add_argument('--domains',nargs='+',choices=DOMAINS,default=list(DOMAINS))
    args=ap.parse_args(); require(not args.out.exists(),'Choose a NEW output directory')
    runs=pd.read_csv(args.runs); rows=[]; model_rows=[]; info={}
    for domain in args.domains:
        p=args.data_dir/f'{domain}.npy'; digest=sha(p); require(digest==SHA[domain],f'{domain}: historical data hash mismatch')
        a=np.load(p,allow_pickle=False); require(a.ndim==2 and a.shape[1]==3 and a.dtype==np.float32 and np.isfinite(a).all(),f'{domain}: invalid data')
        x=a[:,0]; n=len(a); t1=int(n*.7);t2=int(n*(.7+.1)); origins=np.arange(t2,n-H+1); iy=origins[:,None]+np.arange(H)
        domain_runs=runs.loc[runs.domain==domain].copy(); require(len(domain_runs)>0 and (domain_runs.data_sha256==digest).all(),f'{domain}: run hash mismatch')
        info[domain]={'sha256':digest,'data_length':n,'train_end':t1,'test_start':t2,'test_windows':len(origins),'regimes':[]}
        if domain=='weather':
            info[domain]['sentinel_diagnostics']=diagnostics(x,t1,t2)
            raw=args.data_dir/'weather'/'weather.csv'
            if raw.exists():
                df=pd.read_csv(raw); dates=pd.to_datetime(df['date'],errors='coerce'); raw_x=pd.to_numeric(df['OT'],errors='coerce').to_numpy(dtype=np.float32)
                require(np.array_equal(raw_x,x), 'Weather raw OT does not match NPY target')
                info[domain]['raw_check']={'sha256':sha(raw),'raw_target_equal':True,'duplicate_dates':int(dates.duplicated().sum()),'null_dates':int(dates.isna().sum()),'non_10min_intervals':int((dates.diff().dropna()!=pd.Timedelta(minutes=10)).sum())}
            else: info[domain]['raw_check']='Raw CSV not supplied; prepared array hash verified.'
        for regime,g in domain_runs.groupby('regime'):
            g=g[(g.lookback==L)&(g.horizon==H)]
            if g.empty: continue
            start=max(0,min(int(t1-regime*t1),t1-L-1)); train=a[start:t1]
            mu=train.mean(axis=0,keepdims=True); sd=train.std(axis=0,keepdims=True)+1e-8
            require(np.allclose(g.target_mean,float(mu[0,0]),rtol=1e-5,atol=1e-6) and np.allclose(g.target_sd,float(sd[0,0]),rtol=1e-5,atol=1e-6),f'{domain}: scaler mismatch')
            require((g.train_start==start).all() and (g.train_end==t1).all() and (g.test_windows==len(origins)).all(),f'{domain}: split mismatch')
            # Reproduce the archived scaler rather than silently re-normalizing across environments.
            m=np.float32(g.target_mean.iloc[0]); s=np.float32(g.target_sd.iloc[0]); z=((x-m)/s).astype('float64'); y=z[iy]
            train_origins=np.arange(start+L,t1-H+1); require((g.real_windows==len(train_origins)).all(),f'{domain}: training windows mismatch')
            def design(ts): return np.column_stack([np.ones(len(ts)),z[ts[:,None]-L+np.arange(L)]])
            X=design(train_origins); Y=z[train_origins[:,None]+np.arange(H)];penalty=np.eye(L+1);penalty[0,0]=0
            coef=np.linalg.solve(X.T@X+penalty,X.T@Y) # alpha=1, unpenalized intercept
            predictions={'persistence':np.broadcast_to(z[origins-1,None],y.shape),'daily_seasonal':z[iy-PERIODS[domain]],'weekly_seasonal':z[iy-7*PERIODS[domain]],'training_mean':np.zeros_like(y),'ridge96_alpha1':design(origins)@coef}
            for name,pred in predictions.items():
                e=pred-y; mse=float(np.mean(e**2));mae=float(np.mean(abs(e)))
                rows.append(dict(domain=domain,regime=regime,reference=name,lookback=L,horizon=H,test_windows=len(origins),train_start=start,train_end=t1,test_start=t2,target_mean=float(m),target_sd=float(s),mse=mse,mae=mae,mse_raw=mse*float(s)**2,mae_raw=mae*float(s),longer_history=name in ('daily_seasonal','weekly_seasonal') and (PERIODS[domain]*(7 if name=='weekly_seasonal' else 1)>L),data_sha256=digest))
            for (arch,cond),gg in g.groupby(['arch','condition']):
                model_rows.append(dict(domain=domain,regime=regime,arch=arch,condition=cond,n=len(gg),mse_median=gg.test_mse.median(),mse_min=gg.test_mse.min(),mse_max=gg.test_mse.max(),mae_median=gg.test_mae.median()))
            info[domain]['regimes'].append({'regime':float(regime),'train_start':start,'train_sentinel_count':int((x[start:t1]==-9999).sum()),'train_windows':len(train_origins)})
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out/'naive_and_ridge.csv',index=False)
    pd.DataFrame(model_rows).to_csv(args.out/'historical_model_summary.csv',index=False)
    (args.out/'provenance_and_quality.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(rows).query('regime==0.1')[['domain','reference','mse','mae']].to_string(index=False)); print('PASS: verified hashes, splits, scaler; wrote',args.out)

if __name__=='__main__': main()
