"""Timing/checkpoints for copied nonlinear comparison code; no algorithm tuning."""
from pathlib import Path
from functools import lru_cache
import json,time,pickle,csv,hashlib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from data.tikhonov_audited import compute_sk,get_cg_records,reset_cg_records

ROOT=Path('results')
GROUP=None;SAMPLE=None;BRANCH=None
_clock=0.0;_accum=0.0;_last_mark=0.0;_trace=[];_warmed=set()

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def warmup(model,device,two_inputs=False):
    if id(model) in _warmed:return
    with torch.no_grad():
        z=torch.zeros((1,1,128,128),device=device)
        model(z,z) if two_inputs else model(z)
    sync(device);_warmed.add(id(model))
@lru_cache(maxsize=1)
def get_operator():return np.load('R_limited.npy',mmap_mode='r')
@lru_cache(maxsize=4)
def get_dataset(path):return np.load(path)
@lru_cache(maxsize=2)
def get_torch_operator(device):
    # Immutable, shared operator conversion is preparation, not a per-sample iteration.
    return torch.tensor(np.asarray(get_operator()),dtype=torch.float32,device=device)
@lru_cache(maxsize=1)
def get_pinv():
    started=time.perf_counter();own=ROOT/'R_pinv.npy'
    sibling=Path('../linear_radon/cache/R_pinv.npy')
    meta=sibling.parent/'pinv_preparation.json'
    reused=False
    if sibling.exists() and meta.exists():
        m=json.loads(meta.read_text())
        if m['R_sha256']==sha256('R_limited.npy') and m['rcond']==1e-15 and m['numpy_version']==np.__version__:
            path=sibling;reused=True
    if not reused:
        path=own
        if not path.exists():np.save(path,np.linalg.pinv(get_operator()))
    matrix=np.load(path,mmap_mode='r')
    assert matrix.shape==(16384,7680) and matrix.dtype==np.float64
    dump_json(ROOT/'pinv_preparation.json',{'path':str(path),'reused_verified_linear_cache':reused,
        'operator_sha256':sha256('R_limited.npy'),'rcond':1e-15,'numpy_version':np.__version__,
        'preparation_seconds':time.perf_counter()-started,'shape':list(matrix.shape),
        'note':'One shared original numpy pinv, outside iteration timing'})
    return matrix

def configure(group,sample,branch):
    global GROUP,SAMPLE,BRANCH
    GROUP=group;SAMPLE=int(sample);BRANCH=branch
    (ROOT/group).mkdir(parents=True,exist_ok=True)
def begin_method(example,device):
    global _clock,_accum,_last_mark,_trace
    reset_cg_records();_trace=[];_accum=0.0;_last_mark=0.0
    sync(device);_clock=time.perf_counter()
def _pause(device):
    global _accum
    sync(device);_accum+=time.perf_counter()-_clock
    return _accum
def _resume():
    global _clock
    _clock=time.perf_counter()
def record_dl_step(k,gamma,sigma,beta,scale,s_norm,raw_norm,used,residual):
    used_norm=float(torch.norm(used).item());denom=float(gamma*s_norm)
    _trace.append({'group':GROUP,'sample':SAMPLE,'outer_iteration':int(k),'alpha_outer':float(gamma),
        'sigma_inner':float(sigma),'beta':float(beta),'scale_factor':float(scale.item()),'s_norm':float(s_norm),
        'raw_Ntheta_norm':float(raw_norm.item()),'used_Utheta_norm':used_norm,
        'R_k':float(beta*used_norm/denom) if denom>0 else None,'residual_before':float(residual),
        'CG_total_so_far':sum(r['CG_steps'] for r in get_cg_records())})

def _load_bundle():
    path=ROOT/GROUP/('sample_%02d.pkl'%SAMPLE)
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    return {'group':GROUP,'sample':SAMPLE,'rows':[],'branches':{},'DL_trace':[],'CG_audit':{}}
def save_bundle(bundle):
    p=ROOT/GROUP/('sample_%02d'%SAMPLE)
    with p.with_suffix('.pkl').open('wb') as f:pickle.dump(bundle,f,protocol=4)
    dump_json(p.with_suffix('.json'),{'group':GROUP,'sample':SAMPLE,'rows':bundle['rows'],
        'scale_factors':[r['scale_factor'] for r in bundle['DL_trace']],
        'DL_trace':bundle['DL_trace'],'CG_audit':bundle['CG_audit']})
def _record(name,results,recon_key,step_key,error_key,residual_key,R,y,delta,threshold,seconds,
            newton_steps,postprocess_steps=0,strict=False,incremental_seconds=None):
    gt=np.asarray(results['original_images'][0]).reshape(128,128)
    x=np.asarray(results[recon_key][0]).reshape(128,128)
    cg=get_cg_records();actual_steps=int(newton_steps+postprocess_steps)
    residual=float(np.max(np.linalg.norm((((R@x.reshape(-1))**2-y).reshape(60,128)),axis=1)))
    dr=float(gt.max()-gt.min()) or 1.0
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite reconstruction: '+name)
    row={'group':GROUP,'sample':SAMPLE,'source_index':SAMPLE,'method':name,'status':'complete',
        'relative_error':float(np.linalg.norm(x.reshape(-1)-gt.reshape(-1))/np.linalg.norm(gt.reshape(-1))),
        'PSNR':float(peak_signal_noise_ratio(gt,x,data_range=dr)),
        'SSIM':float(structural_similarity(gt,x,data_range=dr)),'metric_data_range':dr,
        'outer_iterations':actual_steps,'newton_outer_iterations':int(newton_steps) if name not in ['NETT','Landweber'] else 0,
        'postprocess_steps':int(postprocess_steps),'CG_total':sum(r['CG_steps'] for r in cg),
        'sigma_trials':len(cg),'seconds':float(seconds),'incremental_seconds':float(seconds if incremental_seconds is None else incremental_seconds),
        'residual':residual,'delta':float(delta),'threshold':float(threshold),
        'discrepancy_met':bool(residual<threshold if strict else residual<=threshold),
        'Cbeta':0.002 if name=='DL-INM' else None,'L':0.4 if name=='DL-INM' else None,
        'stop_reason':('discrepancy' if residual<=threshold else ('postprocessing_after_IN' if postprocess_steps else 'iteration_limit'))}
    bundle=_load_bundle()
    assert name not in [r['method'] for r in bundle['rows']],('Already checkpointed',name)
    bundle['rows'].append(row);bundle['branches'][BRANCH]=results
    bundle['CG_audit'][name]=cg
    if name=='DL-INM':
        assert len(_trace)==newton_steps
        trace=[dict(r) for r in _trace]
        for i,r in enumerate(trace):
            r['residual_after']=float(results[residual_key][0][i+1]);r['RE_after']=float(results[error_key][0][i+1])
        bundle['DL_trace']=trace
    save_bundle(bundle)
    print('METHOD_CHECKPOINT',GROUP,SAMPLE,name,json.dumps(row),flush=True)
    return row

def finish_traditional(example,results,R,y,delta,threshold,device):
    seconds=_pause(device)
    keys={'IN':('IN','Inexact_Newton_recons','Inexact_Newton_steps','error_IN','residual_IN'),
        'res':('DL-INM','res_recons','res_steps','error_res','residual_res'),
        'Landweber':('Landweber','Landweber_recons','Landweber_steps','error_Landweber','residual_Landweber'),
        'IN-TV':('IN-TV','TV_recons','TV_steps','error_TV','residual_TV')}
    name,r,s,e,h=keys[example];steps=len(results[e][0])-1
    _record(name,results,r,s,e,h,R,y,delta,threshold,seconds,steps,strict=example=='Landweber')

def mark_learning_stage(name,results,R,y,delta,threshold,device):
    global _last_mark
    seconds=_pause(device);insteps=int(results['IN-steps'][0])
    keys={'IN':('IN-recons','IN-steps','IN-error','IN_residual',0),
        'IN-RES':('res_recons','res_steps','res_error','res_residual',1),
        'IN-DC':('dc_recons','res_steps','res_error','res_residual',2)}
    r,s,e,h,post=keys[name]
    _record(name,results,r,s,e,h,R,y,delta,threshold,seconds,insteps,postprocess_steps=post,
        incremental_seconds=seconds-_last_mark)
    _last_mark=seconds;_resume()
def finish_learning(name,results,R,y,delta,threshold,device):
    seconds=_pause(device)
    prefix='DL' if name=='DL-INM' else 'NETT'
    steps=int(results[prefix+'_steps'][0])
    _record(name,results,prefix+'_recons',prefix+'_steps',prefix+'_error',prefix+'_residual',R,y,delta,threshold,seconds,steps)

def record_failure(group,sample,branch,error):
    configure(group,sample,branch);bundle=_load_bundle()
    labels={'traditional':{'IN':['IN'],'res':['DL-INM'],'Landweber':['Landweber'],'IN-TV':['IN-TV']},
        'learning':{'res':['IN','IN-RES','IN-DC'],'DL_INM':['DL-INM'],'NETT':['NETT']}}
    for method in labels[group][branch]:
        if method in [r['method'] for r in bundle['rows']]:continue
        row={'group':group,'sample':int(sample),'source_index':int(sample),'method':method,'status':'failed',
            'relative_error':None,'PSNR':None,'SSIM':None,'metric_data_range':None,'outer_iterations':None,
            'newton_outer_iterations':None,'postprocess_steps':None,'CG_total':sum(r['CG_steps'] for r in get_cg_records()),
            'sigma_trials':len(get_cg_records()),'seconds':None,'incremental_seconds':None,'residual':None,
            'delta':None,'threshold':None,'discrepancy_met':False,'Cbeta':0.002 if method=='DL-INM' else None,
            'L':0.4 if method=='DL-INM' else None,'stop_reason':error}
        bundle['rows'].append(row)
    save_bundle(bundle)
