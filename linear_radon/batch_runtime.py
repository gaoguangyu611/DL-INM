"""30-sample recording; delegate every numerical solve to the original code."""
from pathlib import Path
import csv,copy,json,pickle,time,os,hashlib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import solver_runtime as original

ROOT=Path('results')
ROOT.mkdir(exist_ok=True)
CURRENT_SAMPLE=None
DL_TRACE=[]
compute_sk=original.compute_sk
get_operator=original.get_operator
get_dataset=original.get_dataset
get_pinv=original.get_pinv
warmup=original.warmup

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def progress(stage,**kwargs):
    atomic_json(ROOT/'progress.json',dict(stage=stage,sample=CURRENT_SAMPLE,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs))

def begin_method(method):
    original.begin_method(method)
    if method=='DL_INM':DL_TRACE.clear()
    progress('running_method',method=method)

def set_current_sample(index):
    global CURRENT_SAMPLE
    CURRENT_SAMPLE=int(index)

def record_dl_iteration(k,alpha,beta,scale,norm_s,norm_raw,norm_U,residual,RE):
    DL_TRACE.append(dict(k=int(k),alpha=float(alpha),beta=float(beta),scale_factor=float(scale),
                         norm_s=float(norm_s),norm_N=float(norm_raw),norm_U=float(norm_U),
                         R_k=float(beta*norm_U/max(alpha*norm_s,1e-300)),
                         residual=float(residual),RE=float(RE),
                         inner=copy.deepcopy(original.AUDIT.get('DL_INM',[{}])[-1])))

def record_partial(results,index,method):
    folder=ROOT/'learning';folder.mkdir(exist_ok=True)
    with (folder/('sample_%02d.partial.pkl'%index)).open('wb') as f:pickle.dump(results,f,protocol=4)
    atomic_json(folder/('sample_%02d.partial.json'%index),dict(last_method=method,inner_audit=copy.deepcopy(original.AUDIT),DL_trace=copy.deepcopy(DL_TRACE)))
    progress('method_completed',method=method)

METHODS={'learning':{'IN':'IN','res':'IN-RES','dc':'IN-DC','DL':'DL-INM','iNETT':'iNETT'},
         'traditional':{'IN':'IN','DL':'DL-INM','Landweber':'Landweber','TV':'IN-TV'}}

def checkpoint_sample(group,idx,results,position,dataset_path,tau):
    folder=ROOT/group;folder.mkdir(exist_ok=True)
    one={k:[copy.deepcopy(v[position])] for k,v in results.items() if isinstance(v,list) and len(v)>position}
    one['indices']=[int(idx)]
    with (folder/('sample_%02d.pkl'%idx)).open('wb') as f:pickle.dump(one,f,protocol=4)
    data=get_dataset(dataset_path);gt=data['xk'][idx].reshape(-1).astype(np.float64)
    y=data['ydelta'][idx].reshape(-1).astype(np.float64);R=get_operator()
    delta=float(np.linalg.norm(R@gt-y));dr=float(gt.max()-gt.min()) or 1.
    rows=[]
    for pre,label in METHODS[group].items():
        x=np.asarray(one[pre+'_recons'][0],dtype=np.float64).reshape(-1)
        history=one[pre+'_error'][0];N=int(one[pre+'_steps'][0])
        if pre=='Landweber':N=len(history)-1
        outer=N-1 if pre=='res' else N-2 if pre=='dc' else N
        inner=int(one.get(pre+'_inner_total',[0])[0])
        residual=float(np.linalg.norm(R@x-y));b=gt.reshape(128,128);a=x.reshape(128,128)
        row=dict(group=group,sample=int(idx),source_index=int(data['source_indices'][idx]),method=label,
                 N=N,newton_outer=outer if pre not in ['Landweber','iNETT'] else None,
                 postprocess_stages=N-outer,RE=float(np.linalg.norm(x-gt)/np.linalg.norm(gt)),
                 PSNR=float(peak_signal_noise_ratio(b,a,data_range=dr)),
                 SSIM=float(structural_similarity(b,a,data_range=dr)),
                 PSNR_unit_range=float(peak_signal_noise_ratio(b,a,data_range=1.)),
                 SSIM_unit_range=float(structural_similarity(b,a,data_range=1.)),
                 data_range=dr,Time=float(one[pre+'_time'][0]),CG_total=inner if pre not in ['Landweber','iNETT'] else 0,
                 Adam_total=inner if pre=='iNETT' else 0,
                 residual=residual,delta=delta,threshold=float(tau*delta),discrepancy_met=residual<=tau*delta,
                 final_history_RE=float(history[-1]),returned_RE_matches_last=bool(np.isclose(np.linalg.norm(x-gt)/np.linalg.norm(gt),history[-1],atol=1e-8)))
        rows.append(row)
    trace=copy.deepcopy(DL_TRACE)
    audit=copy.deepcopy(original.AUDIT)
    if group=='traditional':
        ref=json.loads((ROOT/'learning'/('sample_%02d.json'%idx)).read_text())
        trace=ref['DL_trace']
        audit={'IN':ref['inner_audit'].get('res',[]),
               'DL-INM':ref['inner_audit'].get('DL_INM',[]),
               'Landweber':[], 'IN-TV':audit.get('IN-TV',[])}
    atomic_json(folder/('sample_%02d.json'%idx),dict(rows=rows,DL_trace=trace,inner_audit=audit))
    progress('sample_completed',group=group)
    summarize()
    print('CHECKPOINT_SAVED',group,idx,flush=True)

def learning_batch(one_function,model_path,model_path_res,model_path_iNETT,dataset_path,out_max,tau,device,num_samples,selected_indices):
    global CURRENT_SAMPLE
    indices=list(range(num_samples)) if selected_indices is None else list(selected_indices)
    get_pinv()  # shared offline operator preparation outside every method timer
    merged={}
    for idx in indices:
        CURRENT_SAMPLE=idx
        path=ROOT/'learning'/('sample_%02d.pkl'%idx)
        if path.exists():
            with path.open('rb') as f:one=pickle.load(f)
            print('RESUME_COMPLETED',idx,flush=True)
        else:
            one=one_function(model_path,model_path_res,model_path_iNETT,dataset_path,out_max,tau,device,sample_index=idx)
            checkpoint_sample('learning',idx,one,0,dataset_path,tau)
        for k,v in one.items():merged.setdefault(k,[]).extend(v)
    with (ROOT/'learning/results_all.pkl').open('wb') as f:pickle.dump(merged,f,protocol=4)
    return merged

def reuse_learning_method(results,idx,prefix):
    with (ROOT/'learning'/('sample_%02d.pkl'%idx)).open('rb') as f:one=pickle.load(f)
    for key in list(results):
        if key.startswith(prefix+'_'):
            source='model_item_norm' if key=='DL_model_norm' else key
            if source in one:results[key].append(copy.deepcopy(one[source][0]))

def summarize():
    rows=[];traces=[]
    for group in METHODS:
        for path in sorted((ROOT/group).glob('sample_*.json')):
            if '.partial.' in path.name:continue
            obj=json.loads(path.read_text());rows.extend(obj['rows'])
            if group=='learning':
                for trace in obj['DL_trace']:traces.append(dict(sample=obj['rows'][0]['sample'],source_index=obj['rows'][0]['source_index'],**{k:v for k,v in trace.items() if k!='inner'}))
    if rows:
        with (ROOT/'results_per_sample.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for group in METHODS:
        for label in METHODS[group].values():
            subset=[r for r in rows if r['group']==group and r['method']==label]
            if not subset:continue
            item=dict(group=group,method=label,samples=len(subset),discrepancy_count=sum(r['discrepancy_met'] for r in subset))
            for metric in ['N','RE','PSNR','SSIM','Time','CG_total','Adam_total','newton_outer','residual']:
                v=[r[metric] for r in subset if r[metric] is not None]
                item[metric+'_mean']=float(np.mean(v)) if v else None
                item[metric+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else None
            summaries.append(item)
    atomic_json(ROOT/'summary.json',summaries)
    if summaries:
        with (ROOT/'summary_mean_std.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    if traces:
        with (ROOT/'DL_scale_factors_per_iteration.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(traces[0]));w.writeheader();w.writerows(traces)
    lines=['# 30-sample comparison: mean ± sample standard deviation','',
           'N is the iteration/stage count returned by the code. IN-RES/IN-DC include 1/2 postprocessing stages. Sample count is separate (30).','']
    for group in METHODS:
        lines+=['## '+group,'','| Method | N | RE | PSNR (dB) | SSIM | Time (s) |','|---|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['group']!=group:continue
            cells=[]
            for m in ['N','RE','PSNR','SSIM','Time']:
                std=r[m+'_std'];cells.append('%.4f ± %.4f'%(r[m+'_mean'],std) if std is not None else '%.4f'%r[m+'_mean'])
            lines.append('| '+r['method']+' | '+' | '.join(cells)+' |')
        lines.append('')
    (ROOT/'paper_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    return rows,summaries
