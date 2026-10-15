"""Sanitized analysis of private native repeatability traces."""
import argparse
from pathlib import Path
import torch
from scripts import v31_value_decision_replay as io


def error(a,b):
    if a.shape!=b.shape:raise RuntimeError('trace shapes differ')
    a=a.reshape(-1,a.shape[-1]);b=b.reshape(-1,b.shape[-1])
    total=0.;square=0.;maximum=0.;count=0;different=0;invalid=0
    for x,y in zip(a.split(16),b.split(16)):
        mask=torch.isfinite(x)&torch.isfinite(y)
        invalid+=int(((~mask)&(x!=y)).sum())
        delta=(x[mask].double()-y[mask].double()).abs()
        total+=float(delta.sum());square+=float(delta.square().sum())
        maximum=max(maximum,float(delta.max()) if delta.numel() else 0.)
        count+=delta.numel();different+=int((delta!=0).sum())
    return dict(equal=torch.equal(a,b),finite_values=count,different_finite_values=different,
        nonfinite_pattern_disagreement=invalid,mean_abs=total/count,max_abs=maximum,
        rms=(square/count)**.5,top1_row_disagreement=int((a.argmax(-1)!=b.argmax(-1)).sum()))


def same(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def components(roots,study,hashes):
    fps=[r/'parameter_fingerprints.json' for r in roots]
    if not all(f.exists() for f in fps):return None
    for f in fps:hashes[str(f.relative_to(study))]=io.sha(f)
    a,b=[io.read(f) for f in fps]
    names=sorted(a.keys()|b.keys())
    result=dict(parameter_scope='embedding, layer0 parameters/buffers and self-conditioning; not a full-model weight audit',
        parameter_entries=len(names),parameter_differences=[n for n in names if a.get(n)!=b.get(n)],layers=[])
    for phase in ('prefill','decode'):
        for layer in range(30):
            files=[root/f'{phase}_layer{layer:02d}.pt' for root in roots]
            if not all(f.exists() for f in files):raise RuntimeError('incomplete first-layer-forward capture')
            for f in files:hashes[str(f.relative_to(study))]=io.sha(f)
            a,b=[torch.load(f,map_location='cpu',weights_only=True) for f in files]
            if a['actual_tokens']!=b['actual_tokens']:raise RuntimeError('actual token counts differ')
            metrics={}
            for name in ('query','key','value','output'):
                if a[name] is None or b[name] is None:metrics[name]=dict(equal=same(a[name],b[name]))
                else:metrics[name]=error(a[name],b[name])
            keys=a['metadata'].keys()|b['metadata'].keys()
            result['layers'].append(dict(phase=phase,layer=layer,actual_tokens=a['actual_tokens'],metrics=metrics,
                metadata_differences=[k for k in sorted(keys) if not same(a['metadata'].get(k),b['metadata'].get(k))]))
    return result


def run(study):
    done=io.read(study/'final_complete.json');summary=io.read(study/'summary.json')
    if done['summary_sha256']!=io.sha(study/'summary.json'):raise RuntimeError('diagnostic summary changed')
    if not all(r['identical_repeated_output'] for r in summary['results']):
        raise RuntimeError('observed/unobserved repetition differs; do not interpret observer traces')
    roots=[];hashes={}
    for job in ('original_repeat001','original_repeat002'):
        markers=list((study/job).glob('attempt*/attempt_complete.json'))
        if len(markers)!=1:raise RuntimeError('unique completed original process required')
        marker=markers[0];leaf=marker.parent;proof=io.read(marker)
        if proof['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('native trace records changed')
        hashes[str(marker.relative_to(study))]=io.sha(marker)
        roots.append(leaf/'private/trace')
    rows=[]
    for call in range(2):
        files=[root/f'call{call:03d}.pt' for root in roots]
        for f in files:hashes[str(f.relative_to(study))]=io.sha(f)
        a,b=[torch.load(f,map_location='cpu',weights_only=True) for f in files]
        r=dict(call=call,logits=error(a['before']['logits'],b['before']['logits']),
               scaled=error(a['after']['scaled'],b['after']['scaled']))
        for phase in ('before','after'):
            r[phase]=dict(cuda_rng_equal=torch.equal(a[phase]['cuda_rng'],b[phase]['cuda_rng']),
                cpu_rng_equal=torch.equal(a[phase]['cpu_rng'],b[phase]['cpu_rng']),
                canvas_token_disagreements=int((a[phase]['canvas']!=b[phase]['canvas']).sum()))
            if phase=='after':r[phase]['argmax_row_disagreements']=int((a[phase]['argmax']!=b[phase]['argmax']).sum())
        rows.append(r)
        del a,b
    layer_report=components(roots,study,hashes)
    result=dict(status='complete',rows=rows,input_sha256=hashes,
        diagnostic_complete_sha256=io.sha(study/'final_complete.json'),
        analysis_source_sha256=io.sha(__file__),observer_qualification='within-process observed and unobserved repeats are identical',
        scope='two original V31 processes; first two measured denoise calls; no value refinement or extra forward',timing='paused')
    if layer_report is not None:result['components']=layer_report
    dest=study/'rng_localization';io.write(dest/'summary.json',result,True)
    io.write(dest/'analysis_complete.json',dict(status='complete',summary_sha256=io.sha(dest/'summary.json')),True)
    print({k:v for k,v in result.items() if k in ('status','rows','observer_qualification')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    torch.set_num_threads(1)
    run(p.parse_args().study.resolve())
