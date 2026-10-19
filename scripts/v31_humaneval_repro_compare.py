"""Sanitized, ordered first-divergence comparison of private native traces."""
import argparse
import hashlib
import json
from pathlib import Path


def differences(a,b,path=''):
    import torch
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return [dict(field=path,kind='layout',shape_a=list(a.shape),shape_b=list(b.shape))]
        if torch.equal(a,b):
            return []
        delta=(a.double()-b.double()).abs()
        return [dict(field=path,kind='tensor',different=int((a!=b).sum()),elements=a.numel(),
                     max_abs=float(delta.max()),mean_abs=float(delta.mean()))]
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in a.keys() | b.keys():
            if k=='request_id':
                continue
            field=f'{path}.{k}' if path else k
            if k not in a or k not in b:
                result.append(dict(field=field,kind='missing'))
            else:
                result.extend(differences(a[k],b[k],field))
        return result
    if a != b:
        return [dict(field=path,kind='scalar')]
    return []


def compare(left,right):
    import torch
    def rows(root):
        return [json.loads(l) for l in (root/'records.jsonl').read_text().splitlines()]
    a,b=rows(left),rows(right)
    if len(a)!=len(b):
        raise ValueError('different schedule length')
    out=[]
    fields=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','rng_seed','prompt_sha256')
    for i,(x,y) in enumerate(zip(a,b)):
        if (x['index'],x['panel_seed'])!=(y['index'],y['panel_seed']):
            raise ValueError('different schedule')
        record=dict(position=i,index=x['index'],outcome_differences=[f for f in fields if x[f]!=y[f]])
        ap,bp=left/f'private/trace/request{i+1:03d}.pt',right/f'private/trace/request{i+1:03d}.pt'
        if ap.exists() and bp.exists():
            aa,bb=torch.load(ap,map_location='cpu',weights_only=False),torch.load(bp,map_location='cpu',weights_only=False)
            events=[]
            for j,(u,v) in enumerate(zip(aa,bb)):
                d=differences(u,v)
                if d:
                    events.append(dict(event=j,kind=u['kind'],call=u.get('call'),differences=d))
            record.update(trace_events_a=len(aa),trace_events_b=len(bb),first_divergence=events[0] if events else None,
                          differing_events=events)
        out.append(record)
    return dict(left=str(left),right=str(right),records=len(out),
                outcome_differences=sum(bool(r['outcome_differences']) for r in out),comparisons=out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();result=compare(args.left,args.right)
    if args.out:
        if args.out.exists():raise RuntimeError('immutable comparison output required')
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=result['records'],outcome_differences=result['outcome_differences'],
        first_differences=[dict(index=r['index'],outcome=r['outcome_differences'],first=r.get('first_divergence'))
                           for r in result['comparisons'][8:14]]),indent=2))
