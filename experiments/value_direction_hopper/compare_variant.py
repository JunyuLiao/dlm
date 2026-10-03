"""Pair an isolated scheduling variant with the frozen kernel on identical QKV."""
import argparse
import json
import math
from pathlib import Path

import torch

from .capture import SOURCE
from .cuda import Kernel
from .validate import timed


def run(base,variant,states,output):
    if output.exists():raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    baseline=Kernel(base);new=Kernel(variant);rows=[]
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    for item in json.loads((states/'index.json').read_text())['states']:
        s=torch.load(item['path'],map_location='cuda',weights_only=True)
        q,k,v,z,r,m=(s[x] for x in ('q','k','v','z','reference','valid'))
        args=q,k,v,z,r;mask=baseline.pack_mask(m.contiguous())
        for threshold in (-math.inf,policy[s['kind']]['log_threshold']):
            options=dict(mask=mask,scale=s['scale'],log_threshold=threshold,precision='tf32x3_register',tma=True)
            a=baseline(*args,trace=True,**options);b=new(*args,trace=True,**options)
            row=dict(layer=s['layer'],kind=s['kind'],threshold=str(threshold),eligible=int(a.eligible.sum()),
                skipped_base=int(a.skipped.sum()),skipped_variant=int(b.skipped.sum()),fields={})
            for field in ('output','skipped','eligible','log_normalizer','projected_state','risk'):
                x,y=getattr(a,field),getattr(b,field)
                equal=(x==y)|(torch.isnan(x)&torch.isnan(y)) if x.is_floating_point() else x==y
                values=dict(different=int((~equal).sum()),total=x.numel())
                if x.is_floating_point():
                    good=torch.isfinite(x)&torch.isfinite(y)
                    values.update(max_abs=float((x[good].float()-y[good].float()).abs().max()),
                                  rel_l2=float((x[good].float()-y[good].float()).norm()/x[good].float().norm().clamp_min(1e-12)))
                row['fields'][field]=values
            # Alternate order; each measurement has the same graph/load warmup.
            times={'base':[],'variant':[]}
            for order in (('base','variant'),('variant','base'),('base','variant')):
                for name in order:
                    fn=baseline if name=='base' else new
                    times[name].append(timed(lambda:fn(*args,**options)))
            row['timings_us']=times;row['base_us']=sorted(times['base'])[1];row['variant_us']=sorted(times['variant'])[1]
            row['speedup']=row['base_us']/row['variant_us']
            row['passed']=row['fields']['skipped']['different']==0 and row['fields']['eligible']['different']==0 and row['fields']['output']['rel_l2']<.001
            rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(base=str(base),variant=str(variant),cases=rows,passed=all(r['passed'] for r in rows),
        scope='Scheduling/compiler operand experiment on same native development QKV; thresholds unchanged; cached projections; no final score tuning'),indent=2)+'\n')
    if not all(r['passed'] for r in rows):raise AssertionError('Variant changed masks/output beyond qualification gate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--variant',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.variant,a.states,a.output)
