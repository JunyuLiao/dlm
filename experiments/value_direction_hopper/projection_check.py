"""Native-state cache-refresh numerical and timing audit."""
import argparse
import json
from pathlib import Path

import torch
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from .capture import SOURCE
from .cuda import Kernel
from .projection import refresh
from .validate import timed


def run(library,states,output):
    if output.exists():raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    matrices=Projections();kernel=Kernel(library);rows=[]
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    for item in json.loads((states/'index.json').read_text())['states']:
        s=torch.load(item['path'],map_location='cuda',weights_only=True)
        q,k,v,oldz,oldr,mask=(s[key] for key in ('q','k','v','z','reference','valid'))
        b,h,n,d=v.shape;valid=mask.reshape(b,h,q.shape[1]//h,q.shape[-2],n).any((2,3))
        matrix=matrices.get(s['layer'],h,d,'gaussian',32,1729,v.device)
        packed=kernel.pack_mask(mask.contiguous());threshold=policy[s['kind']]['log_threshold']
        expected=kernel(q,k,v,oldz,oldr,mask=packed,scale=s['scale'],log_threshold=threshold,precision='tf32x3_register',tma=True)
        for start in (0,(n-q.shape[-2])//64*64):
            z=oldz.clone();norm=v.float().norm(dim=-1).square()
            def old():
                x=v[...,start:,:].float();torch.matmul(x,matrix,out=z[...,start:,:]);norm[...,start:]=x.norm(dim=-1).square()
                return (norm.masked_fill(~valid,0.).sum(-1)/valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
            old_us=timed(old)
            fn=lambda:refresh(v,matrix,z,norm,valid,start)
            ref=fn();actual=kernel(q,k,v,z,ref,mask=packed,scale=s['scale'],log_threshold=threshold,precision='tf32x3_register',tma=True)
            row=dict(layer=s['layer'],kind=s['kind'],start=start,tokens=n-start,
                projection_max_abs=float((z-oldz).abs().max()),projection_rel_l2=float((z-oldz).norm()/oldz.norm()),
                reference_max_abs=float((ref-oldr).abs().max()),reference_max_relative=float(((ref-oldr)/oldr).abs().max()),
                mask_disagreements=int((actual.skipped!=expected.skipped).sum()),old_us=old_us,fused_us=timed(fn))
            row['passed']=row['mask_disagreements']==0 and row['projection_rel_l2']<2e-6 and row['reference_max_relative']<2e-6
            rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(cases=rows,passed=all(r['passed'] for r in rows),matrices=matrices.manifest),indent=2)+'\n')
    if not all(r['passed'] for r in rows):raise AssertionError('Projection qualification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.library,a.states,a.output)
