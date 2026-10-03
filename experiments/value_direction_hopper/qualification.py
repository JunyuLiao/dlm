"""Numerical qualification that separates dot-product rounding from routing.

The old absolute native-state gate remains preserved in native_benchmark results.
This audit uses *identical* kernel logits for the strict router/state comparison;
it does not silently relax that failed gate or claim bitwise native dense parity.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .capture import SOURCE
from .cuda import Kernel
from .validate import trusted


def run(library,states,output):
    if output.exists():raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    kernel=Kernel(library);rows=[]
    for item in json.loads((states/'index.json').read_text())['states']:
        s=torch.load(item['path'],map_location='cuda',weights_only=True)
        q,k,v,z,ref,valid=(s[key] for key in ('q','k','v','z','reference','valid'))
        mask=kernel.pack_mask(valid.contiguous())
        for threshold in (-math.inf,policy[s['kind']]['log_threshold']):
            scores=torch.empty_like(valid,dtype=torch.float32)
            actual=kernel(q,k,v,z,ref,mask=mask,scale=s['scale'],log_threshold=threshold,
                precision='tf32x3_register',tma=True,trace=True,debug_scores=scores)
            native=trusted(q,k,v,z,ref,valid,threshold,scale=s['scale'])
            replay=trusted(q,k,v,z,ref,valid,threshold,scores=scores)
            row=dict(layer=s['layer'],kind=s['kind'],threshold=str(threshold),eligible=int(actual.eligible.sum()),
                skipped=int(actual.skipped.sum()),native_mask_disagreements=int((actual.skipped!=native['skipped']).sum()),
                replay_mask_disagreements=int((actual.skipped!=replay['skipped']).sum()),
                finite_output=bool(torch.isfinite(actual.output).all()))
            for name,expected in [('native',native),('same_logits',replay)]:
                for field in ('projected_state','log_normalizer','output'):
                    a,b=getattr(actual,field).float(),expected[field].float();good=torch.isfinite(b)
                    row[name+'_'+field+'_max_abs']=float((a[good]-b[good]).abs().max())
                    row[name+'_'+field+'_rel_l2']=float((a[good]-b[good]).norm()/b[good].norm().clamp_min(1e-12))
            selected=actual.skipped.repeat_interleave(128,-2)[...,:q.shape[-2],:].repeat_interleave(64,-1)[...,:k.shape[-2]]
            has=valid.any(-1)
            p=torch.softmax(torch.where(has[...,None],scores,0.),-1).masked_fill(~valid,0.)
            row['retained_mass_sum']=float(p.masked_fill(selected,0.).sum(-1)[has].sum());row['valid_query_rows']=int(has.sum())
            ev=v.repeat_interleave(q.shape[1]//v.shape[1],1).float()
            dense=p@ev;sparse=torch.softmax(scores.masked_fill(selected,-torch.inf),-1)@ev
            row['operator_error_sq']=float((sparse-dense).square().sum());row['dense_output_sq']=float(dense.square().sum())
            row['passed']=(row['native_mask_disagreements']==0 and row['replay_mask_disagreements']==0 and
                row['finite_output'] and row['same_logits_projected_state_max_abs']<1e-4 and
                row['same_logits_log_normalizer_max_abs']<1e-4 and row['same_logits_output_rel_l2']<.01)
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(library=str(library),cases=rows,passed=all(x['passed'] for x in rows),
        coverage='one previously examined development prompt; first denoising step; layers0,5,29; all16 heads, prefix/canvas, local/global; not all model states',
        gates='0 physical-mask mismatches, finite outputs, shared-logit FP32 state/LSE max error<1e-4, BF16 online-output relative L2<1%; numerical qualification only',
        caveat='Native QK reductions and online normalized BF16 PV are not bitwise identical to the historical eager implementation. Generation must be evaluated independently.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise AssertionError('Numerical qualification failed; results preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.library,a.states,a.output)
