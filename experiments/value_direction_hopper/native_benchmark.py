"""Shared-state frozen-threshold validation and timing on captured native QKV."""
import argparse
import json
import math
from pathlib import Path

import torch
from .cuda import Kernel
from .masks import pack
from .validate import trusted,timed
from .capture import SOURCE


def run(states,library,output,precisions=('ieee','tf32x3'),schedule='cp'):
    if output.exists(): raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    kernel=Kernel(library)
    rows=[]
    for case in json.loads((states/'index.json').read_text())['states']:
        state=torch.load(case['path'],map_location='cuda',weights_only=True)
        q,k,v,z,reference,valid=(state[x] for x in ('q','k','v','z','reference','valid'))
        # Preserve nonzero additive bias; packing is only a lossless bool-mask path.
        bias=state['mask'] if state['mask'] is not None and state['mask'].dtype!=torch.bool else None
        mask=pack(valid) if bias is None else bias.contiguous()
        args=(q,k,v,z,reference)
        dense=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=valid if bias is None else bias,scale=state['scale'],enable_gqa=True)
        dense_us=timed(dense)
        native_dense=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=state['mask'],scale=state['scale'],enable_gqa=True)
        native_dense_us=timed(native_dense)
        for threshold in (-math.inf,policy[state['kind']]['log_threshold']):
            expected=trusted(*args,valid,threshold,scale=state['scale'],bias=bias)
            for precision in precisions:
                split=schedule=='split' and q.shape[-1]==512
                timing=torch.zeros((q.shape[0],q.shape[1],(q.shape[-2]+127)//128*(4 if split else 2),(k.shape[-2]+63)//64,kernel.timing_fields),device=q.device,dtype=torch.uint64)
                invoke=lambda **kw:kernel(*args,mask=mask,scale=state['scale'],log_threshold=threshold,precision=precision,split_pv=split,tma=schedule=='tma',**kw)
                actual=invoke(trace=True,timings=timing)
                stages=timing.float().mean((0,1,2,3)).tolist()
                if split:
                    # Half the CTAs own routing stages; the other half own PV.
                    stages=[x*2 for x in stages]
                row=dict(layer=case['layer'],kind=case['kind'],threshold=str(threshold),precision=precision,schedule=schedule,
                         shape=list(q.shape),kv_shape=list(k.shape),eligible=int(actual.eligible.sum()),skipped=int(actual.skipped.sum()),
                         mask_disagreements=int((actual.skipped!=expected['skipped']).sum()),
                         stages_mean_ns=stages,dense_reference_mask_sdpa_us=dense_us,
                         dense_native_mask_sdpa_us=native_dense_us,native_reference_masks_differ=not bool(valid.all()) if state['mask'] is None else False)
                for field in ('output','log_normalizer','projected_state'):
                    a,e=getattr(actual,field).float(),expected[field].float()
                    finite=torch.isfinite(e)
                    row[field+'_max_abs']=float((a[finite]-e[finite]).abs().max())
                    row[field+'_rel_l2']=float((a[finite]-e[finite]).norm()/e[finite].norm().clamp_min(1e-12))
                row['finite_output']=bool(torch.isfinite(actual.output).all())
                row['kernel_us']=timed(invoke)
                row['passed_shared_state_gate']=(row['mask_disagreements']==0 and row['finite_output'] and row['projected_state_max_abs']<2e-4 and row['log_normalizer_max_abs']<2e-4 and row['output_rel_l2']<.01)
                rows.append(row)
                print(json.dumps(row),flush=True)
                output.with_suffix('.jsonl').open('a').write(json.dumps(row)+'\n')
    result=dict(library=str(library),source=str(states),policy=policy,cases=rows,
                timing_fields=['qk_staging','softmax','pz_tensorcore','cluster_vote','pv_staging','risk_reduction'],
                scope='shared-state diagnostics; cached Z; not full inference or final benchmark accuracy',
                passed=all(row['passed_shared_state_gate'] for row in rows))
    output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise AssertionError('Native shared-state correctness gate failed; results preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=('ieee','tf32x3','tf32x3_register','tf32x3_shared'))
    p.add_argument('--schedule',choices=('cp','tma','split'),default='cp')
    a=p.parse_args();run(a.states,a.library,a.output,(a.precision,) if a.precision else ('ieee','tf32x3'),a.schedule)
