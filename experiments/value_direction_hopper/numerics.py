"""Separate QK rounding differences from router-state arithmetic differences."""
import argparse
import json
from pathlib import Path

import torch
from .cuda import Kernel
from .masks import pack
from .capture import SOURCE
from .validate import trusted


def run(library,states,output):
    if output.exists():raise FileExistsError(output)
    kernel=Kernel(library);rows=[]
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    for file in sorted(states.glob('layer*.pt')):
        s=torch.load(file,map_location='cuda',weights_only=True)
        q,k,v,z,r,mask=(s[key] for key in ('q','k','v','z','reference','valid'))
        threshold=policy[s['kind']]['log_threshold']
        scores=torch.empty_like(mask,dtype=torch.float32)
        actual=kernel(q,k,v,z,r,mask=pack(mask),scale=s['scale'],log_threshold=threshold,trace=True,precision='tf32x3_register',debug_scores=scores)
        ek=k.repeat_interleave(q.shape[1]//k.shape[1],1)
        expected=((q@ek.transpose(-2,-1))*s['scale']).float().masked_fill(~mask,-torch.inf)
        different=mask&(scores!=expected)
        # Float64 dot products adjudicate which BF16 rounding is nearer to the
        # exact sum. Diagnostic only: no FP64 calculation enters the router.
        exact=((q.double()@ek.double().transpose(-2,-1)).bfloat16()*s['scale']).float().masked_fill(~mask,-torch.inf)
        row=dict(layer=s['layer'],kind=s['kind'],valid=int(mask.sum()),
                 qk_disagreements=int(different.sum()),qk_max_abs=float((scores[mask]-expected[mask]).abs().max()),
                 kernel_diff_from_fp64_rounded=int((mask&(scores!=exact)).sum()),
                 torch_diff_from_fp64_rounded=int((mask&(expected!=exact)).sum()))
        # Replay the trusted retained-state equations using the captured kernel
        # scores, isolating arithmetic downstream of QK without changing masks.
        from experiments.diffusion_gemma_jl_output_aware import reference as oracle
        from experiments.diffusion_gemma_jl_output_aware.config import Config
        groups=q.shape[1]//k.shape[1];ez=z.repeat_interleave(groups,1);er=r.repeat_interleave(groups,1)
        state=[];skip=[]
        for start in range(0,q.shape[-2],128):
            st=oracle.block_statistics(scores[...,start:start+128,:],mask[...,start:start+128,:],ez)
            dropped,trace=oracle.route(st,er,Config(rank=32,log_threshold=threshold),return_trace=True)
            final=trace[-1];state.append(torch.where(dropped[...,-1,None,None],final['previous'],final['previous']+final['delta']));skip.append(dropped)
        refstate=torch.cat(state,-2);refskip=torch.stack(skip,-2)
        row['same_qk_router_mask_disagreements']=int((actual.skipped!=refskip).sum())
        row['same_qk_projected_state_max_abs']=float((actual.projected_state-refstate).abs().max())
        row['same_qk_projected_state_rel_l2']=float((actual.projected_state-refstate).norm()/refstate.norm())
        rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(cases=rows,scope='diagnostic-only QK rounding audit; no threshold adjustment'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.library,a.states,a.output)
