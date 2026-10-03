"""Short CUDA/NVTX trace for host launch gaps; not a latency benchmark."""
import argparse
import json
from pathlib import Path

import torch
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from .capture import SOURCE
from .cuda import Kernel
from .projection import refresh


def run(library,bridge,states):
    torch.backends.cuda.matmul.allow_tf32=False
    ordinary=Kernel(library);fast=Kernel(library,torch_library=bridge)
    policies=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    matrices=Projections();cases=[]
    for layer in (0,5):
        s=torch.load(states/f'layer{layer}.pt',map_location='cuda',weights_only=True)
        q,k,v,z,r,m=(s[x] for x in ('q','k','v','z','reference','valid'))
        packed=ordinary.pack_mask(m.contiguous());b,h,n,d=v.shape
        matrix=matrices.get(layer,h,d,'gaussian',32,1729,v.device)
        valid=m.reshape(b,h,q.shape[1]//h,q.shape[-2],n).any((2,3));norm=v.float().norm(dim=-1).square()
        start=(n-q.shape[-2])//64*64
        options=dict(mask=packed,scale=s['scale'],log_threshold=policies[s['kind']]['log_threshold'],precision='tf32x3_register',tma=True)
        # Bind loop values explicitly; queued probes must not all use layer5.
        cases.extend([(f'{s["kind"]}/native_sdpa',lambda q=q,k=k,v=v,s=s:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=s['mask'],scale=s['scale'],enable_gqa=True)),
            (f'{s["kind"]}/ctypes',lambda q=q,k=k,v=v,z=z,r=r,kw=options:ordinary(q,k,v,z,r,**kw)),
            (f'{s["kind"]}/aten',lambda q=q,k=k,v=v,z=z,r=r,kw=options:fast(q,k,v,z,r,**kw)),
            (f'{s["kind"]}/refresh',lambda v=v,matrix=matrix,z=z,norm=norm,valid=valid,start=start:refresh(v,matrix,z,norm,valid,start))])
    for _,fn in cases:
        for _ in range(3):fn()
    torch.cuda.synchronize();torch.cuda.profiler.start()
    for label,fn in cases:
        torch.cuda.nvtx.range_push(label)
        for _ in range(30):fn()
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--states',type=Path,required=True)
    a=p.parse_args();run(a.library,a.bridge,a.states)
