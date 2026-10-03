"""Reproducible H100 control-path tests and CUDA-event timings."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.diffusion_gemma_jl_output_aware import reference as oracle
from experiments.diffusion_gemma_jl_output_aware.config import Config
from .fused import attention


def inputs(nq=128, nk=257, d=256, h=4, hk=2, seed=42, local=False):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn((1,h,nq,d), device='cuda', dtype=torch.bfloat16, generator=gen)
    k = torch.randn((1,hk,nk,d), device='cuda', dtype=torch.bfloat16, generator=gen)
    v = torch.randn(k.shape, device='cuda', dtype=torch.bfloat16, generator=gen)
    r = torch.randn((hk,d,32), device='cuda', dtype=torch.float32, generator=gen)/math.sqrt(32)
    z = (v.float()@r).contiguous()
    mask = torch.ones((1,1,nq,nk), device='cuda', dtype=torch.bool)
    if local:
        qp = torch.arange(nq,device='cuda')+nk-nq
        kp = torch.arange(nk,device='cuda')
        mask = (kp[None,:] >= qp[:,None]-128+1)[None,None]
    validkv = mask.any(-2)
    ref = (v.float().norm(dim=-1).square().masked_fill(~validkv,0.).sum(-1)/validkv.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
    return q,k,v,z,ref,mask


def trusted(q,k,v,z,ref,mask,threshold,*,scale=None,bias=None,scores=None):
    gqa = q.shape[1]//k.shape[1]
    ek,ev,ez,er = (x.repeat_interleave(gqa,1) for x in (k,v,z,ref))
    if scores is None:
        scores = (q@ek.transpose(-2,-1))*(q.shape[-1]**-.5 if scale is None else scale)
        if bias is not None: scores=scores+bias
    scores=scores.masked_fill(~mask,-torch.inf)
    states, masks, risks, projected = [],[],[],[]
    config = Config(rank=32,log_threshold=threshold)
    for start in range(0,q.shape[-2],128):
        st = oracle.block_statistics(scores[...,start:start+128,:],mask[...,start:start+128,:],ez)
        skip,trace = oracle.route(st,er,config,return_trace=True)
        masks.append(skip)
        risks.append(torch.stack([t['worst'] for t in trace],-1))
        states.append(st)
        last = trace[-1]
        proposed = last['previous']+last['delta']
        projected.append(torch.where(skip[...,-1,None,None],last['previous'],proposed))
    skip = torch.stack(masks,-2)
    selected = skip.repeat_interleave(128,-2)[...,:q.shape[-2],:].repeat_interleave(64,-1)[...,:k.shape[-2]]
    allowed = mask & ~selected
    has = allowed.any(-1)
    ss = scores.float().masked_fill(~allowed,-torch.inf)
    probs = torch.softmax(torch.where(has[...,None],ss,0.),-1).masked_fill(~allowed,0.)
    output = probs.to(v.dtype)@ev
    lse = torch.logsumexp(ss,-1)
    return dict(output=output,skipped=skip,log_normalizer=lse,
                projected_state=torch.cat(projected,-2),risk=torch.stack(risks,-2))


def timed(fn, repeats=30):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    # Graph the whole measurement batch so Python allocation/launch gaps do not
    # get misreported as kernel time on microsecond-sized attention calls.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats): fn()
    # Stabilize clock/load before comparing short dense calls to longer sparse
    # calls. The warm period is excluded, and applies identically to all paths.
    warm_until=time.perf_counter()+.2
    while time.perf_counter()<warm_until:
        graph.replay();torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    times=[]
    for _ in range(5):
        start.record()
        graph.replay()
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end)*1000/repeats)
    return sorted(times)[len(times)//2]


def run(quick=False, journal=None, cuda_library=None):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cases=[]
    shapes=[(63,131,256 if cuda_library else 64,4,2,False),(129,257,256,4,2,True),
            (256,1024,256,16,8,True),(256,4096,512,16,2,False)]
    if quick: shapes=shapes[:2]
    kernel=None
    if cuda_library:
        from .cuda import Kernel
        kernel=Kernel(cuda_library)
    for nq,nk,d,h,hk,local in shapes:
        q,k,v,z,ref,mask=inputs(nq,nk,d,h,hk,local=local)
        for threshold in (-math.inf,math.log(.15)):
            expected=trusted(q,k,v,z,ref,mask,threshold)
            for precision in (('cuda_serial','cuda_overlap') if kernel else ('ieee','tf32x3')):
                started=time.monotonic()
                invoke=(lambda trace:kernel(q,k,v,z,ref,mask=mask,log_threshold=threshold,trace=trace,overlap=precision=='cuda_overlap')) if kernel else (
                    lambda trace:attention(q,k,v,z,ref,mask=mask,log_threshold=threshold,trace=trace,precision=precision))
                actual=invoke(True)
                torch.cuda.synchronize()
                row=dict(shape=list(q.shape),kv_length=nk,local=local,threshold=str(threshold),precision=precision,
                    compile_and_test_s=time.monotonic()-started,
                    skipped=int(actual.skipped.sum()),eligible=int(actual.eligible.sum()),
                    mask_disagreements=int((actual.skipped!=expected['skipped']).sum()))
                for field in ('output','log_normalizer','projected_state'):
                    a,e=getattr(actual,field).float(),expected[field].float()
                    finite=torch.isfinite(e)
                    row[field+'_max_abs']=float((a[finite]-e[finite]).abs().max()) if finite.any() else 0.
                    row[field+'_relative_l2']=float((a[finite]-e[finite]).norm()/e[finite].norm().clamp_min(1e-12))
                row['finite_output']=bool(torch.isfinite(actual.output).all())
                row['us']=timed(lambda:invoke(False))
                cases.append(row)
                if journal is not None:
                    with journal.open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    return dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cases=cases,
                purpose='fused control, not production acceptance',time=time.time())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda-library',type=Path)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=run(args.quick,args.output.with_suffix('.jsonl'),args.cuda_library)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
