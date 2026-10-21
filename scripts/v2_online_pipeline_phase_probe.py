"""Diagnostic device timestamps; excluded from clean performance comparisons."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keys',type=int,default=4096);p.add_argument('--schedule',type=int,default=0);a=p.parse_args()
    if a.out.exists():raise RuntimeError('new attempt required')
    a.out.mkdir(parents=True);started=time.monotonic()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    if any(int(x.split(',')[0])!=os.getpid() for x in apps.splitlines() if x.strip()):raise RuntimeError('foreign GPU worker')
    from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel,Params
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    op=Kernel(a.library)
    if not op.options.get('profile'):raise ValueError('diagnostic phase-timestamp build required')
    nk=a.keys;n=256;h=16;hk=2;g=torch.Generator(device='cuda').manual_seed(nk)
    q=torch.randn((1,h,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,hk,nk,512),device='cuda',dtype=torch.bfloat16,generator=g);v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
    z=torch.empty((1,hk,nk,32),device='cuda');norm=torch.empty((1,hk,nk),device='cuda')
    matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device);nu=project(v,matrix,z,norm,torch.ones_like(norm,dtype=torch.bool),0)
    scale=512**-.5;pt=(nk-n)//64;tiles=(nk+63)//64
    lo,hi=0.,2.;best=None
    for _ in range(13):
        tau=(lo+hi)/2;keep=discover(q,k,z,nu,scale,pt,tau,ratio=True)[0];count=int(keep.sum());err=abs(count-keep.numel()*.5)
        if best is None or err<best[0]:best=(err,tau,keep)
        if count>keep.numel()*.5:lo=tau
        else:hi=tau
    tau=best[1];out=torch.empty_like(q);skipped=torch.empty((1,h,2,tiles),device='cuda',dtype=torch.bool);eligible=torch.empty_like(skipped)
    lm=torch.empty((1,h,n),device='cuda');state=torch.empty((1,),device='cuda');risks=torch.empty((1,),device='cuda')
    timings=torch.zeros((h,4,tiles,6),device='cuda',dtype=torch.int64)
    params=Params(*(x.data_ptr() if x is not None else 0 for x in (q,k,v,z,nu,None,out,skipped,eligible,lm,state,risks)),
        1,h,hk,n,nk,512,0,1,1,0,scale,__import__('math').log(tau),timings.data_ptr(),2,0,0,0,0,0.,pt)
    for _ in range(5):
        status=op.fn(ct.byref(params),torch.cuda.current_stream().cuda_stream,a.schedule)
        if status:raise RuntimeError(status)
    torch.cuda.synchronize();keep=~skipped
    if int((keep.reshape_as(best[2])!=best[2]).sum()):raise RuntimeError('profile mask mismatch')
    expected=v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(keep),scale)
    rel=float((out.transpose(1,2).float()-expected.float()).norm()/expected.float().norm())
    values=timings[...,:5].cpu().float();kept=keep[0].repeat_interleave(2,dim=1).cpu()
    names=['qk_softmax_probability_staging','projected_pv_and_partial_risk','q128_vote','conditional_full_pv','next_k_load_and_sync']
    phases={}
    for i,name in enumerate(names):
        x=values[...,i].reshape(-1)
        phases[name]=dict(mean_ns=float(x.mean()),median_ns=float(x.median()),p95_ns=float(x.quantile(.95)),
            kept_mean_ns=float(values[...,i][kept].mean()),skipped_mean_ns=float(values[...,i][~kept].mean()))
    (a.out/'receipt.json').write_text(json.dumps(dict(status='complete',diagnostic_only=True,instrumented=True,
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),library_options=op.options,queries=n,keys=nk,schedule=a.schedule,
        threshold=tau,eligible=keep.numel(),kept=int(keep.sum()),same_mask_fa4_relative_l2=rel,phases=phases,
        reservation_seconds=time.monotonic()-started,method='thread0 globaltimer differences in instrumented kernel; not a clean timing or hardware-stall attribution'),indent=2)+'\n')
    print(json.dumps(phases),flush=True)


if __name__=='__main__':main()
