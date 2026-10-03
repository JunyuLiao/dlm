"""Dense-dispatch audit and matched kernel benchmark, no inferred speedup."""
import argparse
import json
import math
from pathlib import Path

import torch
from .cuda import Kernel
from .validate import inputs, timed, trusted


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--keys',type=int,default=1024)
    p.add_argument('--queries',type=int,default=256)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    q,k,v,z,ref,mask=inputs(args.queries,args.keys,args.width,16,8 if args.width==256 else 2)
    kernel=Kernel(args.library)
    call=lambda **kw:kernel(q,k,v,z,ref,mask=mask,**kw)
    if args.profile:
        call(log_threshold=math.log(.4));torch.cuda.synchronize();return
    result=dict(shape=list(q.shape),keys=args.keys,kernel_library=str(args.library),timing='CUDA event median of five 30-call CUDA graph replays, cached sketches, synthetic QKV',cases=[])
    dense=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        dense();torch.cuda.synchronize()
    result['dense_dispatch']=[x.key for x in prof.key_averages() if 'attention' in x.key.lower() or 'fmha' in x.key.lower()]
    result['sdpa_us']=timed(dense)
    try:
        from flash_attn.cute.interface import flash_attn_func
        cq,ck,cv=(x.transpose(1,2) for x in (q,k,v))
        flash=lambda:flash_attn_func(cq,ck,cv,causal=False)
        flash();torch.cuda.synchronize()
        result['flash_sm90_us']=timed(flash)
        result['flash_backend']='installed flash_attn.cute Hopper WGMMA'
    except Exception as exc:
        result['flash_failure']=repr(exc)
    for mode in ('dense','blasst','value'):
        for threshold in ((-math.inf,) if mode=='dense' else ((-.5,1.) if mode=='blasst' else (math.log(.12),math.log(.2),math.log(.4)))):
            for overlap in (False,True):
                actual=call(mode=mode,log_threshold=threshold,overlap=overlap)
                torch.cuda.synchronize()
                us=timed(lambda:call(mode=mode,log_threshold=threshold,overlap=overlap))
                row=dict(mode=mode,threshold=str(threshold),overlap=overlap,us=us,
                         eligible=int(actual.eligible.sum()),skipped=int(actual.skipped.sum()),sdpa_speedup=result['sdpa_us']/us)
                result['cases'].append(row);print(json.dumps(row),flush=True)
    print(json.dumps(result,indent=2),flush=True)
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
