"""Synthetic shape/sparsity sweep, explicitly separate from benchmark accuracy.

Thresholds here are test inputs, not a new calibration policy. Every timing is
paired with observed tile counts; no synthetic target is reported as achieved.
"""
import argparse
import json
from pathlib import Path

import torch

from .cuda import Kernel
from .validate import inputs,timed


def run(library,bridge,output):
    if output.exists():raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    kernel=Kernel(library,torch_library=bridge);rows=[];failures=[]
    for width,hk in ((256,8),(512,2)):
        for length in (1024,4096,8192,16384):
            q,k,v,z,r,mask=inputs(256,length,width,16,hk,seed=42)
            packed=kernel.pack_mask(mask);dense=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=width**-.5,enable_gqa=True)
            dense_us=timed(dense)
            flash_us=None
            try:
                from flash_attn.cute.interface import flash_attn_func
                flash=lambda:flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=False)
                flash_us=timed(flash)
            except Exception as exc:failures.append(dict(width=width,length=length,method='flash',error=repr(exc)))
            for log_tau in (-float('inf'),-3.,-2.,-1.,0.):
                call=lambda:kernel(q,k,v,z,r,mask=packed,log_threshold=log_tau,precision='tf32x3_register',tma=True)
                actual=call();us=timed(call)
                row=dict(width=width,queries=256,keys=length,seed=42,log_tau=str(log_tau),
                    eligible=int(actual.eligible.sum()),skipped=int(actual.skipped.sum()),
                    kernel_us=us,sdpa_us=dense_us,flash_us=flash_us,sdpa_ratio=dense_us/us,
                    flash_ratio=None if flash_us is None else flash_us/us)
                row['physical_sparsity']=row['skipped']/row['eligible'];rows.append(row)
                with output.with_suffix('.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(cases=rows,failures=failures,library=str(library),
        scope='Synthetic all-valid QKV, rank32, cached projections; no benchmark score/calibration or end-to-end claim. D256 long contexts are shape stress tests, not native local-window length.',
        timing='Same CUDA graph/event median/warmup convention as native-state microbenchmark'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.library,a.bridge,a.output)
