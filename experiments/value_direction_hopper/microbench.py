"""Matched native-state kernel comparisons; never infer speed from sparsity."""
import argparse
import json
import math
from pathlib import Path

import torch

from .capture import SOURCE
from .cuda import Kernel
from .fused import attention as naive
from .validate import timed


def run(library,states,output,bridge=None,tma_controls=False):
    if output.exists():raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32=False
    kernel=Kernel(library,torch_library=bridge);rows=[];failures=[]
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    bp=json.loads((SOURCE/'policies/ruler4k/blasst_s70.json').read_text())['policy']
    for item in json.loads((states/'index.json').read_text())['states']:
        s=torch.load(item['path'],map_location='cuda',weights_only=True)
        q,k,v,z,r,valid=(s[key] for key in ('q','k','v','z','reference','valid'))
        args=q,k,v,z,r;packed=kernel.pack_mask(valid.contiguous());scale=s['scale']
        threshold=policy[s['kind']]['log_threshold'];length=int(valid.any((1,2)).sum(-1).item())
        blasst=bp[s['kind']]['log_scale']-math.log(length)
        functions=[('native_sdpa',None,lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=s['mask'],scale=scale,enable_gqa=True)),
            ('matched_mask_sdpa',None,lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=valid,scale=scale,enable_gqa=True))]
        try:
            from flash_attn.cute.interface import flash_attn_func
            cq,ck,cv=(x.transpose(1,2) for x in (q,k,v))
            functions.append(('matched_window_flash_hopper',None,lambda:flash_attn_func(cq,ck,cv,softmax_scale=scale,causal=False,
                window_size=(1023,None) if s['kind']=='local' else (None,None))))
        except ImportError as exc:failures.append(dict(layer=s['layer'],method='flash_import',error=repr(exc)))
        functions += [('our_dense_cp',-math.inf,lambda:kernel(*args,mask=packed,scale=scale,mode='dense',precision='tf32x3_register')),
            ('original_blasst_capped_cp',min(0.,blasst),lambda:kernel(*args,mask=packed,scale=scale,mode='blasst',precision='tf32x3_register',log_threshold=min(0.,blasst))),
            ('calibrated_blasst_cp',blasst,lambda:kernel(*args,mask=packed,scale=scale,mode='blasst',precision='tf32x3_register',log_threshold=blasst)),
            ('naive_value_triton',threshold,lambda:naive(*args,mask=valid,scale=scale,log_threshold=threshold,precision='tf32x3')),
            ('value_serial_cp',threshold,lambda:kernel(*args,mask=packed,scale=scale,log_threshold=threshold,precision='tf32x3_register',overlap=False)),
            ('value_overlap_cp',threshold,lambda:kernel(*args,mask=packed,scale=scale,log_threshold=threshold,precision='tf32x3_register')),
            ('value_overlap_tma',threshold,lambda:kernel(*args,mask=packed,scale=scale,log_threshold=threshold,precision='tf32x3_register',tma=True))]
        if tma_controls:
            functions += [('our_dense_tma',-math.inf,lambda:kernel(*args,mask=packed,scale=scale,mode='dense',precision='tf32x3_register',tma=True)),
                ('original_blasst_capped_tma',min(0.,blasst),lambda:kernel(*args,mask=packed,scale=scale,mode='blasst',precision='tf32x3_register',log_threshold=min(0.,blasst),tma=True)),
                ('calibrated_blasst_tma',blasst,lambda:kernel(*args,mask=packed,scale=scale,mode='blasst',precision='tf32x3_register',log_threshold=blasst,tma=True))]
        for name,t,fn in functions:
            try:
                actual=fn();torch.cuda.synchronize()
                record=dict(layer=s['layer'],kind=s['kind'],queries=q.shape[-2],keys=k.shape[-2],width=q.shape[-1],method=name,threshold=str(t),us=timed(fn))
                if name.endswith('sdpa'):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        fn();torch.cuda.synchronize()
                    record['dispatch']=[x.key for x in prof.key_averages() if 'attention' in x.key.lower() or 'fmha' in x.key.lower()]
                if hasattr(actual,'eligible'):
                    record.update(eligible=int(actual.eligible.sum()),skipped=int(actual.skipped.sum()))
                    record['physical_sparsity']=record['skipped']/record['eligible']
                rows.append(record);print(json.dumps(record),flush=True)
                with output.with_suffix('.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            except Exception as exc:
                failure=dict(layer=s['layer'],method=name,error=repr(exc));failures.append(failure);print(json.dumps(failure),flush=True)
                if 'illegal memory' in str(exc):raise
    result=dict(library=str(library),bridge=None if bridge is None else str(bridge),cases=rows,failures=failures,
        timing='CUDA event median of 5 graph batches of30 calls after0.2s load warmup; cached sketches; no profiler instrumentation',
        controls='Our BLASST-compatible physical-tile schedule, not a claimed measurement of the unmodified upstream artifact',
        scope='first-step native development QKV. Include projection/cache costs only in separate preparation/profile and end-to-end runs.',
        dense_caveat='native local mask differs from frozen sparse reference; matched-window FA and matched-mask SDPA are additionally reported; installed Hopper FlashAttention rejects D512')
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bridge',type=Path)
    p.add_argument('--tma-controls',action='store_true')
    a=p.parse_args();run(a.library,a.states,a.output,a.bridge,a.tma_controls)
