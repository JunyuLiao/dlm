"""Matched-scope refresh kernel optimization screen, synthetic states only."""
import argparse
import json
import time
import hashlib
import subprocess
import os
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--keys',type=int,nargs='+',default=[512,1024,4096,8192])
    p.add_argument('--v31-splits',type=int,nargs='+',default=[2,4,8])
    p.add_argument('--tma',action='store_true')
    p.add_argument('--schedules',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('new attempt required')
    a.out.mkdir(parents=True);start=time.monotonic()
    from experiments.numerical_qk_reuse import v27_fa4,v31_fa4_observe
    from experiments.numerical_qk_reuse.refresh_v31_fast import Observer,select
    from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover,refresh as two_pass_refresh
    from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from scripts.v31_humaneval_observe_no_overlap import wrap_fwd
    from scripts.v2_sparse_refresh_bench import timed,graph_functions
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    observer=Observer(a.out/'private/fa4');kernel=Kernel(a.library)
    v31_fa4_observe._flash_attn_fwd=wrap_fwd(v31_fa4_observe._flash_attn_fwd,dict(observing_calls=0))
    adapter=VllmMethodAdapter(['full_attention']*30,arm='mage',mage_select='fa4',mage_granularity='qblock_max',mage_k=128)
    adapter.calls.update(mage_kept_prefix_tiles=0,mage_prefix_tiles=0)
    captures=[0];capture=torch.cuda.CUDAGraph.capture_begin
    def capture_begin(self,*args,**kwargs):captures[0]+=1;return capture(self,*args,**kwargs)
    torch.cuda.CUDAGraph.capture_begin=capture_begin
    rows=[];resource_checks=[]
    for nk in a.keys:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        if any(int(line.split(',')[0])!=os.getpid() for line in apps.splitlines() if line.strip()):raise RuntimeError('foreign GPU process; timing prohibited')
        resource_checks.append(dict(keys=nk,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),apps=apps,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used,clocks.current.sm','--format=csv,noheader'],text=True)))
        g=torch.Generator(device='cuda').manual_seed(nk)
        q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=g)
        k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g)
        v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
        scale=512**-.5;prefix=nk-256;pt=prefix//64
        z=torch.empty((16,2,pt,128),device='cuda')
        matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device)
        sketch=torch.empty((1,2,nk,32),device='cuda');norm=torch.empty((1,2,nk),device='cuda')
        valid=torch.ones_like(norm,dtype=torch.bool)
        nu=project(v,matrix,sketch,norm,valid,0)
        low,high=0.,2.;best=None
        for _ in range(13):
            threshold=(low+high)/2
            chosen=discover(q,k,sketch,nu,scale,pt,threshold,ratio=True)[0]
            count=int(chosen.sum());error=abs(count-chosen.numel()*.5)
            if best is None or error<best[0]:best=(error,threshold,count,chosen.numel())
            if count>chosen.numel()*.5:low=threshold
            else:high=threshold
        tau=best[1]
        def inherited():
            output,keep=adapter._mage_select_fa4(q,k,v,scale,prefix,256)
            return output,v27_fa4.block_sparse_tensors(keep)
        def old_observe_fast_select():
            out=v31_fa4_observe.observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale,z)
            return out,select(z,256,nk,128)
        def online(schedule,projection,tma=False):
            reference=project(v,matrix,sketch,norm,valid,0) if projection else nu
            out,keep,*_=kernel(q,k,v,sketch,reference,scale,tau,prefix_tiles=pt,schedule=schedule,tma=tma)
            lists=v27_fa4.block_sparse_tensors(keep)
            return out,keep,lists
        functions=dict(dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),
            v31_inherited=inherited,v31_fused_select=old_observe_fast_select,
            v31_softmax_observer=lambda:observer.refresh(q,k,v,scale,prefix,128,overlap=False),
            dense_no_overlap=lambda:v27_fa4.load()(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                causal=False,softmax_scale=scale,pack_gqa=False,num_splits=1,intra_wg_overlap=False)[0])
        for splits in a.v31_splits:
            functions['v31_softmax_observer_split'+str(splits)]=lambda s=splits:observer.refresh(q,k,v,scale,prefix,128,overlap=False,num_splits=s)
        functions['v31_softmax_dynamic']=lambda:observer.refresh(q,k,v,scale,prefix,128,overlap=False,num_splits=0,dynamic_causal=True)
        dynamic_false=torch.tensor([False],device=q.device)
        functions['dense_dynamic_no_overlap']=lambda:v27_fa4.load()(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            causal=True,dynamic_causal=dynamic_false,softmax_scale=scale,pack_gqa=False,num_splits=0,intra_wg_overlap=False)[0]
        functions['v2_two_pass_ratio_w4']=lambda:two_pass_refresh(q,k,v,matrix,scale,pt,tau,backend='fused',ratio=True,warps=4)
        functions['v2_two_pass_parallel_fast']=lambda:two_pass_refresh(q,k,v,matrix,scale,pt,tau,backend='parallel',fast_scan=True)
        functions['v2_two_pass_split_d']=lambda:two_pass_refresh(q,k,v,matrix,scale,pt,tau,backend='fused',ratio=True,split_d=True,warps=4)
        for schedule in a.schedules:
            functions['v2_online_core_s'+str(schedule)]=lambda s=schedule:kernel(q,k,v,sketch,nu,scale,tau,prefix_tiles=pt,schedule=s)
            functions['v2_online_refresh_s'+str(schedule)]=lambda s=schedule:online(s,True)
        if a.tma:
            for schedule in (1,2):
                functions['v2_online_tma_core_s'+str(schedule)]=lambda s=schedule:kernel(q,k,v,sketch,nu,scale,tau,prefix_tiles=pt,schedule=s,tma=True)
                functions['v2_online_tma_refresh_s'+str(schedule)]=lambda s=schedule:online(s,True,tma=True)
        for f in functions.values():
            for _ in range(2):f()
        torch.cuda.synchronize()
        calls,owners=graph_functions(functions)
        captured=captures[0]
        timing=timed(calls,rounds=7,repeats=30)
        if captures[0]!=captured:raise RuntimeError('capture during timing')
        output,keep,lists=functions['v2_online_refresh_s'+str(a.schedules[0])]()
        operator_ref=v27_fa4.sparse_lists(q,k,v,lists,scale)
        error=output.float()-operator_ref.float()
        row=dict(keys=nk,queries=256,threshold=tau,eligible=best[3],kept=int(keep.sum()),timing=timing,
            v2_mask_disagreement=int((keep.reshape_as(chosen)!=discover(q,k,sketch,nu,scale,pt,tau,ratio=True)[0]).sum()),
            v2_output_fa4_relative_l2=float(error.norm()/operator_ref.float().norm()),
            v2_output_fa4_max_abs=float(error.abs().max()))
        if row['v2_mask_disagreement'] or row['v2_output_fa4_relative_l2']>.01:raise RuntimeError('benchmark geometry correctness failed')
        rows.append(row)
        print(json.dumps(dict(keys=nk,sparsity=1-row['kept']/row['eligible'],us={k:round(v['median_ms']*1000,2) for k,v in timing.items()})),flush=True)
        (a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
    (a.out/'receipt.json').write_text(json.dumps(dict(status='complete',synthetic=True,rows=rows,
        captures_during_timing=0,captures_outside_timing=captures[0],reservation_seconds=time.monotonic()-start,
        fa4_observer_source_sha256=observer.source_sha256,resource_checks=resource_checks,
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path('experiments/numerical_qk_reuse/v2_online_sm90.py'),Path('experiments/numerical_qk_reuse/refresh_v31_fast.py')]},
        numerical_policy='FP32 QK, normalized FP32 routing; BF16 P output; finite QKV required'),indent=2)+'\n')


if __name__=='__main__':main()
