"""Matched synthetic screen of the V2 producer/consumer pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--previous-library',type=Path,default=Path('results/refresh_kernel_optimization_20261021/build/c97fc7b83f7ddb44/v2_online_sm90.so'))
    p.add_argument('--keys',type=int,nargs='+',default=[1024,4096,8192]);p.add_argument('--schedules',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--previous-schedule',type=int,default=3)
    p.add_argument('--queries',type=int,default=256);p.add_argument('--rounds',type=int,default=7);p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('new attempt required')
    a.out.mkdir(parents=True);started=time.monotonic()
    # Preserve the exact entry point and library build before launching work.
    build_meta=json.loads((a.library.parent/'build_receipt.json').read_text())
    frozen=a.out/'source';frozen.mkdir()
    freeze_files=[Path(__file__),Path('experiments/numerical_qk_reuse/v2_online_pipeline.py'),Path('experiments/numerical_qk_reuse/v2_online_warpspec.py')]
    freeze_files += [Path(x) for x in build_meta['sources'] if x.endswith(('.cu','.h'))]
    frozen_hashes={}
    for src in freeze_files:
        dest=frozen/src.name;shutil.copyfile(src,dest)
        frozen_hashes[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
    for src in a.library.parent.glob('*.cu'):shutil.copyfile(src,frozen/('compiled_'+src.name))
    shutil.copyfile(a.library.parent/'build_receipt.json',frozen/'build_receipt.json')
    (frozen/'receipt.json').write_text(json.dumps(frozen_hashes,indent=2)+'\n')
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel
    from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover,refresh as two_pass
    from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from scripts.v2_sparse_refresh_bench import graph_functions,timed
    if build_meta['options'].get('wavefront'):
        from experiments.numerical_qk_reuse.v2_online_pipeline import WavefrontKernel
        kernel=WavefrontKernel(a.library)
    elif build_meta['options'].get('packed'):
        from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
        kernel=PackedKernel(a.library)
    else:kernel=Kernel(a.library)
    previous=Kernel(a.previous_library)
    captures=[0];original=torch.cuda.CUDAGraph.capture_begin
    def capture(self,*args,**kwargs):captures[0]+=1;return original(self,*args,**kwargs)
    torch.cuda.CUDAGraph.capture_begin=capture
    results=[];checks=[]
    for nk in a.keys:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        if any(int(x.split(',')[0])!=os.getpid() for x in apps.splitlines() if x.strip()):raise RuntimeError('foreign GPU process; stop timing')
        checks.append(dict(keys=nk,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),apps=apps,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu,clocks.current.sm','--format=csv,noheader'],text=True)))
        g=torch.Generator(device='cuda').manual_seed(nk);n=a.queries
        q=torch.randn((1,16,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
        k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g);v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
        scale=512**-.5;pt=max(0,(nk-n)//64)
        matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device)
        z=torch.empty((1,2,nk,32),device='cuda');norm=torch.empty((1,2,nk),device='cuda');valid=torch.ones_like(norm,dtype=torch.bool)
        nu=project(v,matrix,z,norm,valid,0)
        trials=[];lo,hi=0.,2.
        for _ in range(13):
            tau=(lo+hi)/2;mask=discover(q,k,z,nu,scale,pt,tau,ratio=True)[0];count=int(mask.sum())
            trials.append(dict(threshold=tau,kept=count,eligible=mask.numel()))
            if count>mask.numel()*.5:lo=tau
            else:hi=tau
        winner=min(trials,key=lambda r:abs(r['kept']-r['eligible']*.5));tau=winner['threshold']
        mask=discover(q,k,z,nu,scale,pt,tau,ratio=True)[0].reshape(1,16,(n+127)//128,(nk+63)//64)
        lists=v27_fa4.block_sparse_tensors(mask)
        false=torch.tensor([False],device=q.device)
        def refresh(op,schedule,fast_lists=False):
            fresh=project(v,matrix,z,norm,valid,0)
            out,chosen,*_=op(q,k,v,z,fresh,scale,tau,prefix_tiles=pt,schedule=schedule)
            return out,chosen,(pack_lists(chosen) if fast_lists else v27_fa4.block_sparse_tensors(chosen))
        functions=dict(dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),
            allkept_fa4=lambda:v27_fa4.dense(q,k,v,scale),
            dense_dynamic=lambda:v27_fa4.load()(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True,
                dynamic_causal=false,softmax_scale=scale,pack_gqa=False,num_splits=0,intra_wg_overlap=False)[0],
            projection_rms=lambda:project(v,matrix,z,norm,valid,0),
            same_mask_fa4=lambda:v27_fa4.sparse_lists(q,k,v,lists,scale),
            two_pass_parallel=lambda:two_pass(q,k,v,matrix,scale,pt,tau,backend='parallel',fast_scan=True),
            previous_core=lambda:previous(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=a.previous_schedule),
            previous_refresh=lambda:refresh(previous,a.previous_schedule),
            previous_refresh_fast_lists=lambda:refresh(previous,a.previous_schedule,True),
            fast_lists=lambda:pack_lists(mask))
        if kernel.options.get('tma_all'):
            if kernel.options.get('packed'):
                from experiments.numerical_qk_reuse.v2_online_warpspec import PackedRefresh
                complete=PackedRefresh(a.library,schedule=a.schedules[0])
            else:
                from experiments.numerical_qk_reuse.v2_online_pipeline import OnlineRefresh
                complete=OnlineRefresh(a.library)
            functions['pipeline_complete_api']=lambda:complete(q,k,v,matrix,scale,pt,tau)
        for s in a.schedules:
            functions[f'pipeline_core_s{s}']=lambda s=s:kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=s)
            functions[f'pipeline_refresh_s{s}']=lambda s=s:refresh(kernel,s)
            functions[f'pipeline_refresh_fast_lists_s{s}']=lambda s=s:refresh(kernel,s,True)
            functions[f'pipeline_allkept_core_s{s}']=lambda s=s:kernel(q,k,v,z,nu,scale,0.,prefix_tiles=pt,schedule=s)
        for fn in functions.values():
            for _ in range(2):fn()
        torch.cuda.synchronize();calls,owners=graph_functions(functions);before=captures[0]
        timing=timed(calls,rounds=a.rounds,repeats=a.repeats)
        if captures[0]!=before:raise RuntimeError('capture during timing')
        after_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        if any(int(x.split(',')[0])!=os.getpid() for x in after_apps.splitlines() if x.strip()):raise RuntimeError('foreign GPU process after timing; discard attempt')
        checks.append(dict(keys=nk,position='after_timing',apps=after_apps))
        qualifications=[]
        for s in a.schedules:
            out,chosen,*_=kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=s)
            expected=v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(chosen),scale)
            error=out.float()-expected.float();rel=float(error.norm()/expected.float().norm())
            disagreements=int((chosen!=mask).sum())
            qualifications.append(dict(schedule=s,mask_disagreements=disagreements,relative_l2=rel,max_abs=float(error.abs().max())))
            if disagreements or rel>.01:raise RuntimeError('benchmark operator qualification failed')
        row=dict(keys=nk,queries=n,prefix_tiles=pt,threshold=tau,global_eligible_tiles=mask.numel(),global_kept_tiles=int(mask.sum()),
            sparsity=1-float(mask.float().mean()),calibration=trials,timing=timing,qualification=qualifications)
        results.append(row);print(json.dumps(dict(keys=nk,sparsity=row['sparsity'],us={k:round(v['median_ms']*1000,2) for k,v in timing.items()})),flush=True)
        (a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=results),indent=2)+'\n')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),a.library,a.previous_library,
        *freeze_files]}
    (a.out/'receipt.json').write_text(json.dumps(dict(status='complete',synthetic=True,rows=results,source_sha256=hashes,
        captures_during_timing=0,captures_outside_timing=captures[0],resource_checks=checks,reservation_seconds=time.monotonic()-started,
        torch=torch.__version__,cuda=torch.version.cuda,library_options=kernel.options),indent=2)+'\n')


if __name__=='__main__':main()
