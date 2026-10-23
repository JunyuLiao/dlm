"""Matched exact-output V2 paged/cached refresh and held execution diagnostics."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from experiments.numerical_qk_reuse.v2_paged_refresh import PagedRefresh
from experiments.numerical_qk_reuse.v2_online_warpspec import FusedPackedRefresh,CanvasPackedRefresh
from experiments.numerical_qk_reuse.v29_paged_copy import copy_paged_kv
from experiments.numerical_qk_reuse.resident_sparse import PagedContext,ResidentMask
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from scripts.v2_sparse_refresh_bench import timed,graph_functions

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--paged',type=Path);p.add_argument('--swizzle',type=Path);p.add_argument('--contiguous-candidate',type=Path);p.add_argument('--unique',type=Path,nargs='*',default=[]);p.add_argument('--keys',type=int,nargs='+',default=[8192,131072]);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    oldlib=Path('results/sparse_execution_20261021/attempt002/build/200354cb919fb6b8/v2_online_sm90.so')
    old=FusedPackedRefresh(oldlib,2);cached=CanvasPackedRefresh(oldlib,2)
    candidates={'paged':PagedRefresh(a.paged,2)} if a.paged else {}
    if a.swizzle:candidates['swizzle']=PagedRefresh(a.swizzle,2)
    captures=[0];original=torch.cuda.CUDAGraph.capture_begin
    def capture(self,*args,**kwargs):captures[0]+=1;return original(self,*args,**kwargs)
    torch.cuda.CUDAGraph.capture_begin=capture;rows=[]
    for nk in a.keys:
        n=256;h=16;hk=2;d=512;prefix=nk-n;g=torch.Generator(device='cuda').manual_seed(nk)
        q=torch.randn((1,h,n,d),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,hk,nk,d),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
        pages=nk//128;table=torch.randperm(pages,device='cuda',generator=g).int();cache=torch.empty((pages,128,hk,1024),device='cuda',dtype=q.dtype);key,value=cache[...,:d],cache[...,d:]
        key[table.long()]=k[0].transpose(0,1).reshape_as(key);value[table.long()]=v[0].transpose(0,1).reshape_as(value)
        matrix=Projections().get(5,hk,d,'gaussian',32,1729,q.device);ck=torch.empty_like(k);cv=torch.empty_like(v)
        for target in [.45,.90]:
            lo,hi=0.,2.;trials=[]
            for _ in range(14):
                tau=(lo+hi)/2;_,keep,_=old(q,k,v,matrix,d**-.5,prefix//64,tau);s=1-int(keep.sum())/keep.numel();trials.append((abs(s-target),tau,s))
                if s<target:lo=tau
                else:hi=tau
            _,tau,sparsity=min(trials);expected=old(q,k,v,matrix,d**-.5,prefix//64,tau)
            funcs={'contiguous_refresh':lambda:old(q,k,v,matrix,d**-.5,prefix//64,tau)}
            def old_module():
                copy_paged_kv(key,value,table,ck,cv,prefix,n)
                return old(q,ck,cv,matrix,d**-.5,prefix//64,tau)
            copy_paged_kv(key,value,table,ck,cv,0,nk)
            funcs['contiguous_canvas_copy_refresh']=old_module
            funcs['cached_contiguous_refresh']=lambda:cached(q,k,v,matrix,d**-.5,prefix//64,tau,cache_key=5,canvas=nk,prefix=prefix)
            diagnostic_parity={}
            if a.contiguous_candidate:
                candidate=CanvasPackedRefresh(a.contiguous_candidate,2)
                def run_contiguous():return candidate(q,k,v,matrix,d**-.5,prefix//64,tau,cache_key=5,canvas=nk,prefix=prefix)
                got=run_contiguous()
                if not torch.equal(got[0],expected[0]) or not torch.equal(got[1],expected[1]):raise RuntimeError('CTA reorder changes output')
                funcs['cta_order_cached_refresh']=run_contiguous
            for library in a.unique:
                unique=CanvasPackedRefresh(library,3)
                def run_unique(unique=unique):return unique(q,k,v,matrix,d**-.5,prefix//64,tau,cache_key=5,canvas=nk,prefix=prefix)
                got=run_unique();name='unique_'+library.parent.name
                diagnostic_parity[name]=dict(mask_differences=int((got[1]!=expected[1]).sum()),output_bitwise_equal=torch.equal(got[0],expected[0]),relative_l2=float((got[0].float()-expected[0].float()).norm()/expected[0].float().norm()))
                funcs[name]=run_unique
            for name,op in candidates.items():
                def run(op=op):return op(q,key,value,table,nk,matrix,d**-.5,prefix,tau,cache_key=5,canvas=nk)
                for _ in range(2):actual=run()
                if not torch.equal(actual[0],expected[0]) or not torch.equal(actual[1],expected[1]):raise RuntimeError('candidate changes output or mask: '+name)
                funcs[name+'_cached_refresh']=run
                funcs[name+'_full_refresh']=lambda op=op:op(q,key,value,table,nk,matrix,d**-.5,prefix,tau)
            context=PagedContext(key,value,table,nk,splits=2);mask=ResidentMask.create(expected[1],splits=2)
            funcs['held_identical_mask']=lambda:context(q,mask,d**-.5)
            for fn in funcs.values():
                for _ in range(3):fn()
            torch.cuda.synchronize();calls,owners=graph_functions(funcs);before=captures[0]
            measurements=timed(calls,rounds=5,repeats=10)
            host=timed(funcs,rounds=3,repeats=3)
            if before!=captures[0]:raise RuntimeError('timed capture')
            row=dict(keys=nk,target=target,threshold=tau,actual_sparsity=sparsity,bitwise_output_and_mask=True,timing=measurements,host_timing=host,captures_during_timing=0,diagnostic_parity=diagnostic_parity)
            rows.append(row);(a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
            print(json.dumps(dict(keys=nk,sparsity=sparsity,us={k:round(v['median_ms']*1000,2) for k,v in measurements.items()})),flush=True)
    files=[Path(__file__),Path('experiments/numerical_qk_reuse/v2_paged_refresh.py'),oldlib]+[x for x in [a.paged,a.swizzle,a.contiguous_candidate,*a.unique] if x]
    (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic',rows=rows,captures_during_timing=0,captures_outside_timing=captures[0],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},reservation_seconds=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':main()
