"""Standalone diagnostic: identical total work, varying per-unit load skew.

Not run concurrently with a native benchmark worker. Split variants are only
eligible as output-preserving changes if bitwise output equivalence holds.
"""
import argparse,json,time
from pathlib import Path
import torch
from experiments.numerical_qk_reuse.resident_sparse import PagedContext,ResidentMask
from scripts.v2_sparse_refresh_bench import timed,graph_functions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();rows=[]
    captures=[0];original=torch.cuda.CUDAGraph.capture_begin
    def capture(self,*args,**kwargs):captures[0]+=1;return original(self,*args,**kwargs)
    torch.cuda.CUDAGraph.capture_begin=capture
    for nk in [8192,131072]:
        g=torch.Generator(device='cuda').manual_seed(nk);n=256;h=16;hk=2;d=512;tiles=nk//64;pages=nk//128
        q=torch.randn((1,h,n,d),device='cuda',dtype=torch.bfloat16,generator=g)
        cache=torch.randn((pages,128,hk,1024),device='cuda',dtype=q.dtype,generator=g);key,value=cache[...,:d],cache[...,d:]
        table=torch.randperm(pages,device=q.device,generator=g).int()
        for fraction in [.1,.5]:
            target=round(tiles*fraction)
            for skew in [0.,.5,.9]:
                spread=round(min(target-1,tiles-target)*skew)
                counts=torch.full((32,),target,device=q.device,dtype=torch.int32)
                counts[::2]+=spread;counts[1::2]-=spread
                # Setup only: deterministic compact support, mandatory first tile.
                keep=(torch.arange(tiles,device=q.device)[None,:]<counts[:,None]).reshape(1,h,2,tiles).contiguous()
                functions={};outputs={}
                for splits in [1,2,4,8]:
                    context=PagedContext(key,value,table,nk,splits=splits);mask=ResidentMask.create(keep,splits=splits)
                    def run(context=context,mask=mask):return context(q,mask,d**-.5)
                    functions[str(splits)]=run;outputs[str(splits)]=run().clone()
                for fn in functions.values():
                    for _ in range(3):fn()
                torch.cuda.synchronize();calls,owners=graph_functions(functions);before=captures[0];timing=timed(calls,rounds=5,repeats=10)
                if captures[0]!=before:raise RuntimeError('timed capture')
                base=outputs['2'];parity={k:dict(bitwise=torch.equal(x,base),relative_l2=float((x.float()-base.float()).norm()/base.float().norm())) for k,x in outputs.items()}
                row=dict(keys=nk,kept_fraction=float(counts.sum())/(32*tiles),kept_tiles=int(counts.sum()),eligible_tiles=32*tiles,counts_min=int(counts.min()),counts_max=int(counts.max()),unit_cv=float(counts.float().std()/counts.float().mean()),skew=skew,timing=timing,parity_vs_selected_split2=parity)
                rows.append(row);print(json.dumps(dict(keys=nk,fraction=fraction,skew=skew,us={k:1000*v['median_ms'] for k,v in timing.items()},parity=parity)),flush=True)
    (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic',rows=rows,captures_during_timing=0,captures_outside_timing=captures[0],reservation_seconds=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':main()
