"""Bounded explicit library qualification, independent of build-time paths."""
import argparse,json,os,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
 from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
 from experiments.numerical_qk_reuse import v27_fa4
 from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
 kfn=PackedKernel(a.library);rows=[]
 for nk,n in [(641,137),(4096,256)]:
  for schedule in [0,3]:
   g=torch.Generator(device='cuda').manual_seed(nk);q=torch.randn((1,4,n,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,nk,512),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
   z=torch.randn((1,2,nk,32),device=q.device,generator=g);nu=torch.full((1,2),2.,device=q.device);held=torch.zeros((1,4,(n+127)//128,(nk+63)//64),device=q.device,dtype=torch.bool);pt=(nk-n)//64
   def run():return kfn(q,k,v,z,nu,512**-.5,.08,prefix_tiles=pt,held=held,sticky=1.386,schedule=schedule)
   for _ in range(2):run()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out,keep,*_=run()
   worst=0.;changes=0;previous=None
   for iteration in range(20):
    q.copy_(torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=g));k.copy_(torch.randn(k.shape,device=q.device,dtype=k.dtype,generator=g));v.copy_(torch.randn(v.shape,device=q.device,dtype=v.dtype,generator=g));z.copy_(torch.randn(z.shape,device=q.device,generator=g))
    if iteration%2==0:z.mul_(.001)
    held.copy_(torch.rand(held.shape,device=q.device,generator=g)>.5)
    expected=discover(q,k,z,nu,512**-.5,pt,.08,held=held,sticky=1.386,ratio=True)[0].reshape_as(keep);graph.replay()
    if not torch.equal(keep,expected):raise RuntimeError('online mask disagreement')
    ref=v27_fa4.sparse_lists(q,k,v,pack_lists(keep),512**-.5);rel=float((out.float()-ref.float()).norm()/ref.float().norm());worst=max(worst,rel)
    if not torch.isfinite(out).all() or rel>.01:raise RuntimeError('online output mismatch')
    if previous is not None:changes+=int((previous!=keep).sum())
    previous=keep.clone()
   if not changes:raise RuntimeError('no decision transitions')
   rows.append(dict(keys=nk,queries=n,schedule=schedule,iterations=20,mask_disagreements=0,changes=changes,relative_l2_max=worst));print(rows[-1],flush=True)
 (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_operator_qualification',library=str(a.library),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
