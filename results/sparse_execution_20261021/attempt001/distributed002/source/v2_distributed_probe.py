"""Bounded distributed online variant qualification before any timing."""
import argparse,hashlib,json,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--ctas',type=int,default=4);p.add_argument('--warps',type=int,default=4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 from experiments.numerical_qk_reuse.v2_online_distributed import online
 from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
 from experiments.numerical_qk_reuse.v31_value_projection import refresh
 from experiments.diffusion_gemma_jl_output_aware.projections import Projections
 from experiments.numerical_qk_reuse import v27_fa4
 from scripts.v2_sparse_refresh_bench import graph_functions,timed
 paths=[Path(__file__),Path('experiments/numerical_qk_reuse/v2_online_distributed.py')];(a.out/'source').mkdir()
 for x in paths:(a.out/'source'/x.name).write_bytes(x.read_bytes())
 rows=[]
 for nk,n in [(641,137),(1024,256),(4096,256),(8192,256)]:
  g=torch.Generator(device='cuda').manual_seed(nk);h=16;hk=2;q=torch.randn((1,h,n,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,hk,nk,512),device=q.device,dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
  matrix=Projections().get(5,hk,512,'gaussian',32,1729,q.device);z=torch.empty((1,hk,nk,32),device=q.device);norm=torch.empty((1,hk,nk),device=q.device);nu=refresh(v,matrix,z,norm,torch.ones_like(norm,dtype=torch.bool),0);pt=(nk-n)//64
  lo,hi=0.,2.;trials=[]
  for _ in range(12):
   tau=(lo+hi)/2;keep=discover(q,k,z,nu,512**-.5,pt,tau,ratio=True)[0];count=int(keep.sum());trials.append((abs(count/keep.numel()-.5),tau))
   if count>keep.numel()/2:lo=tau
   else:hi=tau
  tau=min(trials)[1];chosen=discover(q,k,z,nu,512**-.5,pt,tau,ratio=True)[0]
  out,keep=online(q,k,v,z,nu,512**-.5,tau,pt,ctas=a.ctas,warps=a.warps);expected=v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(keep),512**-.5)
  diff=int((keep.reshape_as(chosen)!=chosen).sum());error=float((out.float()-expected.float()).norm()/expected.float().norm());print(dict(keys=nk,mask_disagreements=diff,relative_l2=error),flush=True)
  if diff or error>.01 or not torch.isfinite(out).all():raise RuntimeError('distributed online qualification')
  fn=lambda:online(q,k,v,z,nu,512**-.5,tau,pt,ctas=a.ctas,warps=a.warps)
  for _ in range(2):fn()
  fns,owners=graph_functions(dict(online_core=fn,dense=lambda:v27_fa4.dense_plain(q,k,v,512**-.5)));t=timed(fns,rounds=5,repeats=20)
  rows.append(dict(keys=nk,queries=n,threshold=tau,actual_sparsity=1-float(keep.float().mean()),timing=t,mask_disagreements=diff,relative_l2=error));print(rows[-1],flush=True)
 (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic_screen',rows=rows,ctas=a.ctas,warps=a.warps,source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}),indent=2)+'\n')
if __name__=='__main__':main()
