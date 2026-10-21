"""Single synthetic warmed online core for Nsight Compute (not clean timing)."""
import argparse,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--schedule',type=int,default=2);p.add_argument('--keys',type=int,default=8192);a=p.parse_args()
 from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel,pack_z
 from experiments.numerical_qk_reuse.v31_value_projection import refresh
 from experiments.diffusion_gemma_jl_output_aware.projections import Projections
 nk=a.keys;g=torch.Generator(device='cuda').manual_seed(nk);q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,nk,512),device=q.device,dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
 matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device);z=torch.empty((1,2,nk,32),device=q.device);norm=torch.empty((1,2,nk),device=q.device);nu=refresh(v,matrix,z,norm,torch.ones_like(norm,dtype=torch.bool),0);packed=pack_z(z);op=PackedKernel(a.library);pt=(nk-256)//64
 lo,hi=0.,2.;trials=[]
 for _ in range(15):
  tau=(lo+hi)/2;keep=op(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed)[1];count=int(keep.sum());trials.append((abs(count-keep.numel()/2),tau))
  if count>keep.numel()/2:lo=tau
  else:hi=tau
 tau=min(trials)[1]
 for _ in range(3):op(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed)
 torch.cuda.synchronize();torch.cuda.profiler.start()
 out,keep,*_=op(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed)
 torch.cuda.synchronize();torch.cuda.profiler.stop()
 print(json.dumps(dict(keys=nk,threshold=tau,eligible=keep.numel(),kept=int(keep.sum()),finite=bool(torch.isfinite(out).all()),schedule=a.schedule)),flush=True)
if __name__=='__main__':main()
