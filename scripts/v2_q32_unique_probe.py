"""Operator and complete-refresh screen for the Q32 unique online experiment."""
import argparse,hashlib,json,time,shutil,subprocess,os
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--keys',type=int,nargs='+',default=[1024,4096,8192,16384]);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),Path('experiments/numerical_qk_reuse/csrc/v2_online_q32_unique_sm90.cu'),Path('experiments/numerical_qk_reuse/v2_online_sm90.py')]
 config=dict(scope='synthetic only; not native calibration or completed benchmark',keys=a.keys,shape=dict(q=256,h=16,hkv=2,d=512),actual_sparsity_target=.5,qualification_l2_limit=.01,library=str(a.library),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},warmup=True,capture_during_timing=False)
 (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 src=a.out/'source';src.mkdir()
 for f in files:shutil.copyfile(f,src/f.name)
 from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel
 from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
 from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
 from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
 from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
 from experiments.numerical_qk_reuse import v27_fa4
 from experiments.diffusion_gemma_jl_output_aware.projections import Projections
 from scripts.v2_sparse_refresh_bench import graph_functions,timed
 kernel=Kernel(a.library);best=PackedKernel(Path('results/sparse_execution_20261021/attempt001/build/3f78b2e221e5bbfa/v2_online_sm90.so'))
 captures=[0];capture_begin=torch.cuda.CUDAGraph.capture_begin
 def counted_capture(self,*args,**kwargs):captures[0]+=1;return capture_begin(self,*args,**kwargs)
 torch.cuda.CUDAGraph.capture_begin=counted_capture
 rows=[];start=time.monotonic()
 for nk in a.keys:
  n,h,hk,d=256,16,2,512;g=torch.Generator(device='cuda').manual_seed(nk)
  q=torch.randn((1,h,n,d),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,hk,nk,d),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
  z=torch.empty((1,hk,nk,32),device=q.device);norm=torch.empty((1,hk,nk),device=q.device);valid=torch.ones_like(norm,dtype=torch.bool);matrix=Projections().get(5,hk,d,'gaussian',32,1729,q.device);nu=project(v,matrix,z,norm,valid,0);pt=(nk-n)//64
  lo,hi=0.,2.;trials=[]
  for _ in range(15):
   tau=(lo+hi)/2;keep=kernel(q,k,v,z,nu,d**-.5,tau,prefix_tiles=pt,schedule=3)[1];count=int(keep.sum());trials.append((abs(count/keep.numel()-.5),tau,count))
   if count>keep.numel()*.5:lo=tau
   else:hi=tau
  _,tau,_=min(trials);out,keep,*_=kernel(q,k,v,z,nu,d**-.5,tau,prefix_tiles=pt,schedule=3)
  expected=discover(q,k,z,nu,d**-.5,pt,tau,ratio=True)[0].reshape_as(keep)
  ref=v27_fa4.sparse_lists(q,k,v,pack_lists(keep),d**-.5);relative=float((out.float()-ref.float()).norm()/ref.float().norm())
  if not torch.equal(keep,expected) or not torch.isfinite(out).all() or relative>.01:raise RuntimeError('Q32 operator qualification failed')
  def complete():
   nu=project(v,matrix,z,norm,valid,0);out,keep,*_=kernel(q,k,v,z,nu,d**-.5,tau,prefix_tiles=pt,schedule=3);return out,pack_lists(keep)
  funcs=dict(dense_plain=lambda:v27_fa4.dense_plain(q,k,v,d**-.5),q32_core=lambda:kernel(q,k,v,z,nu,d**-.5,tau,prefix_tiles=pt,schedule=3),q32_allkept=lambda:kernel(q,k,v,z,nu,d**-.5,0.,prefix_tiles=pt,schedule=3),q32_complete=complete,bulk3d_core=lambda:best(q,k,v,z,nu,d**-.5,tau,prefix_tiles=pt,schedule=3))
  for f in funcs.values():
   for _ in range(3):f()
  fns,owners=graph_functions(funcs);torch.cuda.synchronize();before=captures[0];timing=timed(fns,rounds=5,repeats=30)
  if captures[0]!=before:raise RuntimeError('capture during timing')
  row=dict(keys=nk,threshold=tau,global_eligible_tiles=keep.numel(),global_kept_tiles=int(keep.sum()),actual_sparsity=1-float(keep.float().mean()),relative_l2=relative,timing=timing,captures_during_timing=captures[0]-before,capture_counter_verified=True)
  rows.append(row);print(json.dumps(row),flush=True)
  (a.out/'receipt.json').write_text(json.dumps(dict(status='running',config=config,rows=rows),indent=2)+'\n')
 (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic_diagnostic',config=config,rows=rows,reservation_seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
