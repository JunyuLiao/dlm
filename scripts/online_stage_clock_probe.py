"""Instrumented CTA stage-cycle diagnostics, never clean kernel latency."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--selected',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'source').mkdir()
 paths=[Path(__file__),Path('experiments/numerical_qk_reuse/v2_online_warpspec.py'),a.library.parent/'v2_online_stage_clocks_sm90.cu',a.library.parent/'build_receipt.json']
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 for x in paths:shutil.copyfile(x,a.out/'source'/x.name)
 from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel,pack_z
 from experiments.numerical_qk_reuse.v31_value_projection import refresh
 from experiments.diffusion_gemma_jl_output_aware.projections import Projections
 diagnostic=PackedKernel(a.library);selected=PackedKernel(a.selected);rows=[];stages=['qk_softmax_sketch_staging','projected_pv','risk_vote','conditional_pv_and_value_wait','next_key_wait']
 for nk in [1024,4096,8192,16384]:
  g=torch.Generator(device='cuda').manual_seed(nk);q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,nk,512),device=q.device,dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
  matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device);z=torch.empty((1,2,nk,32),device=q.device);norm=torch.empty((1,2,nk),device=q.device);nu=refresh(v,matrix,z,norm,torch.ones_like(norm,dtype=torch.bool),0);packed=pack_z(z);pt=(nk-256)//64
  for target in [.4,.5,.6]:
   lo,hi=0.,2.;trials=[]
   for _ in range(15):
    tau=(lo+hi)/2;out,keep,*_=selected(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=2,prepacked=packed);count=int(keep.sum());trials.append((abs(count/keep.numel()-(1-target)),tau))
    if count>keep.numel()*(1-target):lo=tau
    else:hi=tau
   tau=min(trials)[1];ref,keep,*_=selected(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=2,prepacked=packed)
   profiles=[]
   for _ in range(5):
    out,chosen,_,_,clocks=diagnostic(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=2,prepacked=packed,phase_profile=True)
    if not torch.equal(chosen,keep) or not torch.equal(out,ref) or not torch.isfinite(clocks).all():raise RuntimeError('instrumented operator changed')
    profiles.append(clocks.detach().clone())
   mean=torch.stack(profiles).mean(0).double();totals=mean.sum((0,1,2,3));fractions=totals/totals.sum();perstage={name:dict(mean_cycles_per_cta_tile=float(totals[i]/mean[...,i].numel()),instrumented_cta_cycle_fraction=float(fractions[i])) for i,name in enumerate(stages)}
   rows.append(dict(keys=nk,target_sparsity=target,actual_sparsity=1-float(keep.float().mean()),threshold=tau,stages=perstage,output_and_mask_bitwise_equal=True));print(json.dumps(rows[-1]),flush=True)
 (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_instrumented_diagnostic',rows=rows,source_sha256=hashes,library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),scope='clock64 and global stores perturb instruction scheduling/register pressure; fractions describe instrumented CTA stage intervals, not hardware stall reasons, clean latency fractions, total FLOPs or request time.'),indent=2)+'\n')
if __name__=='__main__':main()
