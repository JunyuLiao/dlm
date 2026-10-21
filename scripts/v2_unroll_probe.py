from pathlib import Path
import json
import torch
from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
from scripts.v2_sparse_refresh_bench import graph_functions,timed
root=Path('results/v2_online_warpspec_20261021/unroll/attempt002');root.mkdir(parents=True,exist_ok=False)
libs={'u32':'91f2e0734c00df02','u1':'280c98656cd817e9','u4':'3e295452f3cffd53','u8':'dbe77df609aeb411'}
ks={name:PackedKernel(Path('results/v2_online_warpspec_20261021/build')/key/'v2_online_sm90.so') for name,key in libs.items()};rows=[]
for nk in [1024,4096,8192]:
 g=torch.Generator(device='cuda').manual_seed(nk);q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,nk,512),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g);z=torch.randn((1,2,nk,32),device='cuda',generator=g);nu=torch.full((1,2),2.,device='cuda');pt=(nk-256)//64;tau={1024:.3,4096:.05,8192:.025}[nk]
 expected=discover(q,k,z,nu,512**-.5,pt,tau,ratio=True)[0]
 f={}
 for name,op in ks.items():
  out,keep,*_=op(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=2)
  assert torch.equal(keep.reshape_as(expected),expected)
  f[name]=lambda op=op:op(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,schedule=2)
 calls,owners=graph_functions(f);t=timed(calls,rounds=7,repeats=30);rows.append(dict(keys=nk,timing=t));print(json.dumps(rows[-1]),flush=True)
(root/'receipt.json').write_text(json.dumps(dict(status='complete_diagnostic',rows=rows),indent=2)+'\n')
