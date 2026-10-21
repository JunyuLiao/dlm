import json
from pathlib import Path
import torch
from experiments.numerical_qk_reuse.v2_online_factored import FactoredKernel
from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
from experiments.numerical_qk_reuse import v27_fa4
from scripts.v2_sparse_refresh_bench import graph_functions,timed
root=Path('results/v2_online_warpspec_20261021/factored/attempt001');root.mkdir(parents=True,exist_ok=False)
rows=[]
for nk in [1024,4096,8192]:
 g=torch.Generator(device='cuda').manual_seed(nk);q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g);v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=g);z=torch.randn((1,2,nk,32),device='cuda',generator=g);nu=torch.full((1,2),2.,device='cuda');pt=(nk-256)//64;tau={1024:.25,4096:.05,8192:.025}[nk]
 ref=discover(q,k,z,nu,512**-.5,pt,tau,ratio=True,trace=True)
 fn=FactoredKernel();out,keep,state,lm,_=fn(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,trace=True)
 expected=v27_fa4.sparse(q,k,v,~keep,torch.ones_like(keep),512**-.5);rel=float((out.float()-expected.float()).norm()/expected.float().norm());errors=int((ref[0].reshape_as(keep)!=keep).sum());se=float((state.reshape_as(ref[1])-ref[1]).abs().max());print(nk,errors,rel,se,flush=True)
 if errors or rel>.01 or se>1.e-4:raise RuntimeError('factored failed')
 f={'dense':lambda:v27_fa4.dense_plain(q,k,v,512**-.5),'factored':lambda:fn(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt)}
 for op in f.values():op()
 torch.cuda.synchronize();calls,owners=graph_functions(f);timing=timed(calls,rounds=5,repeats=30);rows.append(dict(keys=nk,keep=float(keep.float().mean()),mask_errors=errors,rel=rel,state=se,timing=timing));print(json.dumps(rows[-1]),flush=True)
(root/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic_probe',rows=rows),indent=2)+'\n')
