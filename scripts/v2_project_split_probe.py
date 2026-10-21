import json
from pathlib import Path
import torch
from experiments.numerical_qk_reuse.v2_online_warpspec import project_split
from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from scripts.v2_sparse_refresh_bench import graph_functions,timed
root=Path('results/v2_online_warpspec_20261021/projection/attempt001');root.mkdir(parents=True,exist_ok=False);rows=[]
for nk in [449,1024,4096,8192]:
 g=torch.Generator(device='cuda').manual_seed(nk);v=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g);r=Projections().get(5,2,512,'gaussian',32,1729,v.device);z=torch.empty((1,2,nk,32),device='cuda');a=torch.empty_like(z);norm=torch.empty((1,2,nk),device='cuda');valid=torch.ones_like(norm,dtype=torch.bool)
 nu=project(v,r,z,norm,valid);new=project_split(v,r,a,norm,valid);err=float((z-a).abs().max());rel=float((z-a).norm()/z.norm());ne=float((nu-new).abs().max());print(nk,err,rel,ne,flush=True)
 f={'old':lambda:project(v,r,z,norm,valid),'split':lambda:project_split(v,r,a,norm,valid)}
 for fn in f.values():fn()
 torch.cuda.synchronize();calls,owners=graph_functions(f);t=timed(calls,rounds=5,repeats=30);rows.append(dict(keys=nk,z_abs=err,z_rel=rel,nu_abs=ne,timing=t));print(json.dumps(rows[-1]),flush=True)
(root/'receipt.json').write_text(json.dumps(dict(status='complete_diagnostic',rows=rows),indent=2)+'\n')
