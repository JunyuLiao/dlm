"""Development-only native generation smoke; never selects new thresholds."""
import argparse
import json
import time
from pathlib import Path

import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense,_set_context,_request
from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL,REVISION
from experiments.diffusion_gemma_jl_output_aware.routing import Attention as ReferenceAttention
from experiments.diffusion_gemma_ruler8k_jl import score
from .capture import SOURCE
from .integration import install


def run(library,output,samples=2,budget=16):
    if (output/'summary.json').exists():raise FileExistsError(output/'summary.json')
    output.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads((SOURCE/'development_manifest.json').read_text())[:samples]
    policy=json.loads((SOURCE/'policies/ruler4k/jl_gaussian_r32_s70.json').read_text())['policy']
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    results=[]
    for original in rows:
        row=dict(original,generation_budget=budget)
        # Native warmup is excluded; graph/kernel first-use costs remain labeled.
        adapter.generate(_request(dict(row,generation_budget=1)))
        for method in ('native_dense','historical_dense','historical_sparse','kernel_sparse'):
            dest=output/(original['id'].replace('/','_')+'.'+method+'.json')
            if dest.exists():results.append(json.loads(dest.read_text()));continue
            torch.cuda.synchronize();start=time.perf_counter();router=None
            if method=='native_dense':generated=adapter.generate(_request(row))
            elif method=='kernel_sparse':
                with install(adapter,library,policy) as (binding,router):
                    _set_context(binding,row);generated=adapter.generate(_request(row))
            else:
                binding=_install_dense(adapter)
                selected=policy if method=='historical_sparse' else {kind:dict(unpruned=True) for kind in ('local','global')}
                ref=ReferenceAttention(dict(family='gaussian',rank=32),selected,trusted=True)
                binding.runtime.attention_override=ref
                try:_set_context(binding,row);generated=adapter.generate(_request(row))
                finally:binding.close()
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            record=dict(id=row['id'],prompt_hash=row['prompt_hash'],seed=row['seed'],method=method,
                diagnostic_budget=budget,official_budget=original['generation_budget'],
                completion_tokens=generated.completion_tokens,prediction=generated.text,
                score=score(row,generated.text),metadata=generated.metadata,elapsed_seconds=seconds,
                timing_scope='development smoke, includes Python setup/first-use costs; not final performance claim')
            if router is not None:
                raw=router.records()
                record['routing']=raw
                record['projection_work']=dict(projected_tokens=router.cache.projected_tokens,reused_tokens=router.cache.reused_tokens)
                record['projection_matrices']=router.cache.projections.manifest
            dest.write_text(json.dumps(record,indent=2)+'\n');results.append(record)
            print(json.dumps({key:record[key] for key in ('id','method','score','elapsed_seconds')}),flush=True)
    comparisons=[]
    for row in rows:
        grouped={r['method']:r for r in results if r['id']==row['id']}
        for method in ('native_dense','historical_sparse'):
            a=grouped[method]['completion_tokens'];b=grouped['kernel_sparse']['completion_tokens']
            comparisons.append(dict(id=row['id'],reference=method,exact=a==b,matches=sum(x==y for x,y in zip(a,b)),compared=max(len(a),len(b))))
    summary=dict(results=results,comparisons=comparisons,source=str(SOURCE),library=str(library),thresholds=policy,
                 scope='previously examined development prompts, capped generation for smoke only; no accuracy selection')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=2);p.add_argument('--budget',type=int,default=16)
    a=p.parse_args();run(a.library,a.output,a.samples,a.budget)
