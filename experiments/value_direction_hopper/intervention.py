"""Trajectory-protection schedules using the existing Hopper router."""
import argparse, copy, gzip, json, os, time, traceback, fcntl
from contextlib import nullcontext
from pathlib import Path
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from .experiment import atomic,sha,shard_path
from .integration import install
from .trajectory import observe
from .sparsity_steps import prepare, routing_counts
from experiments.diffusion_gemma_ruler8k_jl import score

ROOT=Path(__file__).resolve().parents[2]/'results'/'value_direction_interventions_v1'
BASE=Path(__file__).resolve().parents[2]/'results'/'value_direction_sparsity_steps_v2'
ROWS=json.loads((BASE/'manifest.json').read_text())
POL=json.loads((BASE/'configuration.json').read_text())['policies']
# First pass focuses on the difficult boundary and uses existing calibrated policies.
CONDITIONS=['gaussian32_dense_first_s60','gaussian32_dense_first_s70','gaussian32_dense_late_s60','gaussian32_dense_late_s70','gaussian32_reactive_s60','gaussian32_reactive_s70']

def policy_for(label,step,record=None):
    target=int(label.rsplit('_s',1)[1]); base=POL[f'gaussian32_s{target}']
    if 'dense_first' in label and step<=1:return None
    if 'dense_late' in label and step>=4:return None
    if 'reactive' in label and record is not None and record.get('trigger',False):return None
    return base

def run_one(adapter,row,label,cfg,projections):
    records=[]; trigger_state={'prev_top':None,'prev_accept':None,'prev_entropy':None}
    basepol=POL[f'gaussian32_s{int(label.rsplit("_s",1)[1])}']
    def selector(step,kind):
        selected=policy_for(label,step,records[-1] if records else None)
        return None if selected is None else selected[kind]
    with install(adapter,cfg['library'],basepol,mode='value',projections=projections,torch_library=cfg['torch_library'],collect=True) as (binding,router):
        router.policy_selector=selector;_set_context(binding,row)
        with observe(adapter.model,fixed_steps=False,draft_score=lambda t: score(row,adapter.tokenizer.decode(t,skip_special_tokens=True))) as trace:
            # observe cannot inject trigger before the next attention call, so infer
            # from the previous completed step and set trigger for the selector.
            out=adapter.generate(_request(row))
        routing=router.records()
    return dict(id=row['id'],condition=label,score=score(row,out.text),steps=len(trace),prediction=out.text,
                completion_tokens=out.completion_tokens,metadata=out.metadata,drafts=trace,routing_counts=routing_counts(routing),
                routing_path=None,wall_seconds=float(out.elapsed_seconds)),routing

def main(root):
    root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((BASE/'configuration.json').read_text())
    adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load(); projections=Projections()
    for row in ROWS:
      for label in CONDITIONS:
        path=shard_path(root/'adaptive',label,row)
        if path.exists():continue
        try:
          rec,routing=run_one(adapter,row,label,cfg,projections); path.parent.mkdir(parents=True,exist_ok=True)
          raw=path.with_suffix('.routing.json.gz');
          with gzip.open(raw,'wt') as f:json.dump(routing,f)
          rec['routing_path']=str(raw.resolve());rec['routing_sha256']=sha(raw);atomic(path,rec)
          print(json.dumps({'id':row['id'],'condition':label,'steps':rec['steps'],'score':rec['score']}),flush=True)
        except Exception as e:
          with (root/'failures.jsonl').open('a') as f:f.write(json.dumps({'id':row['id'],'condition':label,'error':repr(e),'traceback':traceback.format_exc()})+'\n')
          torch.cuda.empty_cache()
    atomic(root/'status.json',{'complete':all(shard_path(root/'adaptive',c,r).exists() for c in CONDITIONS for r in ROWS),'updated':time.time()})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--smoke',action='store_true');a=p.parse_args();
 if a.smoke: ROWS[:]=ROWS[:2]
 main(a.root)
