"""Separate untraced, interleaved H100 generation timing for frozen policies."""
import argparse
from contextlib import contextmanager,nullcontext
import csv
import json
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from .experiment import atomic,shard_path
from .integration import install
from .query_adaptive import State,observe


class ProfileState(State):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.policy_events=[]
    def begin(self,*args,**kwargs):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();result=super().begin(*args,**kwargs);b.record();self.policy_events.append((a,b));return result
    def observe_logits(self,*args,**kwargs):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();result=super().observe_logits(*args,**kwargs);b.record();self.policy_events.append((a,b));return result


@contextmanager
def decoder_events(model,events):
    old=model._denoising_step;had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step')
    def step(this,**kwargs):
        forward=kwargs['decoder_forward']
        def measured(*args,**kw):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();result=forward(*args,**kw);end.record();events.append((start,end));return result
        kwargs['decoder_forward']=measured
        return old(**kwargs)
    model._denoising_step=MethodType(step,model)
    try:yield
    finally:
        if had:model._denoising_step=saved
        else:del model._denoising_step


@contextmanager
def prefix_events(model,events):
    handles=[];starts=[]
    for module in model.modules():
        if type(module).__name__!='DiffusionGemmaEncoderModel':continue
        def before(*_):
            event=torch.cuda.Event(enable_timing=True);event.record();starts.append(event)
        def after(*_):
            event=torch.cuda.Event(enable_timing=True);event.record();events.append((starts.pop(),event))
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    try:yield
    finally:
        for handle in handles:handle.remove()


def one(adapter,row,name,cfg,policies,projections,m_ref,*,profile=False):
    if name=='native_dense':ctx=nullcontext((None,None))
    else:
        policy=({k:dict(log_threshold=-math.inf) for k in ('local','global')} if name=='kernel_dense' else policies[name])
        ctx=install(adapter,cfg['library'],policy,mode='value',projections=projections,
            torch_library=cfg['torch_library'],collect=False)
    with ctx as (binding,router):
        if binding:_set_context(binding,row)
        base=name.rsplit('_s',1)[0] if '_s' in name else name
        spec=cfg['methods'][base]
        state=(ProfileState if profile else State)(spec['method'],router,m_ref=m_ref,beta=cfg['beta'],gamma=cfg['gamma'],
            allocation=spec['allocation'],bootstrap=spec['bootstrap'],seed=row['seed'],diagnostics=False)
        events=[];encoding=[]
        needs_policy=spec['method'] in ('M','C','T','MT','CT') or spec['bootstrap']
        with (observe(adapter.model,state) if needs_policy or profile else nullcontext()):
            with (decoder_events(adapter.model,events) if profile else nullcontext()):
                with (prefix_events(adapter.model,encoding) if profile else nullcontext()):
                    torch.cuda.synchronize();started=time.perf_counter()
                    output=adapter.generate(_request(row))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-started
        return dict(id=row['id'],condition=name,seed=row['seed'],prompt_hash=row['prompt_hash'],
            steps=output.metadata['actual_denoising_step_count'],completion_tokens=output.completion_tokens,
            e2e_seconds=elapsed,decoder_seconds=sum(a.elapsed_time(b) for a,b in events)/1000 if profile else None,
            prefix_encoding_seconds=sum(a.elapsed_time(b) for a,b in encoding)/1000 if profile else None,
            sensitivity_update_seconds=sum(a.elapsed_time(b) for a,b in state.policy_events)/1000 if profile else None)


def run(root,repeats=2,limit=None):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    frozen=json.loads((root/'configs/frozen_policies.json').read_text())
    rows=json.loads((root/'configs/final_manifest.json').read_text())
    if limit is not None:rows=rows[:limit]
    # E2E timing covers every prompt. Event instrumentation is confined to
    # one deterministic prompt per task, paired across every timed condition.
    profile_ids={next(row['id'] for row in rows if row['task']==task)
        for task in {row['task'] for row in rows}}
    summary=json.loads((root/'summary.json').read_text())['summary']
    labels=['native_dense','kernel_dense','unweighted_s50','CT_s50','unweighted_s70','CT_s70']
    # At most one other calibration-frozen candidate per target, selected by
    # final exploratory Pareto status for timing only; no threshold retuning.
    for target in (50,70):
        candidates=[x for x in summary if x['target_sparsity']==target and x['method'] in ('M','C','T','MT')]
        baseline=next((x for x in summary if x['condition']==f'unweighted_s{target}'),None)
        if baseline:
            candidates=[x for x in candidates if x['accuracy']>=baseline['accuracy'] and
                x['mean_iterations']<=baseline['mean_iterations'] and
                abs(x['sparsity']-baseline['sparsity'])<=.02]
            if candidates:labels.append(min(candidates,key=lambda x:x['mean_iterations'])['condition'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
    projections=Projections();policies=frozen['policies']
    for name in labels:one(adapter,rows[0],name,cfg,policies,projections,frozen['m_ref'])
    for repeat in range(repeats):
        for index,row in enumerate(rows):
            order=labels[(index+repeat)%len(labels):]+labels[:(index+repeat)%len(labels)]
            for name in order:
                paired=[]
                for mode in (('e2e','profile') if row['id'] in profile_ids else ('e2e',)):
                    dest=shard_path(root/f'timing/{mode}/repeat_{repeat}',name,row)
                    if dest.exists():result=json.loads(dest.read_text())
                    else:
                        result=one(adapter,row,name,cfg,policies,projections,frozen['m_ref'],profile=mode=='profile')
                        result.update(repeat=repeat,mode=mode);atomic(dest,result)
                    primary_path=shard_path(root/'final',name,row)
                    primary=json.loads(primary_path.read_text())
                    if result['completion_tokens']!=primary['completion_tokens'] or result['steps']!=primary['steps']:
                        raise AssertionError(f'Timing/primary generation mismatch: {name} {row["id"]} repeat={repeat} mode={mode}')
                    paired.append(result)
                    print(json.dumps(dict(event='timing',mode=mode,repeat=repeat,condition=name,id=row['id'],
                        e2e_seconds=result['e2e_seconds'],steps=result['steps'])),flush=True)
                if len(paired)==2 and (paired[0]['completion_tokens']!=paired[1]['completion_tokens'] or
                    paired[0]['steps']!=paired[1]['steps']):
                    raise AssertionError('Timing profile changes generation')
    records=[]
    for repeat in range(repeats):
        for name in labels:
            for row in rows:
                for mode in (('e2e','profile') if row['id'] in profile_ids else ('e2e',)):
                    dest=shard_path(root/f'timing/{mode}/repeat_{repeat}',name,row)
                    if dest.exists():records.append(json.loads(dest.read_text()))
    with (root/'timing.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader();writer.writerows(records)
    table=[]
    for name in labels:
        e2e=[r for r in records if r['condition']==name and r['mode']=='e2e']
        profiled=[r for r in records if r['condition']==name and r['mode']=='profile']
        mean_e2e=float(np.mean([r['e2e_seconds'] for r in e2e]))
        mean_decoder=float(np.mean([r['decoder_seconds'] for r in profiled]))
        mean_prefix=float(np.mean([r['prefix_encoding_seconds'] for r in profiled]))
        table.append(dict(condition=name,n=len(e2e),profile_n=len(profiled),mean_e2e_seconds=mean_e2e,
            mean_decoder_seconds=mean_decoder,mean_prefix_encoding_seconds=mean_prefix,
            mean_unattributed_residual_seconds=None,
            mean_sensitivity_update_seconds=float(np.mean([r['sensitivity_update_seconds'] for r in profiled])),
            mean_steps=float(np.mean([r['steps'] for r in e2e]))))
    dense=next(r for r in table if r['condition']=='native_dense')['mean_e2e_seconds']
    for r in table:r['speed_ratio_vs_native_dense']=dense/r['mean_e2e_seconds']
    atomic(root/'timing_summary.json',dict(complete=len(rows)==130 and repeats>=2,
        table=table,repeats=repeats,rows=len(rows),warmup='One row per condition, excluded',
        interleaving='Rotated condition order by example and repeat',
        precision='E2E synchronized wall from all untraced runs; decoder/policy CUDA events from one task-balanced paired example per task and repeat',
        caveat='Profiled subset is the same across conditions but is not the full 130 prompts. E2E-minus-decoder-minus-encoder subtracts different runs/populations and must not be interpreted as fixed overhead; other CPU work is unattributed'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    parser.add_argument('--repeats',type=int,default=2);parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.root,args.repeats,args.limit)
