"""Versioned tile-mean/max controls for the frozen temporal-sensitivity study.

The prior T/shuffle/uniform worker and its configuration are untouched. This
extension changes only how causally available per-row T weights are broadcast
within a physical 128-query tile, before the unchanged CUDA row-score product.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic,fingerprint,sha,shard_path
from .integration import install
from .query_adaptive import State,observe
from .query_adaptive_study import aggregate,empty_counts
from .sparsity_steps import routing_counts


ROOT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_tile_controls_v2'
PARENT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1'
TARGETS=(50,70)
METHODS=('T_tile_mean','T_tile_max')


def tile_broadcast(values,mode,tile=128):
    """B,Q FP32 T weights -> B,Q tile statistic; partial last tile supported."""
    if mode not in ('mean','max'):raise ValueError(mode)
    blocks=[]
    for start in range(0,values.shape[-1],tile):
        part=values[...,start:start+tile]
        reduced=part.mean(-1,keepdim=True) if mode=='mean' else part.amax(-1,keepdim=True)
        blocks.append(reduced.expand_as(part))
    return torch.cat(blocks,dim=-1).contiguous()


class TileState(State):
    def __init__(self,*args,tile_mode,**kwargs):
        super().__init__(*args,**kwargs);self.tile_mode=tile_mode

    def begin(self,cur_step,canvas):
        super().begin(cur_step,canvas)
        if self.used_weights is None:return
        transformed=tile_broadcast(self.used_weights,self.tile_mode)
        self.used_weights=transformed
        if self.router is not None:self.router.query_sensitivity=transformed
        if self.diagnostics:
            x=transformed.detach().float().flatten()
            q=torch.quantile(x,torch.tensor([.1,.5,.9],device=x.device))
            self.current.update(sensitivity_mean=float(x.mean()),sensitivity_p10=float(q[0]),
                sensitivity_p50=float(q[1]),sensitivity_p90=float(q[2]))


def label(name,target):return f'{name}_s{target}'


def specs():
    return {name:dict(name=name,method='T',allocation='normal',bootstrap=False,
        tile_mode='mean' if name.endswith('mean') else 'max') for name in METHODS}


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    parent_cfg=json.loads((PARENT/'configs/configuration.json').read_text())
    parent_policies=json.loads((PARENT/'configs/frozen_policies.json').read_text())
    manifests={split:json.loads((PARENT/f'configs/{split}_manifest.json').read_text())
        for split in ('calibration','final','development')}
    if len(parent_policies['policies'])!=8 or len(manifests['final'])!=130 or len(manifests['calibration'])!=26:
        raise ValueError('Parent calibration/manifests are not complete')
    files=[Path(__file__),Path(__file__).with_name('query_adaptive.py'),
        Path(__file__).with_name('query_adaptive_study.py'),Path(__file__).with_name('integration.py'),
        Path(__file__).with_name('cuda.py'),Path(parent_cfg['library']),Path(parent_cfg['torch_library']),
        PARENT/'configs/configuration.json',PARENT/'configs/frozen_policies.json',
        *(PARENT/f'configs/{split}_manifest.json' for split in manifests)]
    hashes={str(p.resolve()):sha(p) for p in files}
    path=root/'configs/configuration.json'
    if path.exists():
        cfg=json.loads(path.read_text())
        if hashes!=cfg['source_hashes']:raise ValueError('Frozen source changed; use a new versioned root')
        return cfg,manifests
    cfg=dict(schema='query_adaptive_tile_controls_v2',source_hashes=hashes,
        parent_fingerprint=parent_cfg['fingerprint'],model=parent_cfg['model'],revision=parent_cfg['revision'],
        library=parent_cfg['library'],torch_library=parent_cfg['torch_library'],
        methods=specs(),seed=42,confirmation_seeds=parent_cfg['confirmation_seeds'],
        beta=parent_cfg['beta'],gamma=parent_cfg['gamma'],m_ref=parent_cfg['m_ref'],
        projection_seed=1729,rank=32,physical_tile=[128,64],canvas=256,max_steps=48,
        calibration_tolerance=parent_cfg['calibration_tolerance'],comparison_tolerance=parent_cfg['comparison_tolerance'],
        calibration='Match frozen parent T calibration actual whole/global/local with at most10 full-generation points per tile method and target',
        semantics='Previous-step T sensitivity, then 128-row tile mean or max broadcast; row-risk multiplication and physical max remain in unchanged v4 kernel')
    cfg['fingerprint']=fingerprint(cfg);atomic(path,cfg)
    for split,rows in manifests.items():atomic(root/f'configs/{split}_manifest.json',rows)
    return cfg,manifests


def generate(adapter,row,spec,policy,cfg,projections,*,diagnostics=True):
    with install(adapter,cfg['library'],policy,mode='value',projections=projections,
                 torch_library=cfg['torch_library'],collect=True) as (binding,router):
        _set_context(binding,row)
        state=TileState('T',router,m_ref=cfg['m_ref'],beta=cfg['beta'],gamma=cfg['gamma'],
            allocation='normal',bootstrap=False,seed=row['seed'],diagnostics=diagnostics,
            tile_mode=spec['tile_mode'])
        with observe(adapter.model,state):
            torch.cuda.synchronize();started=time.perf_counter()
            output=adapter.generate(_request(row))
            torch.cuda.synchronize();seconds=time.perf_counter()-started
        routing=router.records()
    calls=output.metadata['actual_denoising_step_count']
    if diagnostics and len(state.steps)!=calls:raise AssertionError('Decoder call count mismatch')
    if output.metadata['native_canvas_length']!=256 or (diagnostics and len(state.canvases)!=1):
        raise ValueError('Expected one full 256-position canvas')
    counts=routing_counts(routing);per_step=defaultdict(empty_counts)
    for item in routing:
        index=item['step']+1
        if not 1<=index<=calls:raise ValueError('Routing step out of bounds')
        for kind in ('whole',item['attention_type']):
            per_step[index][kind]['eligible']+=item['eligible']
            per_step[index][kind]['skipped']+=item['skipped']
    if diagnostics:
        for step in state.steps:step['counts']=per_step[step['iteration']]
        canvas=dict(state.canvases[0],counts=counts,generated_tokens=len(output.completion_tokens),
            returned_tokens=len(output.completion_tokens))
    else:canvas=dict(canvas_index=0,iterations=calls,counts=counts)
    result=dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],
        method=spec['name'],target=None,score=score(row,output.text),prediction=output.text,
        completion_tokens=output.completion_tokens,metadata=output.metadata,steps=calls,seconds=seconds,
        counts=counts,canvas=canvas,step_records=state.steps,projection_manifest=projections.manifest)
    return result,routing


def cached(adapter,root,stage,name,target,row,spec,policy,cfg,projections):
    token=label(name,target)
    identity=fingerprint([cfg['fingerprint'],stage,token,spec,policy,cfg['m_ref'],
        row['id'],row['prompt_hash'],row['seed']])
    dest=shard_path(root/stage,token,row)
    if dest.exists():
        result=json.loads(dest.read_text())
        if result['identity']!=identity:raise ValueError(f'Shard provenance mismatch {dest}')
        if result.get('routing_path') and sha(result['routing_path'])!=result['routing_sha256']:
            raise ValueError(f'Corrupt routing shard {dest}')
        return result
    atomic(root/'status.json',dict(stage=stage,condition=token,id=row['id'],pid=os.getpid(),updated=time.time()))
    result,routing=generate(adapter,row,spec,policy,cfg,projections)
    result.update(identity=identity,condition=token,target=target,policy=policy,status='complete')
    if routing:
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=dest.with_suffix('.routing.json.gz');temporary=raw.with_suffix('.tmp')
        with gzip.open(temporary,'wt',compresslevel=3) as f:json.dump(routing,f)
        os.replace(temporary,raw);result.update(routing_path=str(raw.resolve()),routing_sha256=sha(raw))
    atomic(dest,result)
    print(json.dumps(dict(event='complete',stage=stage,condition=token,id=row['id'],
        steps=result['steps'],sparsity=round(result['counts']['whole']['skipped']/max(1,result['counts']['whole']['eligible']),4))),flush=True)
    return result


def distance(actual,goal):return max(abs(actual[k]-goal[k]) for k in ('whole','local','global'))


def calibrate(adapter,root,cfg,rows,name,target,goal,projections):
    token=label(name,target);dest=root/'configs/thresholds'/f'{token}.json'
    if dest.exists():
        result=json.loads(dest.read_text())
        if result['fingerprint']!=cfg['fingerprint'] or result['calibration_ids']!=[r['id'] for r in rows]:
            raise ValueError('Threshold provenance mismatch')
        return result
    parent=json.loads((PARENT/'configs/frozen_policies.json').read_text())
    start=parent['policies'][label('T',target)]
    logs={kind:float(start[kind]['log_threshold']) for kind in ('local','global')}
    history=[]
    for index in range(10):
        policy={kind:dict(log_threshold=logs[kind]) for kind in ('local','global')}
        key=fingerprint([name,target,policy])[:16]
        results=[cached(adapter,root,f'calibration/{token}/{key}',name,target,row,cfg['methods'][name],
            policy,cfg,projections) for row in rows]
        actual,counts=aggregate(results)
        point=dict(round=index,policy=policy,actual=actual,counts=counts,
            total_steps=sum(r['steps'] for r in results))
        history.append(point);atomic(root/'calibration_traces'/f'{token}.json',history)
        if distance(actual,goal)<=cfg['calibration_tolerance']:break
        next_logs={}
        for kind in ('local','global'):
            seen=[(p['actual'][kind],p['policy'][kind]['log_threshold']) for p in history]
            below=[x for x in seen if x[0]<goal[kind]];above=[x for x in seen if x[0]>goal[kind]]
            if below and above:
                low=max(below,key=lambda x:x[0]);high=min(above,key=lambda x:x[0])
                ratio=(goal[kind]-low[0])/max(high[0]-low[0],1.e-8)
                value=low[1]+ratio*(high[1]-low[1])
            else:value=logs[kind]+max(-.5,min(.5,5*(goal[kind]-actual[kind])))
            value=float(max(logs[kind]-.6,min(logs[kind]+.6,value)))
            if abs(value-logs[kind])<1.e-5:value=logs[kind]+(.05 if actual[kind]<goal[kind] else -.05)
            next_logs[kind]=value
        logs=next_logs
    best=min(history,key=lambda p:(distance(p['actual'],goal),
        sum(abs(p['actual'][k]-goal[k]) for k in ('whole','local','global'))))
    result=dict(fingerprint=cfg['fingerprint'],condition=token,method=name,target=target,goal=goal,
        policy=best['policy'],calibration_actual=best['actual'],
        attained=distance(best['actual'],goal)<=cfg['calibration_tolerance'],
        calibration_ids=[r['id'] for r in rows],trace=history,
        selection='Best of at most10 full-generation points; final examples not used')
    atomic(dest,result);return result


def smoke(adapter,root,cfg,rows,projections):
    dest=root/'smoke.json'
    if dest.exists():
        value=json.loads(dest.read_text())
        if not value['passed'] or value['fingerprint']!=cfg['fingerprint']:raise ValueError('Invalid smoke record')
        return
    parent=json.loads((PARENT/'configs/frozen_policies.json').read_text())
    policy=parent['policies']['T_s50'];checks=[]
    for row in rows[:2]:
        for name in METHODS:
            spec=cfg['methods'][name]
            a,_=generate(adapter,row,spec,policy,cfg,projections)
            b,_=generate(adapter,row,spec,policy,cfg,projections,diagnostics=False)
            if a['completion_tokens']!=b['completion_tokens'] or a['steps']!=b['steps']:
                raise AssertionError(f'Instrumentation changed output {name} {row["id"]}')
            variant=dict(cfg,beta=0.)
            unit,_=generate(adapter,row,spec,policy,variant,projections)
            from .query_adaptive_study import generate as base_generate
            plain_spec=dict(name='unweighted',method='unweighted',allocation='normal',bootstrap=False)
            plain,_=base_generate(adapter,row,plain_spec,policy,cfg,projections,cfg['m_ref'])
            if unit['completion_tokens']!=plain['completion_tokens'] or unit['counts']!=plain['counts']:
                raise AssertionError(f'Unit-weight tile mode differs from unweighted {name}')
            checks.append(dict(id=row['id'],condition=name,instrumentation_parity=True,unit_weight_parity=True))
    atomic(dest,dict(passed=True,fingerprint=cfg['fingerprint'],checks=checks))


def run(root,stage='run'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests=prepare(root)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        atomic(root/'status.json',dict(stage='loading',pid=os.getpid(),updated=time.time()))
        adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
        projections=Projections();smoke(adapter,root,cfg,manifests['development'],projections)
        if stage=='smoke':return
        parent=json.loads((PARENT/'configs/frozen_policies.json').read_text())
        thresholds={}
        for target in TARGETS:
            reference=json.loads((PARENT/'configs/thresholds'/f'T_s{target}.json').read_text())
            for name in METHODS:
                value=calibrate(adapter,root,cfg,manifests['calibration'],name,target,
                    reference['calibration_actual'],projections)
                thresholds[label(name,target)]=value['policy']
        atomic(root/'configs/frozen_policies.json',dict(policies=thresholds,calibration_only=True,
            reference_T_policies={label('T',target):parent['policies'][label('T',target)] for target in TARGETS}))
        if stage=='calibrate':return
        failures=[]
        for target in TARGETS:
            for name in METHODS:
                token=label(name,target);policy=thresholds[token]
                for row in manifests['final']:
                    try:cached(adapter,root,'final',name,target,row,cfg['methods'][name],policy,cfg,projections)
                    except Exception as error:
                        value=dict(condition=token,id=row['id'],error=repr(error),traceback=traceback.format_exc())
                        failures.append(value);atomic(root/'failures.json',failures)
                        if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                        torch.cuda.empty_cache()
        atomic(root/'configs/projection_matrices.json',projections.manifest)
        from .query_adaptive_tile_controls_report import report
        report(root)
        atomic(root/'status.json',dict(stage='complete' if not failures else 'incomplete',errors=len(failures),
            pid=os.getpid(),updated=time.time()))


def confirm(root):
    root=Path(root);cfg,manifests=prepare(root)
    policies=json.loads((root/'configs/frozen_policies.json').read_text())['policies']
    if not json.loads((root/'audit.json').read_text())['complete']:raise ValueError('Primary audit incomplete')
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
        projections=Projections();errors=[]
        for seed in cfg['confirmation_seeds']:
            for target in TARGETS:
                for name in METHODS:
                    policy=policies[label(name,target)]
                    for original in manifests['final']:
                        row=dict(original,seed=seed)
                        try:cached(adapter,root,f'confirmation/seed{seed}',name,target,row,cfg['methods'][name],policy,cfg,projections)
                        except Exception as error:
                            value=dict(seed=seed,condition=label(name,target),id=row['id'],
                                error=repr(error),traceback=traceback.format_exc())
                            errors.append(value);atomic(root/'confirmation_failures.json',errors)
                            if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                            torch.cuda.empty_cache()
        from .query_adaptive_tile_controls_report import report
        report(root)
        atomic(root/'confirmation_status.json',dict(stage='complete' if not errors else 'incomplete',
            errors=len(errors),pid=os.getpid(),updated=time.time()))


def launch(root,stage):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','experiments.value_direction_hopper.query_adaptive_tile_controls',stage,
        '--root',str(root)]
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    logfile=root/f'{stage}_{int(time.time())}.log'
    with logfile.open('xb') as out:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(root/f'{stage}_launch.json',dict(pid=process.pid,log=str(logfile),command=command,started=time.time()))
    print(json.dumps(dict(pid=process.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','smoke','calibrate','run','confirm','launch','launch-confirm','report'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.root)
    elif args.stage=='launch':launch(args.root,'run')
    elif args.stage=='launch-confirm':launch(args.root,'confirm')
    elif args.stage=='confirm':confirm(args.root)
    elif args.stage=='report':
        from .query_adaptive_tile_controls_report import report
        report(args.root)
    else:run(args.root,args.stage)
