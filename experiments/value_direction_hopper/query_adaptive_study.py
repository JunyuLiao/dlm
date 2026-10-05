"""Resumable matched-sparsity RULER4K study of query-weighted Gaussian-32.

Run ``python -m experiments.value_direction_hopper.query_adaptive_study launch``
after building the v4 kernel/ATen bridge in ROOT/build. No training or decoder
modification occurs here; only the row routing score changes.
"""
import argparse
from collections import defaultdict,Counter
from contextlib import nullcontext
from copy import deepcopy
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import statistics
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
from .experiment import atomic,sha,fingerprint,shard_path
from .integration import install
from .query_adaptive import METHODS,State,observe
from .sparsity_steps import routing_counts

ROOT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3'
SOURCE=Path('results/value_direction_sparsity_steps_v2')
TRAJECTORY=Path('results/value_direction_trajectory_v2')
TARGETS=(50,70)
ADAPTIVE=tuple(dict(name=m,method=m) for m in METHODS)


def specs():
    result={name:dict(name=name,method=name,allocation='normal',bootstrap=False)
            for name in ('native_dense','kernel_dense','unweighted')}
    for method in METHODS:result[method]=dict(name=method,method=method,allocation='normal',bootstrap=False)
    for allocation in ('shuffle','uniform'):
        name='CT_'+allocation
        result[name]=dict(name=name,method='CT',allocation=allocation,bootstrap=False)
    result['bootstrap_unweighted']=dict(name='bootstrap_unweighted',method='unweighted',allocation='normal',bootstrap=True)
    result['bootstrap_CT']=dict(name='bootstrap_CT',method='CT',allocation='normal',bootstrap=True)
    return result


def conditions():
    result=[('native_dense',None),('kernel_dense',None)]
    for target in TARGETS:
        for name in ('unweighted',)+METHODS+('CT_shuffle','CT_uniform','bootstrap_unweighted','bootstrap_CT'):
            result.append((name,target))
    return result


def label(name,target):return name if target is None else f'{name}_s{target}'


def prepare(root):
    old=json.loads((SOURCE/'configuration.json').read_text())
    rows=json.loads((SOURCE/'manifest.json').read_text())
    cal=json.loads((SOURCE/'calibration_manifest.json').read_text())
    if len(rows)!=130 or len(cal)!=26 or len({r['task'] for r in rows})!=13:raise ValueError('Wrong evaluation/calibration shape')
    if set(Counter(r['task'] for r in rows).values())!={10} or set(Counter(r['task'] for r in cal).values())!={2}:raise ValueError('Task imbalance')
    for key in ('id','source_id','prompt_hash'):
        if {r[key] for r in rows}&{r[key] for r in cal}:raise ValueError(f'Calibration overlap: {key}')
    kernel=list((root/'build').glob('value_direction_*.so'))
    bridge=list((root/'build').glob('torch_*/value_direction_torch_*.so'))
    if len(kernel)!=1 or len(bridge)!=1:raise ValueError('Build exactly one versioned v4 kernel and bridge before launch')
    paths=[Path(__file__),Path(__file__).with_name('query_adaptive.py'),Path(__file__).with_name('integration.py'),
        Path(__file__).with_name('cuda.py'),Path(__file__).parent/'csrc/value_direction.cu',
        Path(__file__).parent/'csrc/value_direction.h',Path(__file__).parent/'csrc/torch_bridge.cpp',
        Path('src/dllm/models/adapters/diffusion_gemma.py'),
        Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
        Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py'),
        SOURCE/'configuration.json',SOURCE/'manifest.json',SOURCE/'calibration_manifest.json',kernel[0],bridge[0]]
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as generation
    paths.append(Path(generation.__file__))
    config=dict(schema='query_adaptive_v1',source=str(SOURCE.resolve()),model=old['model'],revision=old['revision'],
        library=str(kernel[0].resolve()),torch_library=str(bridge[0].resolve()),
        native_dense_backend='Original SDPA; local window and numeric path differ from the matched kernel',
        baseline_binary=str(Path(old['library']).resolve()),
        source_hashes={str(p.resolve()):sha(p) for p in paths},
        base_policies={t:old['policies'][f'gaussian32_s{t}'] for t in TARGETS},
        methods=specs(),conditions=conditions(),seed=42,projection_seed=1729,rank=32,
        beta=3.,gamma=.5,physical_tile=[128,64],canvas=256,max_steps=48,
        temperature='native 0.8 to 0.4, t=.4+.4*(remaining_step/48)',
        margin='Top-2 processed-logit gap multiplied by native temperature; the frozen run has only this logits processor',
        calibration='26 disjoint task-balanced prompts; each policy tested on complete native adaptive generation; coordinate bisection, max 7 points; local/global measured independently',
        sparsity='summed skipped eligible 128x64 physical tiles divided by summed eligible, across complete denoising; prefix encoding excluded',
        exposure='The 130 final questions have been repeatedly examined; development benchmark, not fresh held-out confirmation')
    config['fingerprint']=fingerprint(config)
    path=root/'configs/configuration.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(config)):
            raise ValueError('Frozen configuration/source changed; choose a new root')
    else:
        atomic(path,config);atomic(root/'configs/final_manifest.json',rows);atomic(root/'configs/calibration_manifest.json',cal)
    return config,rows,cal


def add_counts(a,b):
    for kind in ('whole','local','global'):
        for key in ('eligible','skipped'):a[kind][key]+=b[kind][key]


def empty_counts():return {k:dict(eligible=0,skipped=0) for k in ('whole','local','global')}


def aggregate(rows):
    counts=empty_counts()
    for r in rows:add_counts(counts,r['counts'])
    return {k:(v['skipped']/v['eligible'] if v['eligible'] else 0.) for k,v in counts.items()},counts


def generate(adapter,row,spec,policy,config,projections,m_ref,*,diagnostics=True,collect_margins=False):
    method=spec['method'];native=method=='native_dense'
    ctx=nullcontext((None,None)) if native else install(adapter,config['library'],policy,mode='value',
        projections=projections,torch_library=config['torch_library'],collect=True)
    with ctx as (binding,router):
        if binding:_set_context(binding,row)
        state=State(method,router,m_ref=m_ref,beta=config['beta'],gamma=config['gamma'],
            allocation=spec['allocation'],bootstrap=spec['bootstrap'],seed=row['seed'],
            diagnostics=diagnostics,collect_margins=collect_margins)
        with observe(adapter.model,state):
            torch.cuda.synchronize();start=time.perf_counter()
            output=adapter.generate(_request(row))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
        routing=[] if router is None else router.records()
    n=output.metadata['actual_denoising_step_count']
    if diagnostics and len(state.steps)!=n:raise AssertionError('Native decoder call count mismatch')
    if output.metadata['native_canvas_length']!=256 or len(state.canvases)!=1 and diagnostics:
        raise ValueError('RULER4K manifest no longer has one 256-position canvas; revise accounting')
    counts=routing_counts(routing)
    by_step=defaultdict(empty_counts)
    for item in routing:
        key=item['step']+1
        if key<1 or key>n:raise ValueError('Routing step outside decoder trajectory')
        for kind in ('whole',item['attention_type']):
            by_step[key][kind]['eligible']+=item['eligible'];by_step[key][kind]['skipped']+=item['skipped']
    if diagnostics:
        for step in state.steps:step['counts']=by_step[step['iteration']]
        per_canvas=dict(state.canvases[0],counts=counts,generated_tokens=len(output.completion_tokens),
            returned_tokens=len(output.completion_tokens))
    else:per_canvas=dict(canvas_index=0,iterations=n,counts=counts)
    result=dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],
        method=spec['name'],target=None,score=score(row,output.text),prediction=output.text,
        completion_tokens=output.completion_tokens,metadata=output.metadata,
        steps=n,seconds=seconds,counts=counts,canvas=per_canvas,step_records=state.steps,
        projection_manifest=projections.manifest if not native else {})
    if collect_margins:result['margin_values']=state.margin_samples
    return result,routing


def cached(adapter,root,stage,name,target,row,spec,policy,cfg,projections,m_ref,*,collect_margins=False):
    token=label(name,target)
    identity=fingerprint([cfg['fingerprint'],stage,token,spec,policy,m_ref,row['id'],row['prompt_hash'],row['seed']])
    dest=shard_path(root/stage,token,row)
    if dest.exists():
        result=json.loads(dest.read_text())
        if result['identity']!=identity:raise ValueError(f'Shard provenance mismatch: {dest}')
        if result.get('routing_path') and sha(result['routing_path'])!=result['routing_sha256']:raise ValueError('Corrupt routing shard')
        return result
    atomic(root/'status.json',dict(stage=stage,condition=token,id=row['id'],updated=time.time(),pid=os.getpid()))
    result,routing=generate(adapter,row,spec,policy,cfg,projections,m_ref,collect_margins=collect_margins)
    result.update(identity=identity,condition=token,target=target,
                  policy='all-retained' if name=='kernel_dense' else policy,status='complete')
    if routing:
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=dest.with_suffix('.routing.json.gz');temporary=raw.with_suffix('.tmp')
        with gzip.open(temporary,'wt',compresslevel=3) as file:json.dump(routing,file)
        os.replace(temporary,raw);result.update(routing_path=str(raw.resolve()),routing_sha256=sha(raw))
    atomic(dest,result)
    print(json.dumps(dict(event='complete',stage=stage,condition=token,id=row['id'],
        steps=result['steps'],sparsity=round(result['counts']['whole']['skipped']/max(1,result['counts']['whole']['eligible']),4),
        seconds=round(result['seconds'],3))),flush=True)
    return result


def reference_margin(adapter,root,cfg,cal,projections):
    path=root/'configs/margin_reference.json'
    if path.exists():return json.loads(path.read_text())['m_ref']
    spec=cfg['methods']['kernel_dense'];policy={k:dict(log_threshold=-math.inf) for k in ('local','global')}
    samples=[]
    for row in cal:
        result=cached(adapter,root,'margin_reference','kernel_dense',None,row,spec,policy,cfg,projections,1.,collect_margins=True)
        samples.extend(result['margin_values'])
    value=float(np.median(samples)) if samples else 1.
    value=max(value,1.e-5)
    atomic(path,dict(m_ref=value,positive_count=len(samples),rule='Median positive unscaled top1-top2 margin on all26 disjoint calibration trajectories',
        calibration_ids=[r['id'] for r in cal]))
    return value


def calibrate(adapter,root,cfg,cal,spec,name,target,projections,m_ref):
    dest=root/'configs/thresholds'/f'{label(name,target)}.json'
    if dest.exists():
        value=json.loads(dest.read_text())
        if value['fingerprint']!=cfg['fingerprint'] or value['calibration_ids']!=[r['id'] for r in cal]:raise ValueError('Frozen threshold provenance mismatch')
        return value['policy']
    base=cfg['base_policies'][target]
    current={k:base[k]['log_threshold']+(.0 if name=='unweighted' else .45 if not spec['bootstrap'] else .75)
             for k in ('local','global')}
    history=[]
    for iteration in range(7):
        policy={k:dict(log_threshold=current[k]) for k in current}
        key=fingerprint([name,target,policy])[:16]
        results=[cached(adapter,root,f'calibration/{label(name,target)}/{key}',name,target,row,spec,policy,cfg,projections,m_ref) for row in cal]
        actual,counts=aggregate(results)
        point=dict(iteration=iteration,policy=policy,actual=actual,counts=counts,
            total_steps=sum(r['steps'] for r in results))
        history.append(point);atomic(root/'calibration_traces'/f'{label(name,target)}.json',history)
        print(json.dumps(dict(event='calibration_point',condition=label(name,target),point=point)),flush=True)
        if all(abs(actual[k]-target/100)<=.02 for k in ('whole','local','global')):break
        next_values={}
        for kind in ('local','global'):
            # Generation-level sparsity can be nonmonotone. Use a conservative
            # secant only when two observed points bracket the requested value;
            # otherwise make a bounded directional step and keep the best point.
            candidates=[(p['actual'][kind],p['policy'][kind]['log_threshold']) for p in history]
            below=[x for x in candidates if x[0]<target/100];above=[x for x in candidates if x[0]>=target/100]
            if below and above:
                lo=max(below);hi=min(above)
                guess=(lo[1]+hi[1])/2
            else:guess=current[kind]+(.35 if actual[kind]<target/100 else -.35)
            next_values[kind]=float(np.clip(guess,current[kind]-.6,current[kind]+.6))
        if all(abs(next_values[k]-current[k])<1.e-5 for k in current):break
        current=next_values
    best=min(history,key=lambda p:(max(abs(p['actual'][k]-target/100) for k in ('whole','local','global')),
                                   sum(abs(p['actual'][k]-target/100) for k in ('local','global'))))
    frozen=dict(fingerprint=cfg['fingerprint'],condition=label(name,target),method=name,target=target,
        policy=best['policy'],calibration_actual=best['actual'],
        attained=all(abs(best['actual'][k]-target/100)<=.02 for k in ('whole','local','global')),
        calibration_ids=[r['id'] for r in cal],selection='Best of at most7 full-generation points by max global/local/overall target error',
        trace=history,heldout_used=False)
    atomic(dest,frozen)
    return best['policy']


def smoke(adapter,root,cfg,rows,projections,m_ref):
    path=root/'smoke.json'
    if path.exists():
        result=json.loads(path.read_text())
        if not result['passed'] or result['fingerprint']!=cfg['fingerprint']:raise ValueError('Invalid smoke record')
        return
    checks=[]
    selected=[rows[0],next(r for r in rows if r['task']=='fwe')]
    base=cfg['base_policies'][50]
    for row in selected:
        for name in ('native_dense','kernel_dense','unweighted','M','CT'):
            spec=deepcopy(cfg['methods'][name]);policy={k:dict(log_threshold=-math.inf) for k in ('local','global')} if name=='kernel_dense' else base
            first,_=generate(adapter,row,spec,policy,cfg,projections,m_ref)
            second,_=generate(adapter,row,spec,policy,cfg,projections,m_ref,diagnostics=False)
            if first['completion_tokens']!=second['completion_tokens'] or first['steps']!=second['steps']:
                raise AssertionError(f'Instrumentation alters {name} output {row["id"]}')
            old_name={'native_dense':'native_dense','kernel_dense':'kernel_dense','unweighted':'gaussian32_s50'}.get(name)
            archived=None
            if old_name:
                source=shard_path(SOURCE/'adaptive',old_name,row)
                old=json.loads(source.read_text());archived=first['completion_tokens']==old['completion_tokens'] and first['steps']==old['steps']
            checks.append(dict(id=row['id'],method=name,step=first['steps'],instrumentation_parity=True,archived_parity=archived))
        # A sensitivity vector of ones must reproduce unweighted decisions and
        # output. beta=0 makes all five formulae exactly one after history exists.
        spec=cfg['methods']['M'];saved=cfg['beta'];cfg['beta']=0.
        try:weighted,_=generate(adapter,row,spec,base,cfg,projections,m_ref)
        finally:cfg['beta']=saved
        plain,_=generate(adapter,row,cfg['methods']['unweighted'],base,cfg,projections,m_ref)
        if weighted['completion_tokens']!=plain['completion_tokens'] or weighted['counts']!=plain['counts'] or weighted['steps']!=plain['steps']:
            raise AssertionError('Unit sensitivity does not reproduce unweighted routing')
        checks.append(dict(id=row['id'],method='unit_weight',exact_parity=True))
    atomic(path,dict(passed=True,fingerprint=cfg['fingerprint'],checks=checks,
        backend_caveat='Native SDPA vs matched kernel may differ; archived parity explicitly recorded'))


def run(root,stage='run'):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,rows,cal=prepare(root)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        atomic(root/'status.json',dict(stage='loading',pid=os.getpid(),updated=time.time()))
        adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
        projections=Projections();m_ref=reference_margin(adapter,root,cfg,cal,projections)
        smoke(adapter,root,cfg,rows,projections,m_ref)
        if stage=='smoke':return
        frozen={}
        for target in TARGETS:
            for name in ('unweighted',)+METHODS+('CT_shuffle','CT_uniform','bootstrap_unweighted','bootstrap_CT'):
                frozen[label(name,target)]=calibrate(adapter,root,cfg,cal,cfg['methods'][name],name,target,projections,m_ref)
        atomic(root/'configs/frozen_policies.json',dict(m_ref=m_ref,beta=cfg['beta'],gamma=cfg['gamma'],policies=frozen))
        if stage=='calibrate':return
        errors=[]
        for name,target in conditions():
            token=label(name,target);spec=cfg['methods'][name]
            policy=({k:dict(log_threshold=-math.inf) for k in ('local','global')} if name=='kernel_dense' else
                    None if name=='native_dense' else frozen[token])
            for index,row in enumerate(rows):
                try:cached(adapter,root,'final',name,target,row,spec,policy,cfg,projections,m_ref)
                except Exception as error:
                    item=dict(condition=token,id=row['id'],error=repr(error),traceback=traceback.format_exc())
                    errors.append(item);atomic(root/'failures.json',errors);print(json.dumps(item),flush=True)
                    if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                    torch.cuda.empty_cache()
        atomic(root/'configs/projection_matrices.json',projections.manifest)
        from .query_adaptive_report import report
        report(root)
        atomic(root/'status.json',dict(stage='complete' if not errors else 'incomplete',errors=len(errors),pid=os.getpid(),updated=time.time()))


def launch(root,stage):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.query_adaptive_study',stage,'--root',str(root)]
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    logfile=root/f'worker_{int(time.time())}.log'
    with logfile.open('xb') as out:
        child=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(root/'launch.json',dict(pid=child.pid,log=str(logfile),command=cmd,started=time.time()))
    print(json.dumps(dict(pid=child.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('smoke','calibrate','run','launch','report'))
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    if args.stage=='launch':launch(args.root,'run')
    elif args.stage=='report':
        from .query_adaptive_report import report
        report(args.root)
    else:run(args.root,args.stage)
