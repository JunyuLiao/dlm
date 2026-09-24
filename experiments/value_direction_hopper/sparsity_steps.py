"""Frozen-kernel RULER4K sparsity/iterations sweep and exact-four-step control."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MethodType

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, sha, fingerprint, shard_path
from .integration import install

BASE=Path('results/value_direction_hopper_v1/final_ruler4k130_v1')
TRAJECTORY=Path('results/value_direction_trajectory_v2')
TARGETS=(40,50,60,65,70)
FAMILIES=('gaussian32','blasst')
ROOT=Path('results/value_direction_sparsity_steps_v2')
SOURCES={50:Path('results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16'),
         60:Path('results/diffusion_gemma_ruler4k_value_direction_s60_v17'),
         65:Path('results/diffusion_gemma_ruler4k_value_direction_s65_v18'),
         70:Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19')}


def conditions():
    return {'adaptive':['native_dense','kernel_dense']+[f'{m}_s{t}' for t in TARGETS for m in FAMILIES],
            'fixed4':['native_dense','kernel_dense']+[f'{m}_s{t}' for t in (50,70) for m in FAMILIES]}


def prepare(root):
    old=json.loads((BASE/'configuration.json').read_text())
    rows=json.loads((BASE/'manifest.json').read_text())
    calibration=json.loads((SOURCES[70]/'calibration_manifest.json').read_text())
    for key in ('id','prompt_hash','source_id'):
        if {r[key] for r in rows}&{r[key] for r in calibration}:raise ValueError('Calibration/final overlap')
    if len(rows)!=130 or Counter(r['task'] for r in rows)!=dict.fromkeys({r['task'] for r in rows},10):
        raise ValueError('Incorrect final cohort')
    policies={};files=[BASE/'configuration.json',BASE/'manifest.json',SOURCES[70]/'calibration_manifest.json',
        Path(old['library']),Path(old['torch_library']),Path(__file__),Path(__file__).with_name('integration.py'),
        Path(__file__).with_name('cuda.py'),Path(__file__).with_name('projection.py'),Path(__file__).with_name('masks.py'),
        Path('src/dllm/models/adapters/diffusion_gemma.py'),Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
        Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py')]
    for target,source in SOURCES.items():
        for family,name in [('gaussian32','jl_gaussian_r32'),('blasst','blasst')]:
            path=source/'policies/ruler4k'/f'{name}_s{target}.json'
            data=json.loads(path.read_text());files.append(path)
            if set(data['calibration_ids'])!={r['id'] for r in calibration} or data['heldout_used']:
                raise ValueError('Historical policy calibration provenance mismatch')
            policies[f'{family}_s{target}']=data['policy']
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as generation
    files.append(Path(generation.__file__))
    config=dict(schema='ruler4k_sparsity_steps_v1',model=old['model'],revision=old['revision'],
        library=old['library'],torch_library=old['torch_library'],
        source_hashes={str(p.resolve()):sha(p) for p in files},policies=policies,conditions=conditions(),
        ids=[r['id'] for r in rows],calibration_ids=[r['id'] for r in calibration],
        precision='BF16; frozen873c3073e54a37b8 kernel; FP32 TF32x3 projection',projection_seed=1729,
        physical_tile=[128,64],canvas=256,adaptive_max_steps=48,seed=42,
        fixed4='Exactly4 calls per problem; suppress adaptive stopping; native4-step temperature schedule0.8,0.7,0.6,0.5',
        supplemental65='Pre-existing calibrated65% point rerun with the same kernel to localize the rise',
        calibration40='Independent local/global scalar bisection on26 calibration prompts; actual count-weighted sparsity;2pp tolerance;max12 points',
        threshold_policy='Reuse frozen50/60/65/70 policies;40 calibrated before final evaluation; no final-score tuning',
        blasst='Existing aggressive-capable online maximum rule. At40 allow above1 too, so local/global targets can be matched independently.',
        mask_caveat=old['mask_caveat'],exposure='Previously examined130 RULER4K examples; not fresh held-out confirmation')
    config['fingerprint']=fingerprint(config)
    path=root/'configuration.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise ValueError('Source/configuration changed; use a new run directory')
    else:
        atomic(path,config);atomic(root/'manifest.json',rows);atomic(root/'calibration_manifest.json',calibration)
        from .provenance import snapshot
        snapshot(root)
    return config,rows,calibration


class StoppingProbe:
    def __init__(self,inner,record,force):self.inner,self.record,self.force=inner,record,force
    def __call__(self,*args,**kwargs):
        result=self.inner(*args,**kwargs)
        self.record['would_stop']=bool(result.item())
        return torch.zeros_like(result) if self.force else result


@contextmanager
def drafts(adapter,row,force):
    model=adapter.model;original=model._denoising_step
    had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step');records=[]
    def step(this,**kwargs):
        item=dict(step=len(records)+1,remaining_schedule_step=int(kwargs['cur_step']),
                  input_length=int(kwargs['input_ids'].shape[-1]))
        stopper=kwargs['diffusion_stopping_criteria']
        if stopper is None:raise ValueError('Missing native stopping criterion')
        kwargs['diffusion_stopping_criteria']=StoppingProbe(stopper,item,force)
        result=original(**kwargs)
        tokens,_=adapter._completion(result[1][0],0,row['generation_budget'])
        text=adapter.tokenizer.decode(tokens,skip_special_tokens=True)
        item.update(tokens=tokens,score=score(row,text),stopped=bool(result[3].item()))
        records.append(item)
        return result
    model._denoising_step=MethodType(step,model)
    try:yield records
    finally:
        if had:model._denoising_step=saved
        else:del model._denoising_step


def routing_counts(routing):
    result={kind:dict(eligible=0,skipped=0) for kind in ('whole','local','global')}
    for r in routing:
        if not 0<=r['skipped']<=r['eligible']:raise ValueError('Invalid physical counts')
        for kind in ('whole',r['attention_type']):
            for key in ('eligible','skipped'):result[kind][key]+=r[key]
    return result


def generate(adapter,row,condition,regime,config,projections,policy=None,observe=True):
    if condition=='native_dense':ctx=nullcontext((None,None));selected=None
    else:
        value=condition=='kernel_dense' or condition.startswith('gaussian32')
        selected=deepcopy(policy if policy is not None else config['policies'].get(condition))
        if condition=='kernel_dense':selected={k:dict(log_threshold=-float('inf')) for k in ('local','global')}
        ctx=install(adapter,config['library'],selected,mode='value' if value else 'blasst',
                    projections=projections,torch_library=config['torch_library'],collect=True)
    request=_request(row)
    if regime=='fixed4':request=replace(request,steps=4)
    with ctx as (binding,router):
        if binding:_set_context(binding,row)
        with (drafts(adapter,row,regime=='fixed4') if observe else nullcontext([])) as recorded:
            torch.cuda.synchronize();started=time.perf_counter()
            output=adapter.generate(request)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
        routing=[] if router is None else router.records()
        counts=routing_counts(routing)
        steps=output.metadata['actual_denoising_step_count']
        if observe and steps!=len(recorded):raise ValueError('Direct and metadata step counts differ')
        if regime=='fixed4' and steps!=4:raise ValueError('Fixed4 did not execute4 steps')
        if recorded and (len({r['input_length'] for r in recorded})!=1 or output.metadata['native_canvas_length']!=256):
            raise ValueError('More than one canvas: revise per-block accounting')
        accuracy=score(row,output.text)
        if recorded and (recorded[-1]['tokens']!=output.completion_tokens or recorded[-1]['score']!=accuracy):
            raise ValueError('Draft/final discrepancy')
        result=dict(status='complete',fingerprint=config['fingerprint'],condition=condition,regime=regime,
            id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],
            generation_budget=row['generation_budget'],metadata=output.metadata,completion_tokens=output.completion_tokens,
            prediction=output.text,score=accuracy,steps=steps,canvases=1,counts=counts,drafts=recorded,wall_seconds=seconds,
            policy=('all-retained' if condition=='kernel_dense' else selected),
            projection_manifest=projections.manifest if condition.startswith('gaussian32') else {})
        return result,routing


def save(root,record,routing):
    path=shard_path(root/record['regime'],record['condition'],record)
    path.parent.mkdir(parents=True,exist_ok=True)
    if routing:
        raw=path.with_suffix('.routing.json.gz');tmp=raw.with_suffix('.tmp')
        with gzip.open(tmp,'wt',compresslevel=3) as f:json.dump(routing,f,allow_nan=False)
        os.replace(tmp,raw);record['routing_path']=str(raw.resolve());record['routing_sha256']=sha(raw)
    atomic(path,record)


def validate_cached(path,config,row,condition,regime):
    r=json.loads(path.read_text())
    for key,wanted in [('fingerprint',config['fingerprint']),('id',row['id']),('condition',condition),('regime',regime),
                      ('prompt_hash',row['prompt_hash']),('seed',row['seed']),('generation_budget',row['generation_budget'])]:
        if r[key]!=wanted:raise ValueError(f'Shard {path} differs: {key}')
    if r['status']!='complete':raise ValueError('Incomplete shard')
    if r.get('routing_path') and sha(r['routing_path'])!=r['routing_sha256']:raise ValueError('Corrupt routing')
    return r


def smoke(adapter,root,config,rows,projections):
    path=root/'smoke.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data['fingerprint']!=config['fingerprint'] or not data['passed']:raise ValueError('Invalid smoke')
        return
    checks=[]
    for row in (rows[0],next(r for r in rows if r['task']=='fwe')):
        for condition,oldname in [('native_dense','native_dense'),('kernel_dense','kernel_dense'),
                                  ('gaussian32_s70','kernel_gaussian32'),('blasst_s70','kernel_blasst')]:
            a,routing=generate(adapter,row,condition,'adaptive',config,projections)
            b,_=generate(adapter,row,condition,'adaptive',config,projections,observe=False)
            old=json.loads(shard_path(TRAJECTORY/'adaptive',oldname,row).read_text())
            if not(a['completion_tokens']==b['completion_tokens']==old['completion_tokens'] and
                   a['steps']==b['steps']==old['metadata']['actual_denoising_step_count']):
                raise ValueError('Migration/instrumentation archived reproduction failed')
            checks.append(dict(id=row['id'],condition=condition,steps=a['steps'],exact_parity=True))
            save(root,a,routing)
        fixed,_=generate(adapter,row,'native_dense','fixed4',config,projections)
        if [r['remaining_schedule_step'] for r in fixed['drafts']]!=[4,3,2,1]:raise ValueError('Fixed4 schedule')
        save(root,fixed,[])
    atomic(path,dict(passed=True,fingerprint=config['fingerprint'],checks=checks,fixed4_checked=True))


def import_cached(root,config,rows):
    mapping={'native_dense':'native_dense','kernel_dense':'kernel_dense',
             'gaussian32_s70':'kernel_gaussian32','blasst_s70':'kernel_blasst'}
    for condition,oldname in mapping.items():
        for row in rows:
            dest=shard_path(root/'adaptive',condition,row)
            if dest.exists():validate_cached(dest,config,row,condition,'adaptive');continue
            source=shard_path(TRAJECTORY/'adaptive',oldname,row);old=json.loads(source.read_text())
            if any(old[k]!=row[k] for k in ('id','prompt_hash','seed','generation_budget')):raise ValueError('Imported sample mismatch')
            raw=Path(old['trace_path'])
            if sha(raw)!=old['trace_sha256']:raise ValueError('Imported trace hash')
            with gzip.open(raw,'rt') as f:trace=json.load(f)
            ds=[]
            for s in trace['steps']:
                ds.append(dict(step=s['step'],remaining_schedule_step=s['remaining_schedule_step'],input_length=s['input_length'],
                    score=s['draft_score'],would_stop=s['would_stop'],stopped=s['terminated']))
            if len({s['input_length'] for s in ds})!=1:raise ValueError('Imported multiple canvases')
            record=dict(status='complete',fingerprint=config['fingerprint'],condition=condition,regime='adaptive',
                **{k:old[k] for k in ('id','task','prompt_hash','seed','generation_budget','metadata','completion_tokens','prediction','score','wall_seconds')},
                steps=len(ds),canvases=1,drafts=ds,counts=routing_counts(trace['routing']),
                policy=config['policies'].get(condition,'all-retained' if condition=='kernel_dense' else None),
                source=dict(path=str(source.resolve()),sha256=sha(source),trace_path=str(raw),trace_sha256=sha(raw)),
                timing_caveat='Imported instrumented diagnostic timing; no latency comparison in this study')
            save(root,record,trace['routing'])


def calibrate40(adapter,root,config,calibration,projections):
    for family in FAMILIES:
        label=family+'_s40';path=root/'policies'/(label+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            if data['fingerprint']!=config['fingerprint'] or data['calibration_ids']!=config['calibration_ids']:raise ValueError('Calibration identity')
            config['policies'][label]=data['policy'];continue
        field='log_threshold' if family=='gaussian32' else 'log_scale'
        prior=config['policies'][family+'_s50']
        low={k:(-8. if family=='gaussian32' else 0.) for k in ('local','global')}
        high={k:prior[k][field] for k in low}
        # Frozen s50 calibration gives an upper starting point. First test it
        # on this numerical implementation before using it as an upper bound.
        trace=[];selected=None
        for iteration in range(12):
            trial={k:(high[k] if iteration==0 else (low[k]+high[k])/2) for k in low}
            policy={k:{field:v,**({'cap_one':False} if family=='blasst' else {})} for k,v in trial.items()}
            digest=fingerprint(policy)[:16];totals={k:dict(eligible=0,skipped=0) for k in ('whole','local','global')}
            for row in calibration:
                regime='calibration/'+label+'/'+digest
                shard=shard_path(root/regime,label,row)
                if shard.exists():record=validate_cached(shard,config,row,label,regime)
                else:
                    record,routing=generate(adapter,row,label,regime,config,projections,policy=policy)
                    save(root,record,routing)
                for k in totals:
                    for n in totals[k]:totals[k][n]+=record['counts'][k][n]
            achieved={k:v['skipped']/v['eligible'] for k,v in totals.items()}
            point=dict(iteration=iteration,policy=policy,achieved=achieved,counts=totals)
            trace.append(point);atomic(root/'calibration_traces'/(label+'.json'),trace)
            print(json.dumps(dict(event='calibration',condition=label,**point)),flush=True)
            if max(abs(v-.4) for v in achieved.values())<=.02:selected=point;break
            for k in low:
                if achieved[k]<.4:
                    low[k]=trial[k]
                    if trial[k]>=high[k]:high[k]=trial[k]+1.
                else:high[k]=trial[k]
        if selected is None:raise RuntimeError('No40% calibration within2pp; trace preserved; final40 not run')
        atomic(path,dict(fingerprint=config['fingerprint'],policy=selected['policy'],target=.4,
            achieved=selected['achieved'],calibration_ids=config['calibration_ids'],heldout_used=False,trace=trace))
        config['policies'][label]=selected['policy']


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,rows,calibration=prepare(root)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        atomic(root/'status.json',dict(stage='loading',pid=os.getpid(),updated=time.time()))
        adapter=create_adapter('diffusion_gemma',config['model'],device='cuda',precision='bfloat16',revision=config['revision']).load()
        projections=Projections()
        smoke(adapter,root,config,rows,projections)
        import_cached(root,config,rows)
        atomic(root/'status.json',dict(stage='calibration40',pid=os.getpid(),updated=time.time()))
        calibrate40(adapter,root,config,calibration,projections)
        atomic(root/'frozen_policies.json',config['policies'])
        failures=[]
        for regime,labels in conditions().items():
            for i,row in enumerate(rows):
                for label in labels[i%len(labels):]+labels[:i%len(labels)]:
                    path=shard_path(root/regime,label,row)
                    if path.exists():validate_cached(path,config,row,label,regime);continue
                    atomic(root/'status.json',dict(stage='evaluation',pid=os.getpid(),regime=regime,condition=label,
                        id=row['id'],sample_index=i,updated=time.time()))
                    try:
                        record,routing=generate(adapter,row,label,regime,config,projections)
                        save(root,record,routing)
                        print(json.dumps(dict(event='complete',regime=regime,condition=label,id=row['id'],
                            steps=record['steps'],score=record['score'],seconds=record['wall_seconds'])),flush=True)
                    except Exception as exc:
                        failure=dict(regime=regime,condition=label,id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                        failures.append(failure);atomic(root/'failures.json',failures);print(json.dumps(failure),flush=True)
                        if 'illegal memory' in str(exc) or 'device-side assert' in str(exc):raise
                        torch.cuda.empty_cache()
            from .sparsity_steps_report import report
            report(root)
        atomic(root/'projection_matrices.json',projections.manifest)
        atomic(root/'status.json',dict(stage='complete' if not failures else 'incomplete',errors=len(failures),updated=time.time()))


def launch(root):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');log=root/f'worker_{stamp}.log'
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.sparsity_steps','run','--root',str(root)]
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    with log.open('xb') as f:
        child=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=child.pid,log=str(log),command=cmd,started=stamp)
    atomic(root/f'launch_{stamp}.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','launch','report'));p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    if args.command=='report':
        from .sparsity_steps_report import report
        report(args.root)
    elif args.command=='launch':launch(args.root)
    else:run(args.root)
