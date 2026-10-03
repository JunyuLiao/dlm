"""All intervention families: disjoint pilot calibration, frozen final evaluation."""
import argparse
from collections import defaultdict
from copy import deepcopy
from contextlib import nullcontext
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic,sha,fingerprint,shard_path
from .integration import install
from .intervention_policies import Controller,control
from .sparsity_steps import routing_counts

BASE=Path(__file__).resolve().parents[2]/'results'/'value_direction_sparsity_steps_v2'
ROOT=Path(__file__).resolve().parents[2]/'results'/'value_direction_interventions_v2'


def designs():
    result={}
    def add(name,policy,targets):
        for t in targets:result[f'{name}_s{t}']=dict(policy=policy,target=t/100)
    add('dense_first',dict(schedule='first'),(30,40,50,60,70))
    add('dense_late',dict(schedule='late'),(40,50,60,70))
    add('reactive',dict(schedule='reactive'),(60,65,70))
    add('protect',dict(schedule='sparse',acceptance='protect',strong=.999,weak=.98),(60,65,70))
    for strength in (1,3):
        add('accept_x'+str(strength),dict(schedule='sparse',acceptance='adaptive',strength=strength),(60,65,70))
    for mode in ('random','ranked'):
        add('extra_'+mode,dict(schedule='sparse',acceptance=mode,k=4),(60,70))
    return result


def setup(root):
    source=json.loads((BASE/'configuration.json').read_text());rows=json.loads((BASE/'manifest.json').read_text())
    calibration=json.loads((BASE/'calibration_manifest.json').read_text())
    development=json.loads((Path(__file__).resolve().parents[2]/'results'/'diffusion_gemma_ruler4k_value_direction_s70_v19'/'development_manifest.json').read_text())
    for key in ('id','prompt_hash'):
        sets=[{r[key] for r in cohort} for cohort in (rows,calibration,development)]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Split overlap')
    files=[Path(__file__),Path(__file__).with_name('intervention_policies.py'),Path(__file__).with_name('integration.py'),
        Path(__file__).with_name('cuda.py'),Path(source['library']),Path(source['torch_library']),BASE/'configuration.json',BASE/'manifest.json',
        BASE/'calibration_manifest.json',Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
        Path('src/dllm/models/adapters/diffusion_gemma.py')]
    cfg=dict(schema='interventions_v2',base=source,designs=designs(),ids=[r['id'] for r in rows],
        calibration_ids=[r['id'] for r in calibration],development_ids=[r['id'] for r in development],
        source_hashes={str(p.resolve()):sha(p) for p in files},
        calibration='26 disjoint prompts; common offset to frozen s70 local/global logtau; grid[-3,-2,-1,0,1,2,4] plus3 refinement points; overall count weighting;2pp tolerance; nonmonotonic search retained',
        trigger='95th percentile dense pilot churn;5th percentile entropy decline and acceptance progress; one transition; never consecutive reactive dense calls',
        protection='One-transition stability + p>=.999 + margin>=.98; reopen on top1 change or p<.98. Protect sampled input from renoising by using top1; continue all attention computation and native stopping.',
        diagnostic='Random/ranked choose min(4,pool size) among32 lowest-entropy rejected positions with p>=.5; independent RNG. Identical quotas on shared states; divergent rollout candidate counts can differ and are reported.',
        selection='Development only: Pareto candidates; prefer >=dense-2pp and highest actual sparsity with <=6 mean steps. At most2 late/reactive + best acceptance/protection combinations.',
        exposure='Previously examined final130; not fresh held-out evidence; dense matched kernel semantics',
        timing='Final2 pilot-selected configurations and both dense controls; all130,2 repeats,normal decoder with required policy signals, no routing stats or draft scoring')
    cfg['fingerprint']=fingerprint(cfg)
    if (root/'configuration.json').exists():
        if json.loads((root/'configuration.json').read_text())!=cfg:raise ValueError('Frozen study changed')
    else:
        atomic(root/'configuration.json',cfg)
        for name,data in [('manifest',rows),('calibration_manifest',calibration),('development_manifest',development)]:atomic(root/(name+'.json'),data)
        from .provenance import snapshot
        snapshot(root)
    return cfg,rows,calibration,development


def generate(adapter,row,spec,thresholds,cfg,projections,limits,diagnostics=True):
    state=Controller(spec,thresholds,row['seed'],adapter.model.device,limits,diagnostics)
    def draft(tokens):
        ids,_=adapter._completion(tokens,0,row['generation_budget'])
        return score(row,adapter.tokenizer.decode(ids,skip_special_tokens=True))
    with install(adapter,cfg['base']['library'],thresholds,mode='value',projections=projections,
                 torch_library=cfg['base']['torch_library'],collect=diagnostics) as (binding,router):
        router.policy_selector=state.selector;_set_context(binding,row)
        with control(adapter.model,state,draft if diagnostics else None):
            torch.cuda.synchronize();start=time.perf_counter();out=adapter.generate(_request(row));torch.cuda.synchronize();seconds=time.perf_counter()-start
        routing=router.records() if diagnostics else []
    if state.step!=out.metadata['actual_denoising_step_count']:raise ValueError('Direct count mismatch')
    if diagnostics and len({r['input_length'] for r in state.records})!=1:raise ValueError('Expected one canvas')
    counts=routing_counts(routing)
    return dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
        prediction=out.text,completion_tokens=out.completion_tokens,score=score(row,out.text),metadata=out.metadata,
        steps=state.step,counts=counts,trajectory=state.records,seconds=seconds,spec=spec,thresholds=thresholds,
        dense_steps=sum(r['dense'] for r in state.records),instrumented=diagnostics),routing


def cached(adapter,root,stage,label,row,spec,thresholds,cfg,projections,limits):
    path=shard_path(root/stage,label,row);identity=fingerprint([cfg['fingerprint'],spec,thresholds,limits,row['id'],row['seed'],row['prompt_hash']])
    if path.exists():
        r=json.loads(path.read_text())
        if r['identity']!=identity or sha(r['routing_path'])!=r['routing_sha256']:raise ValueError('Resume mismatch')
        return r
    atomic(root/'status.json',dict(stage=stage,condition=label,id=row['id'],pid=os.getpid(),updated=time.time()))
    r,routing=generate(adapter,row,spec,thresholds,cfg,projections,limits)
    path.parent.mkdir(parents=True,exist_ok=True);raw=path.with_suffix('.routing.json.gz');tmp=raw.with_suffix('.tmp')
    with gzip.open(tmp,'wt',compresslevel=3) as f:json.dump(routing,f)
    os.replace(tmp,raw);r.update(identity=identity,routing_path=str(raw.resolve()),routing_sha256=sha(raw),condition=label,stage=stage)
    atomic(path,r)
    print(json.dumps(dict(stage=stage,condition=label,id=row['id'],steps=r['steps'],score=r['score'])),flush=True)
    return r


def summarize(rows):
    counts={k:{n:sum(r['counts'][k][n] for r in rows) for n in ('eligible','skipped')} for k in ('whole','local','global')}
    task=defaultdict(list)
    for r in rows:task[r['task']].append(r['score'])
    return dict(n=len(rows),accuracy=float(np.mean([np.mean(v) for v in task.values()])),
        mean_steps=float(np.mean([r['steps'] for r in rows])),counts=counts,
        sparsity={k:c['skipped']/max(1,c['eligible']) for k,c in counts.items()},
        dense_steps=sum(r['dense_steps'] for r in rows),steps=sum(r['steps'] for r in rows))


def calibrate(adapter,root,label,design,cohort,cfg,projections,limits):
    path=root/'policies'/(label+'.json')
    if path.exists():
        p=json.loads(path.read_text())
        if p['design']!=design or p['fingerprint']!=cfg['fingerprint']:raise ValueError('Policy mismatch')
        return p
    base=cfg['base']['policies']['gaussian32_s70'];trace=[]
    points=[-3.,-2.,-1.,0.,1.,2.,4.]
    for index in range(10):
        if index<len(points):offset=points[index]
        else:
            ordered=sorted(trace,key=lambda x:x['offset']);pairs=[(a,b) for a,b in zip(ordered,ordered[1:])
                if (a['summary']['sparsity']['whole']-design['target'])*(b['summary']['sparsity']['whole']-design['target'])<=0]
            if pairs:
                a,b=min(pairs,key=lambda ab:abs(ab[0]['summary']['sparsity']['whole']-design['target'])+abs(ab[1]['summary']['sparsity']['whole']-design['target']))
                offset=(a['offset']+b['offset'])/2
            else:break
        thresholds={k:dict(log_threshold=v['log_threshold']+offset) for k,v in base.items()}
        pid=fingerprint([design['policy'],thresholds])[:16]
        rows=[cached(adapter,root,'calibration',pid,r,design['policy'],thresholds,cfg,projections,limits) for r in cohort]
        item=dict(offset=offset,thresholds=thresholds,summary=summarize(rows));trace.append(item)
        atomic(root/'calibration_traces'/(label+'.json'),trace)
        # Always measure the full grid: changing iteration counts can make
        # aggregate sparsity nonmonotone, particularly for rescue schedules.
        if index>=6 and any(abs(p['summary']['sparsity']['whole']-design['target'])<=.02 for p in trace):break
    selected=min(trace,key=lambda p:abs(p['summary']['sparsity']['whole']-design['target']))
    result=dict(fingerprint=cfg['fingerprint'],design=design,thresholds=selected['thresholds'],trace=trace,
        calibration_ids=[r['id'] for r in cohort],heldout_used=False,selected=selected,
        attained=abs(selected['summary']['sparsity']['whole']-design['target'])<=.02,
        maximum_observed=max(p['summary']['sparsity']['whole'] for p in trace),
        limitation='Outside tested policy range' if abs(selected['summary']['sparsity']['whole']-design['target'])>.02 else None)
    atomic(path,result);return result


def selection(development,dense):
    eligible=[(label,p) for label,p in development.items() if p['accuracy']>=dense['accuracy']-.02 and p['mean_steps']<=6]
    candidates=eligible or list(development.items())
    return [label for label,_ in sorted(candidates,key=lambda item:(-item[1]['sparsity']['whole'],item[1]['mean_steps'],-item[1]['accuracy']))[:2]]


def run(root,smoke_only=False):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,rows,calibration,development=setup(root)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        b=cfg['base'];adapter=create_adapter('diffusion_gemma',b['model'],device='cuda',precision='bfloat16',revision=b['revision']).load();projections=Projections()
        thresholds=b['policies']['gaussian32_s70'];limits=dict(churn=1.,entropy_floor=.005,progress=0.,acceptance_progress=0.)
        dense=[cached(adapter,root,'pilot','dense',r,dict(schedule='dense'),thresholds,cfg,projections,limits) for r in calibration]
        transitions=[t for r in dense for t in r['trajectory'] if t['churn'] is not None and not t['terminated']]
        limits=dict(churn=float(np.quantile([t['churn'] for t in transitions],.95)),entropy_floor=.005,
                    progress=float(np.quantile([t['entropy_progress'] for t in transitions],.05)),
                    acceptance_progress=float(np.quantile([t['acceptance_progress'] for t in transitions],.05)))
        atomic(root/'trigger_thresholds.json',dict(values=limits,calibration_ids=cfg['calibration_ids'],n_transitions=len(transitions)))
        checks=[]
        for row in calibration[:2]:
            for name,spec in [('plain',dict(schedule='sparse')),('dense',dict(schedule='dense')),
                             ('first',dict(schedule='first')),('late',dict(schedule='late')),
                             ('reactive',dict(schedule='reactive')),('protect',dict(schedule='sparse',acceptance='protect',strong=.999,weak=.98)),
                             ('adaptive',dict(schedule='sparse',acceptance='adaptive',strength=1)),
                             ('random',dict(schedule='sparse',acceptance='random',k=4)),('ranked',dict(schedule='sparse',acceptance='ranked',k=4))]:
                record=cached(adapter,root,'smoke',name,row,spec,thresholds,cfg,projections,limits)
                quiet,_=generate(adapter,row,spec,thresholds,cfg,projections,limits,False)
                if quiet['completion_tokens']!=record['completion_tokens'] or quiet['steps']!=record['steps']:raise ValueError('Diagnostic noninterference failed')
                if name=='first' and not [t['dense'] for t in record['trajectory']]==[True]+[False]*(record['steps']-1):raise ValueError('First schedule')
                if name=='late' and not [t['dense'] for t in record['trajectory']]==[i>=4 for i in range(1,record['steps']+1)]:raise ValueError('Late schedule')
                checks.append(dict(name=name,id=row['id'],passed=True,steps=record['steps']))
        atomic(root/'smoke.json',dict(passed=True,checks=checks))
        if smoke_only:return
        policies={};dev={};errors=[]
        dense_dev=[cached(adapter,root,'development','dense',r,dict(schedule='dense'),thresholds,cfg,projections,limits) for r in development]
        for label,design in cfg['designs'].items():
            try:
                p=calibrate(adapter,root,label,design,calibration,cfg,projections,limits);policies[label]=p
                results=[cached(adapter,root,'development',label,r,design['policy'],p['thresholds'],cfg,projections,limits) for r in development]
                dev[label]=summarize(results)
            except Exception as exc:
                import traceback
                errors.append(dict(condition=label,error=repr(exc),traceback=traceback.format_exc()));atomic(root/'failures.json',errors)
                if 'illegal memory' in str(exc) or 'device-side assert' in str(exc):raise
        # Choose at most two complementary combinations on development data.
        sched=[k for k in selection({k:v for k,v in dev.items() if k.startswith(('dense_late','reactive'))},summarize(dense_dev))]
        accept=selection({k:v for k,v in dev.items() if k.startswith(('protect','accept_x'))},summarize(dense_dev))
        for i,label in enumerate(sched[:2]):
            if not accept:break
            choice=accept[0];design=deepcopy(policies[label]['design']);design['policy'].update({k:v for k,v in policies[choice]['design']['policy'].items() if k!='schedule'})
            name='combo'+str(i+1)+'_s'+str(round(100*design['target']))
            atomic(root/'combinations'/(name+'.json'),dict(schedule_parent=label,acceptance_parent=choice,design=design,selection_split='development'))
            p=calibrate(adapter,root,name,design,calibration,cfg,projections,limits);policies[name]=p
            dev[name]=summarize([cached(adapter,root,'development',name,r,design['policy'],p['thresholds'],cfg,projections,limits) for r in development])
        selected=selection(dev,summarize(dense_dev));atomic(root/'selection.json',dict(development=dev,dense=summarize(dense_dev),timing_selected=selected,selection_split='development'))
        atomic(root/'frozen_policies.json',policies)
        for label,p in policies.items():
            for row in rows:
                try:cached(adapter,root,'final',label,row,p['design']['policy'],p['thresholds'],cfg,projections,limits)
                except Exception as exc:
                    errors.append(dict(stage='final',condition=label,id=row['id'],error=repr(exc)));atomic(root/'failures.json',errors)
                    if 'illegal memory' in str(exc) or 'device-side assert' in str(exc):raise
        # Warm up outside timing. Required decision signals remain; all extra
        # trace/score/statistics collection is disabled.
        for label in ['native_dense','kernel_dense']+selected:
            spec=dict(schedule='dense') if label.endswith('dense') else policies[label]['design']['policy']
            th=thresholds if label.endswith('dense') else policies[label]['thresholds']
            if label!='native_dense':generate(adapter,rows[0],spec,th,cfg,projections,limits,False)
            else:adapter.generate(_request(rows[0]))
            for repeat in range(2):
                for row in rows:
                    path=shard_path(root/f'timing/{repeat}',label,row)
                    if path.exists():continue
                    if label=='native_dense':
                        torch.cuda.synchronize();start=time.perf_counter();out=adapter.generate(_request(row));torch.cuda.synchronize()
                        r=dict(id=row['id'],steps=out.metadata['actual_denoising_step_count'],seconds=time.perf_counter()-start,completion_tokens=out.completion_tokens,score=score(row,out.text))
                    else:r,_=generate(adapter,row,spec,th,cfg,projections,limits,False)
                    atomic(path,r)
        from .intervention_study_report import report
        report(root)
        atomic(root/'status.json',dict(stage='complete' if not errors else 'incomplete',errors=len(errors),updated=time.time()))


def launch(root,smoke_only):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.intervention_study','smoke' if smoke_only else 'run','--root',str(root)]
    log=root/f'worker_{int(time.time())}.log'
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    with log.open('xb') as f:child=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    result=dict(pid=child.pid,command=cmd,log=str(log));atomic(root/'launch.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('smoke','run','launch','launch-smoke','report'));p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if a.command.startswith('launch'):launch(a.root,a.command=='launch-smoke')
    elif a.command=='report':
        from .intervention_study_report import report
        report(a.root)
    else:run(a.root,a.command=='smoke')
