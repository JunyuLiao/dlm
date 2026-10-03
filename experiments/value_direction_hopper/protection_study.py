"""Calibrate protection refinements without using final benchmark outcomes."""
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
import traceback
import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic,sha,fingerprint,shard_path
from .integration import install
from .sparsity_steps import routing_counts
from .protection_refinement import State,control,Rescue,variants

BASE=Path(__file__).resolve().parents[2]/'results'/'value_direction_sparsity_steps_v2'
OLD=Path(__file__).resolve().parents[2]/'results'/'value_direction_interventions_v2'
ROOT=Path(__file__).resolve().parents[2]/'results'/'protection_refinement_v1'


def setup(root):
    cfg0=json.loads((BASE/'configuration.json').read_text());frozen=json.loads((OLD/'frozen_policies.json').read_text())
    rows=json.loads((BASE/'manifest.json').read_text());cal=json.loads((BASE/'calibration_manifest.json').read_text())
    dev=json.loads((OLD/'development_manifest.json').read_text())
    for key in ('id','prompt_hash'):
        a,b,c=({r[key] for r in group} for group in (rows,cal,dev))
        if a&b or a&c or b&c:raise ValueError('Split overlap')
    policies={f'protect_s{t}':frozen[f'protect_s{t}']['thresholds'] for t in (60,65)}
    policies.update({f'static_s{t}':cfg0['policies'][f'gaussian32_s{t}'] for t in (60,65)})
    paths=[Path(__file__),Path(__file__).with_name('protection_refinement.py'),Path(__file__).with_name('integration.py'),
           Path(__file__).with_name('cuda.py'),Path(__file__).with_name('intervention_policies.py'),
           Path(cfg0['library']),Path(cfg0['torch_library']),BASE/'configuration.json',OLD/'frozen_policies.json',BASE/'manifest.json',BASE/'calibration_manifest.json',OLD/'development_manifest.json',
           Path('src/dllm/models/adapters/diffusion_gemma.py'),Path('experiments/diffusion_gemma_jl_output_aware/projections.py')]
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as generation
    paths.append(Path(generation.__file__))
    cfg=dict(schema='protection_refinement_v1',base=cfg0,policies=policies,variants=variants(),
        source_hashes={str(p.resolve()):sha(p) for p in paths},ids=[r['id'] for r in rows],calibration_ids=[r['id'] for r in cal],development_ids=[r['id'] for r in dev],
        targets=[.6,.65],screen='Fixed old protection thresholds on26 calibration samples; all6 non-rescue variants and2 rescue variants',
        selection='Select2 non-rescue refinements by calibration Pareto, accuracy>=dense-2pp preferred, then minimum steps. Retain one rescue variant independently. Form one combination with fastest refinement.',
        calibration='Only offsets[0,-.25,-.5] to old protection thresholds; never increase skipping to compensate for rescue; select nearest measured output sparsity with2pp tolerance; misses explicit',
        rescue='Reference two full passes, union of complete128-query blocks touched by up to8 difficult queries from previous logits in steps2-5. Output-mask sparsity and actually executed two-pass sparsity/work are reported separately.',
        timing='Two interleaved repeats130 prompts for2 development-selected refinements plus6 controls; separate untraced E2E and CUDA-event decoder/policy timing. All signal computation remains.',
        stop='Unchanged native48-step maximum, all-query stability and entropy; all queries recomputed; input-only protection',
        margin='Old p>=.999 implies margin>=.998, so oldmargin.98 is redundant. Strictmargin.999 is a genuine constraint; confidence-only is an equivalence ablation.',
        exposure='Repeatedly examined130 questions are exploratory. Strict success criterion actual target within2pp, mean<5, accuracy>=.89; report uncertainties, never relax after results.')
    cfg['fingerprint']=fingerprint(cfg)
    if (root/'configuration.json').exists():
        if json.loads((root/'configuration.json').read_text())!=cfg:raise ValueError('Frozen study changed')
    else:
        atomic(root/'configuration.json',cfg)
        for name,data in [('manifest',rows),('calibration_manifest',cal),('development_manifest',dev)]:atomic(root/(name+'.json'),data)
        from .provenance import snapshot
        snapshot(root)
    return cfg,rows,cal,dev


def generate(adapter,row,spec,thresholds,cfg,projections,diagnostics=True,profile=False):
    spec=dict(spec,profile=profile);state=State(spec,thresholds,diagnostics)
    def draft(tokens):
        ids,_=adapter._completion(tokens,0,row['generation_budget']);return dict(draft_tokens=ids,draft_score=score(row,adapter.tokenizer.decode(ids,skip_special_tokens=True)))
    ctx=nullcontext((None,None)) if spec['mode']=='native' else install(adapter,cfg['base']['library'],thresholds,mode='value',projections=projections,torch_library=cfg['base']['torch_library'],collect=diagnostics)
    with ctx as (binding,router):
        if binding:
            _set_context(binding,row)
            if spec['mode']=='dense':router.policy_selector=lambda iteration,kind:None
            if spec.get('rescue'):binding.runtime.attention_override=Rescue(router,state)
        if spec['mode'] in ('native','dense','static') and not diagnostics and not profile:
            observer=nullcontext()
        else:observer=control(adapter.model,state,draft if diagnostics else None)
        with observer:
            torch.cuda.synchronize();start=time.perf_counter();out=adapter.generate(_request(row));torch.cuda.synchronize();seconds=time.perf_counter()-start
        routing=[] if router is None or not diagnostics else router.records()
    steps=out.metadata['actual_denoising_step_count']
    if state.step and steps!=state.step:raise ValueError('Wrong call count')
    if diagnostics and len(state.records)!=steps:raise ValueError('Missing step diagnostics')
    if diagnostics and len({r['input_length'] for r in state.records})!=1:raise ValueError('Unexpected multiple canvases')
    counts=routing_counts(routing);executed=deepcopy(counts)
    for w in state.work:
        for k in ('whole',w['attention_type']):
            executed[k]['eligible']+=w['safer_eligible']
            executed[k]['skipped']+=w['base_skipped']+w['safer_skipped']-w['output_skipped']
    return dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
        prediction=out.text,completion_tokens=out.completion_tokens,score=score(row,out.text),metadata=out.metadata,
        steps=steps,seconds=seconds,trajectory=state.records,counts=counts,executed_counts=executed,rescue_work=state.work,
        spec=spec,thresholds=thresholds,decoder_ms=sum(a.elapsed_time(b) for a,b in state.decoder_events) if profile else None,
        protection_ms=sum(a.elapsed_time(b) for a,b in state.policy_events) if profile else None),routing


def cached(adapter,root,stage,label,row,spec,thresholds,cfg,projections):
    path=shard_path(root/stage,label,row);identity=fingerprint([cfg['fingerprint'],spec,thresholds,row['id'],row['prompt_hash'],row['seed']])
    if path.exists():
        r=json.loads(path.read_text())
        if r['identity']!=identity or sha(r['routing_path'])!=r['routing_sha256']:raise ValueError('Invalid resume')
        return r
    atomic(root/'status.json',dict(stage=stage,condition=label,id=row['id'],pid=os.getpid(),updated=time.time()))
    r,routing=generate(adapter,row,spec,thresholds,cfg,projections)
    path.parent.mkdir(parents=True,exist_ok=True);raw=path.with_suffix('.routing.json.gz');tmp=raw.with_suffix('.tmp')
    with gzip.open(tmp,'wt',compresslevel=3) as f:json.dump(routing,f)
    os.replace(tmp,raw);r.update(identity=identity,routing_path=str(raw),routing_sha256=sha(raw),condition=label,stage=stage)
    atomic(path,r);print(json.dumps(dict(stage=stage,condition=label,id=row['id'],steps=r['steps'],score=r['score'])),flush=True)
    return r


def summary(rows):
    group=defaultdict(list)
    for r in rows:group[r['task']].append(r['score'])
    return dict(n=len(rows),accuracy=float(np.mean([np.mean(v) for v in group.values()])),mean_steps=float(np.mean([r['steps'] for r in rows])),
        output_sparsity=sum(r['counts']['whole']['skipped'] for r in rows)/max(1,sum(r['counts']['whole']['eligible'] for r in rows)),
        executed_sparsity=sum(r['executed_counts']['whole']['skipped'] for r in rows)/max(1,sum(r['executed_counts']['whole']['eligible'] for r in rows)))


def choose(screen,dense,allowed,n):
    items=[(name,v) for name,v in screen.items() if name in allowed]
    good=[(name,v) for name,v in items if v['accuracy']>=max(.89,dense['accuracy']-.02)]
    ranked=good or items
    return [name for name,_ in sorted(ranked,key=lambda item:(item[1]['mean_steps'],-item[1]['accuracy']))[:n]]


def run(root,smoke_only=False):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,rows,cal,dev=setup(root);b=cfg['base'];projections=Projections()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=create_adapter('diffusion_gemma',b['model'],device='cuda',precision='bfloat16',revision=b['revision']).load()
        checks=[]
        for row in [rows[0],next(r for r in rows if r['task']=='fwe')]:
            for name in ['existing','confidence_only','hysteresis','immediate','rescue_conservative','rescue_dense']:
                spec=cfg['variants'][name];th=cfg['policies']['protect_s60']
                r=cached(adapter,root,'smoke',name,row,spec,th,cfg,projections)
                quiet,_=generate(adapter,row,spec,th,cfg,projections,False)
                if r['completion_tokens']!=quiet['completion_tokens'] or r['steps']!=quiet['steps']:raise ValueError('Instrumentation changes outputs')
                if name=='existing':
                    old=json.loads(shard_path(OLD/'final','protect_s60',row).read_text())
                    if old['completion_tokens']!=r['completion_tokens'] or old['steps']!=r['steps']:raise ValueError('Existing protection not reproduced')
                checks.append(dict(id=row['id'],variant=name,passed=True))
        atomic(root/'smoke.json',dict(passed=True,checks=checks))
        if smoke_only:return
        # Diagnose all existing final protection trajectories without using
        # their scores for variant selection; verify exact cached reproduction.
        for target in (60,65):
            for row in rows:
                label=f'protect_s{target}';r=cached(adapter,root,'controls',label,row,cfg['variants']['existing'],cfg['policies'][label],cfg,projections)
                old=json.loads(shard_path(OLD/'final',label,row).read_text())
                if r['completion_tokens']!=old['completion_tokens'] or r['steps']!=old['steps']:raise ValueError('Control reproduction failure')
        screens={};selected={};policies={}
        for target in (60,65):
            th=cfg['policies'][f'protect_s{target}'];screen={}
            dense=summary([cached(adapter,root,'pilot','dense',r,dict(mode='static'),{k:dict(log_threshold=-100.) for k in th},cfg,projections) for r in cal])
            for name,spec in cfg['variants'].items():
                label=f'{name}_s{target}'
                screen[name]=summary([cached(adapter,root,'fixed_threshold',label,r,spec,th,cfg,projections) for r in cal])
            screens[str(target)]=screen
            picks=choose(screen,dense,['hysteresis','confidence_only','strict_margin','immediate','very_confident'],2)
            rescue=choose(screen,dense,['rescue_conservative','rescue_dense'],1)[0]
            designs={name:cfg['variants'][name] for name in picks+[rescue]}
            designs['combined']=dict(cfg['variants'][picks[0]],rescue=cfg['variants'][rescue]['rescue'])
            selected[str(target)]=dict(picks=picks,rescue=rescue,combination=designs['combined'],selection_split='calibration',dense=dense)
            for name,spec in designs.items():
                label=f'{name}_s{target}';trace=[]
                for offset in (0.,-.25,-.5):
                    thresholds={k:dict(log_threshold=v['log_threshold']+offset) for k,v in th.items()}
                    token=fingerprint([spec,thresholds])[:16]
                    measured=summary([cached(adapter,root,'calibration',token,r,spec,thresholds,cfg,projections) for r in cal])
                    trace.append(dict(offset=offset,thresholds=thresholds,summary=measured))
                # Executed two-pass sparsity is authoritative for rescue;
                # also retain output mask sparsity so dilution is visible.
                best=min(trace,key=lambda x:abs(x['summary']['executed_sparsity']-target/100))
                policies[label]=dict(spec=spec,thresholds=best['thresholds'],target=target/100,trace=trace,selected=best,
                    attained=abs(best['summary']['executed_sparsity']-target/100)<=.02,selection_split='calibration')
                atomic(root/'policies'/(label+'.json'),policies[label])
        atomic(root/'fixed_threshold_summary.json',screens);atomic(root/'selection.json',selected);atomic(root/'frozen_policies.json',policies)
        development={}
        for label,p in policies.items():development[label]=summary([cached(adapter,root,'development',label,r,p['spec'],p['thresholds'],cfg,projections) for r in dev])
        # Development timing choices cannot exploit rescue's discarded pass.
        timed=choose(development,dict(accuracy=.90),[name for name in policies if not policies[name]['spec'].get('rescue')],2)
        atomic(root/'timing_selection.json',dict(selected=timed,development=development,selection_split='development'))
        for label,p in policies.items():
            for row in rows:cached(adapter,root,'final',label,row,p['spec'],p['thresholds'],cfg,projections)
        timing_specs={'native_dense':(dict(mode='native'),cfg['policies']['static_s60']),
            'kernel_dense':(dict(mode='dense'),cfg['policies']['static_s60'])}
        for target in (60,65):
            timing_specs[f'static_s{target}']=(dict(mode='static'),cfg['policies'][f'static_s{target}'])
            timing_specs[f'protect_s{target}']=(cfg['variants']['existing'],cfg['policies'][f'protect_s{target}'])
        timing_specs.update({label:(policies[label]['spec'],policies[label]['thresholds']) for label in timed})
        # Two complete interleaved repeats. E2E untraced and event-profiled
        # decoder/policy runs are distinct, both with output parity verification.
        for label,(spec,th) in timing_specs.items():generate(adapter,rows[0],spec,th,cfg,projections,False)
        labels=list(timing_specs)
        for repeat in range(2):
            for i,row in enumerate(rows):
                order=labels[(i+repeat)%len(labels):]+labels[:(i+repeat)%len(labels)]
                for label in order:
                    spec,th=timing_specs[label]
                    for mode in ('e2e','profile'):
                        dest=shard_path(root/f'timing/{mode}/{repeat}',label,row)
                        if dest.exists():continue
                        atomic(root/'status.json',dict(stage='timing',condition=label,id=row['id'],mode=mode,repeat=repeat,pid=os.getpid(),updated=time.time()))
                        result,_=generate(adapter,row,spec,th,cfg,projections,False,mode=='profile');atomic(dest,result)
        from .protection_study_report import report
        report(root);atomic(root/'status.json',dict(stage='complete',updated=time.time()))


def launch(root,smoke_only):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.protection_study','smoke' if smoke_only else 'run','--root',str(root)]
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    log=root/f'worker_{int(time.time())}.log'
    with log.open('xb') as f:child=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(root/'launch.json',dict(pid=child.pid,command=cmd,log=str(log)));print(child.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','smoke','launch','launch-smoke','report'));p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if a.command.startswith('launch'):launch(a.root,a.command=='launch-smoke')
    elif a.command=='report':
        from .protection_study_report import report
        report(a.root)
    else:run(a.root,a.command=='smoke')
