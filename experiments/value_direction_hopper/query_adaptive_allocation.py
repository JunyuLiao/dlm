"""Fresh-prompt T versus shuffled-T/uniform-T RULER4K allocation study.

The decoder, Gaussian-32 kernel, temporal history, and physical tile rule are
imported unchanged. Only the existing State allocation mode varies.
"""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments import diffusion_gemma_ruler4k_gaussian_sweep as pool_source
from .experiment import atomic, fingerprint, sha
from .query_adaptive_study import aggregate, cached, generate


ROOT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1'
OLD=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3'
TARGETS=(50,70)
METHODS=('unweighted','T','T_shuffle','T_uniform')
TASKS=tuple(pool_source.base.official.PAPER_TASKS)
KERNEL=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v1'/'build'/'value_direction_44d4a0ba1f697d4a.so'
BRIDGE=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v1'/'build'/'torch_c27183cf1226bde0'/'value_direction_torch_c27183cf1226bde0.so'


def label(name,target):return name if target is None else f'{name}_s{target}'


def specs():
    return {name:dict(name=name,method='T' if name.startswith('T') else name,
        allocation='shuffle' if name.endswith('shuffle') else 'uniform' if name.endswith('uniform') else 'normal',
        bootstrap=False) for name in ('native_dense','kernel_dense',*METHODS)}


def conditions():
    return [('native_dense',None),('kernel_dense',None)]+[(name,target) for target in TARGETS for name in METHODS]


def _selected_pool():
    rows=pool_source._pool_rows()
    chosen={k:[] for k in ('calibration','final','development')}
    for task in TASKS:
        task_rows=[r for r in rows if r['task']==task]
        task_rows.sort(key=lambda r:pool_source._sha(f'42|{task}|{r["_pool_index"]}|{pool_source._sha(r["input"])}'))
        # Rank-sweep v16 used the first 13. Leave those entirely untouched.
        for split,subset in (('calibration',task_rows[13:15]),
                             ('final',task_rows[15:25]),
                             ('development',task_rows[25:26])):
            chosen[split].extend((dict(r,split=split) for r in subset))
    return chosen


def _row(raw,split,tokenizer):
    prompt=raw['input'];tokens=tokenizer.encode_prompt(prompt,{'thinking':False})
    source_id=f'ruler_4096_{raw["task"]}_p{int(raw["_pool_index"]):04d}'
    return dict(id=f'ruler4k/{source_id}',source_id=source_id,benchmark='ruler4k',
        task=raw['task'],task_base=pool_source._task_base(raw['task']),outputs=raw['outputs'],
        official_index=int(raw['index']),generator_seed=int(raw['index']),split=split,
        calibration=split=='calibration',prompt=prompt,prompt_hash=pool_source._sha(prompt),
        prompt_tokens=tokens,prompt_token_count=len(tokens),
        generation_budget=int({'vt':30,'cwe':120,'fwe':50,'qa_1':32,'qa_2':32}.get(raw['task'],128)),seed=42)


def _audit_manifests(manifests):
    previous=json.loads((OLD/'configs/final_manifest.json').read_text())
    previous+=json.loads((OLD/'configs/calibration_manifest.json').read_text())
    previous+=json.loads((Path(__file__).resolve().parents[2]/'results'/'diffusion_gemma_ruler4k_gaussian_rank_sweep_v16'/'development_manifest.json').read_text())
    existing={key:{r[key] for r in previous} for key in ('id','source_id','prompt_hash')}
    for split,n in (('calibration',2),('final',10),('development',1)):
        rows=manifests[split]
        if Counter(r['task'] for r in rows)!=dict.fromkeys(TASKS,n):raise ValueError(f'Wrong {split} task quotas')
        for key in existing:
            vals=[r[key] for r in rows]
            if len(set(vals))!=len(vals) or set(vals)&existing[key]:raise ValueError(f'Duplicate or previously used {key}')
            existing[key].update(vals)
        for r in rows:
            if (r['split']!=split or r['prompt_hash']!=pool_source._sha(r['prompt']) or
                r['prompt_token_count']!=len(r['prompt_tokens']) or r['seed']!=42):
                raise ValueError('Manifest metadata mismatch')


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=json.loads((OLD/'configs/configuration.json').read_text())
    if not KERNEL.exists() or not BRIDGE.exists():raise FileNotFoundError('The tested v4 H100 binary/bridge is missing')
    files=[Path(__file__),Path(__file__).with_name('query_adaptive.py'),
        Path(__file__).with_name('query_adaptive_study.py'),Path(__file__).with_name('integration.py'),
        Path(__file__).with_name('cuda.py'),KERNEL,BRIDGE,
        Path('src/dllm/models/adapters/diffusion_gemma.py'),
        Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
        Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py'),
        Path('experiments/diffusion_gemma_ruler4k_gaussian_sweep.py'),
        pool_source.POOL/'manifest.json',pool_source.POOL/'samples.jsonl',
        OLD/'configs/configuration.json',OLD/'configs/frozen_policies.json',
        OLD/'configs/final_manifest.json',OLD/'configs/calibration_manifest.json']
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as generation
    files.append(Path(generation.__file__))
    source_hashes={str(p.resolve()):sha(p) for p in files}
    path=root/'configs/configuration.json'
    if path.exists():
        cfg=json.loads(path.read_text())
        if source_hashes!=cfg['source_hashes']:raise ValueError('Frozen source changed; create a new versioned root')
        manifests={split:json.loads((root/f'configs/{split}_manifest.json').read_text())
            for split in ('calibration','final','development')}
        _audit_manifests(manifests)
        return cfg,manifests
    tokenizer=create_adapter('diffusion_gemma',base['model'],device='cpu',precision='float32',
        revision=base['revision']).load_tokenizer()
    selected=_selected_pool()
    manifests={split:[_row(raw,split,tokenizer) for raw in raw_rows]
        for split,raw_rows in selected.items()}
    _audit_manifests(manifests)
    old_frozen=json.loads((OLD/'configs/frozen_policies.json').read_text())
    cfg=dict(schema='query_adaptive_temporal_allocation_v1',model=base['model'],revision=base['revision'],
        library=str(KERNEL.resolve()),torch_library=str(BRIDGE.resolve()),
        source_hashes=source_hashes,source_pool=str(pool_source.POOL.resolve()),
        source_rank_sweep='v16 SHA256 task-local order, ranks 13:15 calibration, 15:25 final, 25:26 development',
        manifest_exposure='Fresh relative to prior v16 split and prior 130+26 query-adaptive cohort; same pre-existing official RULER4K pool',
        methods=specs(),conditions=conditions(),base_policies=base['base_policies'],
        m_ref=old_frozen['m_ref'],seed=42,confirmation_seeds=[43,44],projection_seed=1729,rank=32,
        beta=3.,gamma=.5,physical_tile=[128,64],canvas=256,max_steps=48,
        temperature=base['temperature'],calibration_tolerance=.0125,
        comparison_tolerance=.02,
        calibration='26 fresh task-balanced prompts, full generation per point, at most 10 points; T nominal target then controls matched to T calibration G/L/whole',
        sparsity='pooled skipped eligible physical tiles / pooled eligible over complete decoder trajectory, prefix excluded',
        backend_caveat='Native dense SDPA differs numerically and in local-mask convention from matched v4 kernel')
    cfg['fingerprint']=fingerprint(cfg)
    atomic(path,cfg)
    for split,rows in manifests.items():atomic(root/f'configs/{split}_manifest.json',rows)
    atomic(root/'configs/dataset_audit.json',dict(passed=True,
        counts={key:len(rows) for key,rows in manifests.items()},
        previous_manifest_overlap=False,source_hashes=source_hashes))
    return cfg,manifests


def _policy(logs):return {kind:dict(log_threshold=float(logs[kind])) for kind in ('local','global')}


def _goal_distance(actual,goal):
    return max(abs(actual[k]-goal[k]) for k in ('whole','local','global'))


def calibrate(adapter,root,cfg,rows,name,target,goal,projections):
    token=label(name,target);dest=root/'configs/thresholds'/f'{token}.json'
    if dest.exists():
        frozen=json.loads(dest.read_text())
        if frozen['fingerprint']!=cfg['fingerprint'] or frozen['calibration_ids']!=[r['id'] for r in rows]:
            raise ValueError('Frozen threshold provenance changed')
        return frozen
    base=cfg['base_policies'][str(target)] if str(target) in cfg['base_policies'] else cfg['base_policies'][target]
    logs={kind:float(base[kind]['log_threshold']+(.45 if name!='unweighted' else 0.)) for kind in ('local','global')}
    history=[]
    for round_index in range(10):
        policy=_policy(logs);key=fingerprint([name,target,policy])[:16]
        results=[cached(adapter,root,f'calibration/{token}/{key}',name,target,row,cfg['methods'][name],
            policy,cfg,projections,cfg['m_ref']) for row in rows]
        actual,counts=aggregate(results)
        point=dict(round=round_index,policy=policy,actual=actual,counts=counts,
            total_steps=sum(r['steps'] for r in results))
        history.append(point);atomic(root/'calibration_traces'/f'{token}.json',history)
        if _goal_distance(actual,goal)<=cfg['calibration_tolerance']:break
        next_logs={}
        for kind in ('local','global'):
            observations=[(p['actual'][kind],p['policy'][kind]['log_threshold']) for p in history]
            below=[x for x in observations if x[0]<goal[kind]]
            above=[x for x in observations if x[0]>goal[kind]]
            if below and above:
                low=max(below,key=lambda x:x[0]);high=min(above,key=lambda x:x[0])
                ratio=(goal[kind]-low[0])/max(high[0]-low[0],1.e-8)
                candidate=low[1]+ratio*(high[1]-low[1])
            else:
                candidate=logs[kind]+float(max(-.5,min(.5,5*(goal[kind]-actual[kind]))))
            candidate=float(max(logs[kind]-.6,min(logs[kind]+.6,candidate)))
            if abs(candidate-logs[kind])<1.e-5:
                candidate=logs[kind]+(.05 if actual[kind]<goal[kind] else -.05)
            next_logs[kind]=candidate
        logs=next_logs
    best=min(history,key=lambda p:(_goal_distance(p['actual'],goal),
        sum(abs(p['actual'][k]-goal[k]) for k in ('whole','local','global'))))
    frozen=dict(fingerprint=cfg['fingerprint'],condition=token,method=name,target=target,goal=goal,
        policy=best['policy'],calibration_actual=best['actual'],
        attained=_goal_distance(best['actual'],goal)<=cfg['calibration_tolerance'],
        calibration_ids=[r['id'] for r in rows],trace=history,
        selection='Best of at most 10 full-trajectory points by max G/L/whole deviation; no final labels or sparsity used')
    atomic(dest,frozen);return frozen


def smoke(adapter,root,cfg,rows,projections):
    path=root/'smoke.json'
    if path.exists():
        result=json.loads(path.read_text())
        if not result['passed'] or result['fingerprint']!=cfg['fingerprint']:raise ValueError('Invalid smoke')
        return
    checks=[];base=cfg['base_policies']['50']
    for row in rows[:2]:
        for name in ('native_dense','kernel_dense','unweighted','T','T_shuffle','T_uniform'):
            policy=(None if name=='native_dense' else
                {kind:dict(log_threshold=-math.inf) for kind in ('local','global')} if name=='kernel_dense' else base)
            first,_=generate(adapter,row,cfg['methods'][name],policy,cfg,projections,cfg['m_ref'])
            second,_=generate(adapter,row,cfg['methods'][name],policy,cfg,projections,cfg['m_ref'],diagnostics=False)
            if first['completion_tokens']!=second['completion_tokens'] or first['steps']!=second['steps']:
                raise AssertionError(f'Instrumentation changed {name} {row["id"]}')
            checks.append(dict(id=row['id'],condition=name,steps=first['steps'],instrumentation_parity=True))
        # With beta zero, T's first and later row weights are exactly one.
        variant=deepcopy(cfg);variant['beta']=0.
        for name in ('T','T_shuffle','T_uniform'):
            weighted,_=generate(adapter,row,cfg['methods'][name],base,variant,projections,cfg['m_ref'])
            plain,_=generate(adapter,row,cfg['methods']['unweighted'],base,cfg,projections,cfg['m_ref'])
            if weighted['completion_tokens']!=plain['completion_tokens'] or weighted['counts']!=plain['counts']:
                raise AssertionError(f'Unit row weights differ from unweighted {name}')
        checks.append(dict(id=row['id'],condition='unit_weight_allocation',exact_parity=True))
    atomic(path,dict(passed=True,fingerprint=cfg['fingerprint'],checks=checks))


def run(root,stage='run'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests=prepare(root)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        atomic(root/'status.json',dict(stage='loading',pid=os.getpid(),updated=time.time()))
        adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
        projections=Projections()
        smoke(adapter,root,cfg,manifests['development'],projections)
        if stage=='smoke':return
        frozen={};cal=manifests['calibration']
        for target in TARGETS:
            nominal={kind:target/100 for kind in ('whole','local','global')}
            t=calibrate(adapter,root,cfg,cal,'T',target,nominal,projections)
            frozen[label('T',target)]=t['policy']
            for name in ('unweighted','T_shuffle','T_uniform'):
                result=calibrate(adapter,root,cfg,cal,name,target,t['calibration_actual'],projections)
                frozen[label(name,target)]=result['policy']
        atomic(root/'configs/frozen_policies.json',dict(policies=frozen,m_ref=cfg['m_ref'],beta=cfg['beta'],
            gamma=cfg['gamma'],calibration_only=True))
        if stage=='calibrate':return
        failures=[]
        for name,target in conditions():
            policy=(None if name=='native_dense' else
                {kind:dict(log_threshold=-math.inf) for kind in ('local','global')} if name=='kernel_dense' else
                frozen[label(name,target)])
            for row in manifests['final']:
                try:cached(adapter,root,'final',name,target,row,cfg['methods'][name],policy,cfg,projections,cfg['m_ref'])
                except Exception as error:
                    record=dict(condition=label(name,target),id=row['id'],error=repr(error),traceback=traceback.format_exc())
                    failures.append(record);atomic(root/'failures.json',failures)
                    if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                    torch.cuda.empty_cache()
        atomic(root/'configs/projection_matrices.json',projections.manifest)
        from .query_adaptive_allocation_report import report
        report(root)
        atomic(root/'status.json',dict(stage='complete' if not failures else 'incomplete',
            errors=len(failures),pid=os.getpid(),updated=time.time()))


def confirm(root):
    root=Path(root);cfg,manifests=prepare(root)
    frozen=json.loads((root/'configs/frozen_policies.json').read_text())['policies']
    if not json.loads((root/'audit.json').read_text())['complete']:raise ValueError('Final audit incomplete')
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
        projections=Projections();failures=[]
        for seed in cfg['confirmation_seeds']:
            for target in TARGETS:
                for name in ('T','T_shuffle','T_uniform'):
                    policy=frozen[label(name,target)]
                    for original in manifests['final']:
                        row=dict(original,seed=seed)
                        try:cached(adapter,root,f'confirmation/seed{seed}',name,target,row,cfg['methods'][name],
                            policy,cfg,projections,cfg['m_ref'])
                        except Exception as error:
                            record=dict(seed=seed,condition=label(name,target),id=row['id'],
                                error=repr(error),traceback=traceback.format_exc())
                            failures.append(record);atomic(root/'confirmation_failures.json',failures)
                            if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                            torch.cuda.empty_cache()
        from .query_adaptive_allocation_report import report
        report(root)
        atomic(root/'confirmation_status.json',dict(stage='complete' if not failures else 'incomplete',
            errors=len(failures),pid=os.getpid(),updated=time.time()))


def launch(root,stage):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','experiments.value_direction_hopper.query_adaptive_allocation',stage,
        '--root',str(root)]
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    logfile=root/f'{stage}_{int(time.time())}.log'
    with logfile.open('xb') as out:
        child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(root/f'{stage}_launch.json',dict(pid=child.pid,log=str(logfile),command=command,started=time.time()))
    print(json.dumps(dict(pid=child.pid,log=str(logfile))),flush=True)


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
        from .query_adaptive_allocation_report import report
        report(args.root)
    else:run(args.root,args.stage)
