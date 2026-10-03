"""Supplement: suppress stopping while retaining the original48-step schedule.

The deterministic first two examples per task are chosen before this control.
Run after the main GPU worker finishes; never concurrently with it.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import gzip
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, fingerprint, sha, shard_path
from .trajectory import observe
from .trajectory_experiment import generate, save, FIXED
from .report import paired_ci


def run(root,main):
    root.mkdir(parents=True,exist_ok=True)
    with (main/'worker.lock').open('a') as main_lock, (root/'worker.lock').open('a') as lock:
        fcntl.flock(main_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=json.loads((main/'configuration.json').read_text())
        manifest=json.loads((main/'manifest.json').read_text())
        counts=Counter();rows=[]
        for row in manifest:
            if counts[row['task']]<2:
                rows.append(row);counts[row['task']]+=1
        config=dict(parent,original=parent['original'],supplement='fixed48_same_temperature_schedule',
                    selected_ids=[r['id'] for r in rows],parent_fingerprint=parent['fingerprint'],
                    supplemental_source=sha(Path(__file__)),selection='first2 per task in frozen manifest order; no score selection')
        config['fingerprint']=fingerprint(config)
        if (root/'configuration.json').exists():
            if json.loads((root/'configuration.json').read_text())!=config:
                raise ValueError('Cannot resume changed supplemental configuration')
        else:
            atomic(root/'configuration.json',config);atomic(root/'manifest.json',rows)
        c=config['original']
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=create_adapter('diffusion_gemma',c['model'],device='cuda',precision='bfloat16',revision=c['revision']).load()
        projections=Projections();failures=[]
        for row in rows:
            for method in FIXED:
                dest=shard_path(root/'fixed48',method,row)
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['fingerprint']!=config['fingerprint'] or sha(old['trace_path'])!=old['trace_sha256']:
                        raise ValueError('Invalid supplemental resume shard')
                    continue
                def draft_score(tokens):
                    tokens,_=adapter._completion(torch.tensor(tokens),0,row['generation_budget'])
                    return score(row,adapter.tokenizer.decode(tokens,skip_special_tokens=True))
                try:
                    # Reuse the exact generation/binding implementation. Only the
                    # enclosing observer suppresses stopping; request remains48.
                    with observe(adapter.model,fixed_steps=True,draft_score=draft_score) as steps:
                        record,trace=generate(adapter,row,method,config,projections,instrument=False,fixed=False)
                    if len(steps)!=48 or record['metadata']['actual_denoising_step_count']!=48:
                        raise AssertionError('Supplement is not exactly48 steps')
                    record['regime']='fixed48';trace['steps']=steps
                    save(root,record,trace)
                    print(json.dumps(dict(method=method,id=row['id'],score=record['score'],steps=len(steps))),flush=True)
                except Exception as exc:
                    failure=dict(method=method,id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                    failures.append(failure);atomic(root/'failures.json',failures)
                    print(json.dumps(failure),flush=True)
                    if 'illegal memory' in str(exc) or 'device-side assert' in str(exc):raise
        report(root,main)
        atomic(main/'supplement_reference.json',dict(root=str(root.resolve()),
            reason='Same48-step schedule control; first2 per task in frozen order'))


def report(root,main):
    rows=json.loads((root/'manifest.json').read_text());records={};missing=[]
    for regime in ('adaptive','fixed48','fixed512'):
        for method in FIXED:
            for row in rows:
                base=root if regime=='fixed48' else main
                path=shard_path(base/regime,method,row)
                if not path.exists():missing.append(str(path));continue
                value=json.loads(path.read_text())
                if sha(value['trace_path'])!=value['trace_sha256']:raise ValueError('Trace hash mismatch')
                records[regime,method,row['id']]=value
    summary=[];paired=[]
    for regime in ('adaptive','fixed48','fixed512'):
        for method in FIXED:
            selected=[records[regime,method,r['id']] for r in rows if (regime,method,r['id']) in records]
            if not selected:continue
            tasks=defaultdict(list)
            eligible=skipped=0
            for row,value in zip([r for r in rows if (regime,method,r['id']) in records],selected):
                tasks[row['task']].append(value['score'])
                with gzip.open(value['trace_path'],'rt') as f:trace=json.load(f)
                if len(trace['steps'])!=value['metadata']['actual_denoising_step_count']:raise ValueError('Step count mismatch')
                if regime=='fixed48' and len(trace['steps'])!=48:raise ValueError('Fixed48 gate')
                eligible+=sum(r['eligible'] for r in trace['routing']);skipped+=sum(r['skipped'] for r in trace['routing'])
            summary.append(dict(regime=regime,method=method,n=len(selected),accuracy=float(np.mean([np.mean(v) for v in tasks.values()])),
                steps=sum(v['metadata']['actual_denoising_step_count'] for v in selected),sparsity=skipped/max(1,eligible)))
            for baseline in ('kernel_dense',):
                if method==baseline:continue
                pairs=[(r,records[regime,method,r['id']],records[regime,baseline,r['id']]) for r in rows
                       if (regime,method,r['id']) in records and (regime,baseline,r['id']) in records]
                if pairs:paired.append(dict(regime=regime,**paired_ci(pairs,method,baseline)))
    atomic(root/'summary.json',dict(summary=summary,paired=paired,missing=missing,complete=not missing))
    lines=['# Same-schedule fixed48 control','',
        'Deterministic first2 examples per task (26 total), selected by manifest order. Same48-step temperature schedule as the original adaptive run; only adaptive termination is suppressed. '
        'The fixed512 rows reuse the same26 IDs from the main experiment and have a stretched temperature schedule. Previously examined examples, not held-out confirmation.','',
        '|Regime|Method|N|Steps|Physical sparsity|Accuracy|','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['regime']}|{r['method']}|{r['n']}|{r['steps']}|{100*r['sparsity']:.2f}%|{100*r['accuracy']:.2f}%|")
    lines+=['','Paired method-minus-dense accuracy intervals are in summary.json. This small control isolates stopping suppression under the original annealing schedule, '
            'but limited sample size may leave differences inconclusive. Its latency is instrumented and it does not add attention-output probes.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','report'))
    p.add_argument('--root',required=True,type=Path);p.add_argument('--main',required=True,type=Path)
    a=p.parse_args();(run if a.command=='run' else report)(a.root,a.main)
