"""Resumable diagnosis using the frozen Hopper runner's model, policies and data."""
import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import fcntl
import gzip
import json
import os
from pathlib import Path
import time
import traceback

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, sha, fingerprint, shard_path
from .integration import install
from .trajectory import observe, AttentionProbe

BASE = Path(__file__).resolve().parents[2]/'results'/'value_direction_hopper_v1'/'final_ruler4k130_v1'
METHODS = ('native_dense', 'kernel_dense', 'blasst_capped', 'kernel_gaussian32', 'kernel_blasst')
FIXED = ('kernel_dense', 'kernel_gaussian32', 'kernel_blasst')


def contract(root):
    original = json.loads((BASE/'configuration.json').read_text())
    rows = json.loads((BASE/'manifest.json').read_text())
    source_files = [Path(__file__), Path(__file__).with_name('trajectory.py'),
                    Path(__file__).with_name('integration.py'), Path(__file__).with_name('cuda.py')]
    config = dict(schema='trajectory_diagnosis_v1', original=original,
        methods=METHODS, fixed_methods=FIXED, sample_ids=[r['id'] for r in rows],
        sources={str(p.resolve()):sha(p) for p in source_files},
        native_generation_source=sha(Path(__import__('transformers.models.diffusion_gemma.generation_diffusion_gemma',
            fromlist=['x']).__file__)),
        instrumentation='per-position FP32 raw/processed confidence and native-dtype entropy; reversible acceptance; stopping history',
        fixed='512 native scheduled iterations, suppress stopper return only; all prompts have one canvas',
        attention_sampling='layers0,5,29; heads0,8; queries0,31,63,127,191,255; every step; shared QKV historical mask',
        latency='instrumented; use original uninstrumented run for end-to-end comparison',
        exposure='Previously evaluated samples; diagnostic, not fresh held-out confirmation')
    config['fingerprint'] = fingerprint(config)
    if (root/'configuration.json').exists():
        saved = json.loads((root/'configuration.json').read_text())
        if saved != json.loads(json.dumps(config)):
            raise ValueError('Diagnostic sources/configuration changed; use a fresh output directory')
    else:
        atomic(root/'configuration.json', config)
        atomic(root/'manifest.json', rows)
        for p in source_files:
            dest = root/'sources'/p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    return config, rows


def generate(adapter, row, method, config, projections, *, instrument=True, fixed=False):
    c = config['original']
    if method == 'native_dense':
        binding_context = nullcontext((None,None))
    else:
        value = method in ('kernel_gaussian32','kernel_dense')
        policy = deepcopy(c['policies']['kernel_gaussian32' if value else 'kernel_blasst'])
        if method == 'kernel_dense':
            for p in policy.values():
                p['log_threshold'] = -float('inf')
        if method == 'blasst_capped':
            for p in policy.values():
                p['cap_one'] = True
        binding_context = install(adapter,c['library'],policy,mode='value' if value else 'blasst',
            projections=projections,torch_library=c['torch_library'],collect=True)
    request = _request(row)
    if fixed:
        request = replace(request, steps=512)

    def draft_score(tokens):
        tokens, _ = adapter._completion(torch.tensor(tokens),0,row['generation_budget'])
        return score(row,adapter.tokenizer.decode(tokens,skip_special_tokens=True))

    with binding_context as (binding,router):
        if binding:
            _set_context(binding,row)
        with (observe(adapter.model,fixed_steps=fixed,draft_score=draft_score)
              if instrument else nullcontext([])) as steps:
            probe = None
            if instrument and router:
                probe = AttentionProbe(router,steps)
                binding.runtime.attention_override = probe
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = adapter.generate(request)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
        routing = [] if router is None else router.records()
        if instrument and len(steps) != output.metadata['actual_denoising_step_count']:
            raise AssertionError('Direct decoder calls disagree with returned count')
        if fixed and len(steps) != 512:
            raise AssertionError('Fixed control did not run exactly 512 calls')
        record = dict(status='complete',fingerprint=config['fingerprint'],id=row['id'],task=row['task'],
            method=method,regime='fixed512' if fixed else 'adaptive',prompt_hash=row['prompt_hash'],seed=row['seed'],
            generation_budget=row['generation_budget'],completion_tokens=output.completion_tokens,
            prediction=output.text,metadata=output.metadata,score=score(row,output.text),wall_seconds=seconds)
        return record,dict(steps=steps,attention=[] if probe is None else probe.errors,routing=routing)


def save(root,record,trace):
    dest = shard_path(root/record['regime'],record['method'],record)
    dest.parent.mkdir(parents=True,exist_ok=True)
    raw = dest.with_suffix('.trace.json.gz')
    temp = raw.with_suffix('.tmp')
    with gzip.open(temp,'wt',compresslevel=3) as f:
        json.dump(trace,f,allow_nan=False)
    os.replace(temp,raw)
    record.update(trace_path=str(raw.resolve()),trace_sha256=sha(raw))
    atomic(dest,record)


def smoke(adapter,rows,config,projections,root):
    checks=[]
    # One retrieval and one aggregation task; selection is independent of scores.
    selected=[rows[0],next(r for r in rows if r['task'] == 'fwe')]
    for row in selected:
        for method in METHODS:
            reference,_ = generate(adapter,row,method,config,projections,instrument=False)
            observed,trace = generate(adapter,row,method,config,projections)
            same = reference['completion_tokens'] == observed['completion_tokens']
            same_steps = reference['metadata']['actual_denoising_step_count'] == len(trace['steps'])
            cached = None
            if method in ('native_dense','kernel_gaussian32','kernel_blasst'):
                cached = json.loads(shard_path(BASE,method,row).read_text())
                if cached['completion_tokens'] != observed['completion_tokens'] or cached['metadata']['actual_denoising_step_count'] != len(trace['steps']):
                    raise AssertionError(f'Previous run not reproduced: {method} {row["id"]}')
            check=dict(id=row['id'],method=method,unchanged=same,steps_unchanged=same_steps,
                       steps=len(trace['steps']),cached_parity=cached is not None)
            checks.append(check)
            atomic(root/'smoke.json',dict(passed=all(x['unchanged'] and x['steps_unchanged'] for x in checks),checks=checks))
            if not same or not same_steps:
                raise AssertionError('Instrumentation changed generation')
            save(root,observed,trace)
            print(json.dumps(dict(event='smoke',**check)),flush=True)
    return checks


def run(root, command):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,rows=contract(root)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        c=config['original']
        adapter=create_adapter('diffusion_gemma',c['model'],device='cuda',precision='bfloat16',revision=c['revision']).load()
        projections=Projections()
        if not (root/'smoke.json').exists() or len(json.loads((root/'smoke.json').read_text())['checks'])!=10:
            smoke(adapter,rows,config,projections,root)
        if command=='smoke':
            return
        errors=[]
        for fixed,methods in ((False,METHODS),(True,FIXED)):
            regime='fixed512' if fixed else 'adaptive'
            for i,row in enumerate(rows):
                for method in methods[i%len(methods):]+methods[:i%len(methods)]:
                    dest=shard_path(root/regime,method,row)
                    if dest.exists():
                        old=json.loads(dest.read_text())
                        if old['fingerprint']!=config['fingerprint'] or sha(old['trace_path'])!=old['trace_sha256']:
                            raise ValueError('Invalid resume shard')
                        continue
                    atomic(root/'status.json',dict(status='running',pid=os.getpid(),regime=regime,
                        method=method,id=row['id'],sample_index=i,updated=time.time()))
                    try:
                        record,trace=generate(adapter,row,method,config,projections,fixed=fixed)
                        save(root,record,trace)
                        print(json.dumps(dict(event='complete',regime=regime,method=method,id=row['id'],
                            steps=len(trace['steps']),score=record['score'],seconds=record['wall_seconds'])),flush=True)
                    except Exception as exc:
                        failure=dict(regime=regime,method=method,id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                        errors.append(failure)
                        atomic(root/'failures.json',errors)
                        print(json.dumps(failure),flush=True)
                        if 'illegal memory' in str(exc) or 'device-side assert' in str(exc):
                            raise
                        torch.cuda.empty_cache()
            from .trajectory_report import report
            report(root)
        atomic(root/'status.json',dict(status='complete' if not errors else 'incomplete',errors=len(errors),updated=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('smoke','run','report'))
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.command=='report':
        from .trajectory_report import report
        report(a.root)
    else:
        run(a.root,a.command)
