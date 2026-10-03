"""Interleaved, non-tracing H100 timing of frozen guardrail policies."""
import argparse
from contextlib import nullcontext
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context

from .experiment import atomic, fingerprint, sha, shard_path
from .integration import install
from .query_adaptive import observe
from .query_adaptive_guardrail import PhaseState, _adapter, prepare, phase_policy, pair
from .query_adaptive_timing import decoder_events, prefix_events


class TimedPhaseState(PhaseState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy_events=[]

    def _measure(self,method,*args,**kwargs):
        begin,end=(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
        begin.record()
        result=method(*args,**kwargs)
        end.record()
        self.policy_events.append((begin,end))
        return result

    def begin(self,*args,**kwargs):
        return self._measure(super().begin,*args,**kwargs)

    def observe_logits(self,*args,**kwargs):
        return self._measure(super().observe_logits,*args,**kwargs)


def _condition(root,cfg,name):
    if name=='native_dense':return cfg['methods'][name],None
    if name=='kernel_dense':return cfg['methods'][name],phase_policy(pair(-math.inf,-math.inf))
    method,target=name.rsplit('_s',1)
    frozen=json.loads((root/'configs/thresholds'/f'{name}.json').read_text())
    if 'policy' not in frozen:
        raise ValueError(f'No frozen operating point exists for timing: {name}')
    return cfg['methods'][method],frozen['policy']


def one(adapter,row,cfg,spec,policy,projections,*,profile=False):
    native=spec['method']=='native_dense'
    ctx=nullcontext((None,None)) if native else install(adapter,cfg['library'],
        policy['late'],mode='value',projections=projections,
        torch_library=cfg['torch_library'],collect=False)
    with ctx as (binding,router):
        if binding:_set_context(binding,row)
        state=(TimedPhaseState if profile else PhaseState)(
            spec['method'],router,m_ref=cfg['m_ref'],beta=cfg['beta'],
            gamma=cfg['gamma'],allocation=spec['allocation'],bootstrap=False,
            seed=row['seed'],diagnostics=False,phase_thresholds=policy or {},
            tile_mode=spec['tile_mode'])
        decoder=[];prefix=[]
        needs_policy=spec['method'] not in ('native_dense','kernel_dense')
        with (observe(adapter.model,state) if needs_policy else nullcontext()):
            with (decoder_events(adapter.model,decoder) if profile else nullcontext()):
                with (prefix_events(adapter.model,prefix) if profile else nullcontext()):
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    output=adapter.generate(_request(row))
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-started
    return dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],
        seed=row['seed'],method=spec['method'],calls=output.metadata['actual_denoising_step_count'],
        completion_tokens=output.completion_tokens,e2e_seconds=elapsed,
        decoder_seconds=sum(a.elapsed_time(b) for a,b in decoder)/1000 if profile else None,
        prefix_seconds=sum(a.elapsed_time(b) for a,b in prefix)/1000 if profile else None,
        policy_update_seconds=sum(a.elapsed_time(b) for a,b in state.policy_events)/1000
            if profile else None)


def run(root,conditions,repeats=2):
    root=Path(root);cfg,manifests=prepare(root)
    rows=manifests['final']
    if not conditions:
        conditions=['native_dense','kernel_dense','unweighted_s70','T_s70']
    if 'native_dense' not in conditions:
        raise ValueError('Timing requires native_dense for a measured speed ratio')
    smoke_path=root/'configs/timing_smoke.json'
    if not smoke_path.exists():
        raise ValueError('Run the two-prompt untraced timing smoke before the full timing sweep')
    smoke_record=json.loads(smoke_path.read_text())
    if (not smoke_record.get('passed') or smoke_record['core_fingerprint']!=cfg['fingerprint'] or
        smoke_record['source_sha256']!=sha(Path(__file__)) or
        not set(conditions).issubset({item['condition'] for item in smoke_record['checks']})):
        raise ValueError('Timing smoke is incomplete or uses different source/conditions')
    timing_spec=dict(core_fingerprint=cfg['fingerprint'],conditions=conditions,
        repeats=repeats,source_sha256=sha(Path(__file__)),
        threshold_sha256={name:sha(root/'configs/thresholds'/f'{name}.json')
            for name in conditions if name not in ('native_dense','kernel_dense')},
        order='rotating interleaved per prompt and repeat',
        warmup='one complete generation per condition',
        event_profiles='one deterministic example per task, run separately from E2E timings')
    timing_spec['fingerprint']=fingerprint(timing_spec)
    path=root/'configs/timing.json'
    if path.exists() and json.loads(path.read_text())!=timing_spec:
        raise ValueError('Timing configuration changed; use a separate results root/stage')
    atomic(path,timing_spec)
    specs={name:_condition(root,cfg,name) for name in conditions}
    adapter=_adapter(cfg);projections=Projections()
    for name,(spec,policy) in specs.items():one(adapter,rows[0],cfg,spec,policy,projections)
    representative={next(r['id'] for r in rows if r['task']==task)
                    for task in sorted({r['task'] for r in rows})}
    for repeat in range(repeats):
        for index,row in enumerate(rows):
            offset=(index+repeat)%len(conditions)
            order=conditions[offset:]+conditions[:offset]
            for name in order:
                spec,policy=specs[name]
                for mode in ('e2e','profile') if row['id'] in representative else ('e2e',):
                    dest=shard_path(root/f'timing/{mode}/repeat_{repeat}',name,row)
                    identity=fingerprint([timing_spec['fingerprint'],name,repeat,mode,
                        row['id'],row['prompt_hash'],row['seed']])
                    if dest.exists():
                        saved=json.loads(dest.read_text())
                        if saved.get('identity')!=identity:
                            raise ValueError(f'Timing shard provenance mismatch: {dest}')
                        continue
                    result=one(adapter,row,cfg,spec,policy,projections,profile=mode=='profile')
                    result.update(condition=name,repeat=repeat,mode=mode,identity=identity)
                    atomic(dest,result)
                    print(json.dumps(dict(event='timed',condition=name,
                        id=row['id'],repeat=repeat,mode=mode,
                        seconds=round(result['e2e_seconds'],3),calls=result['calls'])),flush=True)
    summary=[]
    for name in conditions:
        values=[];decoder=[];prefix=[];policy_update=[];mismatch=0
        for repeat in range(repeats):
            for row in rows:
                timed=json.loads(shard_path(root/f'timing/e2e/repeat_{repeat}',name,row).read_text())
                values.append(timed['e2e_seconds'])
                source_root=root if name not in ('native_dense','kernel_dense') else Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1'
                baseline=shard_path(source_root/'final',name,row)
                if baseline.exists():
                    expected=json.loads(baseline.read_text())
                    mismatch+=int(timed['calls']!=expected['steps'] or
                                  timed['completion_tokens']!=expected['completion_tokens'])
                if row['id'] in representative:
                    measured=json.loads(shard_path(root/f'timing/profile/repeat_{repeat}',name,row).read_text())
                    decoder.append(measured['decoder_seconds'])
                    prefix.append(measured['prefix_seconds'])
                    policy_update.append(measured['policy_update_seconds'])
                    if baseline.exists():
                        mismatch+=int(measured['calls']!=expected['steps'] or
                                      measured['completion_tokens']!=expected['completion_tokens'])
        summary.append(dict(condition=name,n=len(values),
            e2e_total_seconds=sum(values),e2e_mean_seconds=statistics.mean(values),
            decoder_mean_profiled_seconds=statistics.mean(decoder),
            prefix_mean_profiled_seconds=statistics.mean(prefix),
            policy_update_mean_profiled_seconds=statistics.mean(policy_update),
            completion_or_call_mismatches=mismatch))
    native=next(x for x in summary if x['condition']=='native_dense')
    for item in summary:item['speed_ratio_vs_native_dense']=native['e2e_total_seconds']/item['e2e_total_seconds']
    atomic(root/'timing_summary.json',summary)
    with (root/'timing_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    return summary


def smoke(root,conditions):
    root=Path(root);cfg,manifests=prepare(root)
    if not conditions:conditions=['native_dense','kernel_dense','T_s70']
    adapter=_adapter(cfg);projections=Projections()
    checks=[]
    for row in manifests['final'][:2]:
        for name in conditions:
            spec,policy=_condition(root,cfg,name)
            result=one(adapter,row,cfg,spec,policy,projections)
            source=root if name not in ('native_dense','kernel_dense') else Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1'
            path=shard_path(source/'final',name,row)
            expected=json.loads(path.read_text())
            if (result['calls']!=expected['steps'] or
                result['completion_tokens']!=expected['completion_tokens']):
                raise AssertionError(f'Untraced timing output differs: {name} {row["id"]}')
            checks.append(dict(condition=name,id=row['id'],calls=result['calls'],
                               output_parity=True))
    atomic(root/'configs/timing_smoke.json',dict(source_sha256=sha(Path(__file__)),
        core_fingerprint=cfg['fingerprint'],checks=checks,passed=True))
    return checks


def launch(root,conditions,repeats):
    root=Path(root).resolve()
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m',
        'experiments.value_direction_hopper.query_adaptive_guardrail_timing',
        'run','--root',str(root),'--repeats',str(repeats),
        '--conditions',*conditions]
    logfile=root/f'timing_{int(time.time())}.log'
    with logfile.open('xb') as output:
        worker=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],
            env=env,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(root/'timing_launch.json',dict(pid=worker.pid,log=str(logfile),
        command=command,started=time.time()))
    print(json.dumps(dict(pid=worker.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('run','launch','smoke'))
    parser.add_argument('--root',type=Path,
        default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1')
    parser.add_argument('--conditions',nargs='*',default=[])
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.stage=='launch':launch(args.root,args.conditions,args.repeats)
    elif args.stage=='smoke':print(json.dumps(smoke(args.root,args.conditions),indent=2))
    else:
        with (args.root/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(json.dumps(run(args.root,args.conditions,args.repeats),indent=2))
