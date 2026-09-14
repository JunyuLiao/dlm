"""Measured reference-backend costs, never labeled sparse-kernel speedup.

Full-budget unprofiled e2e repeats include prefill, model compute, routing and
metadata. A separate capped diagnostic pass times component spans; its event
instrumentation is excluded from the headline timings. No dense mass/error
diagnostics run in either pass. Per-component CUDA event spans include stream
idle time due to host work, and are NOT isolated kernel execution times.
"""
import argparse
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.config import MODEL,REVISION
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_install_dense,_request,_set_context
from .config import RouterConfig
from .online_run import generate,load,fingerprint,router_options
from .run import digest,sample_key
from .report import csv_write


def summarize(output):
    setup=json.loads((output/'setup.json').read_text());rows=[];summaries=[]
    for sample in setup['samples']:
        names=['dense']+[n for n,e in setup['conditions'].items() if e['benchmark']==sample['benchmark']]
        for name in names:
            repeats=[]
            for repeat in range(setup['repeats']):
                path=output/'timings'/name/f'{sample_key(sample)}_{repeat}.json'
                data=json.loads(path.read_text())
                assert data['id']==sample['id'] and data['repeat']==repeat and data['method']==name
                assert data['code_fingerprint']==setup['code_fingerprint'] and data['exact_reference_parity'] and not data['diagnostics']
                repeats.append(data)
            seconds=[r['e2e_seconds'] for r in repeats]
            rows.append(dict(id=sample['id'],benchmark=sample['benchmark'],method=name,repeats=len(repeats),
                median_e2e_seconds=float(np.median(seconds)),min_e2e_seconds=min(seconds),max_e2e_seconds=max(seconds),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in repeats),
                max_state_bytes=max(r['state_bytes'] for r in repeats)))
    dense={r['id']:r for r in rows if r['method']=='dense'}
    for benchmark,name in sorted({(r['benchmark'],r['method']) for r in rows}):
        rs=[r for r in rows if (r['benchmark'],r['method'])==(benchmark,name)]
        seconds=sum(r['median_e2e_seconds'] for r in rs);baseline=sum(dense[r['id']]['median_e2e_seconds'] for r in rs)
        summaries.append(dict(benchmark=benchmark,method=name,prompts=len(rs),repeats=setup['repeats'],
            summed_prompt_median_seconds=seconds,dense_summed_prompt_median_seconds=baseline,
            reference_e2e_time_ratio_vs_dense=seconds/baseline,max_state_bytes=max(r['max_state_bytes'] for r in rs),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rs),executed_tile_qk_avoided=0,executed_tile_pv_avoided=0))
    csv_write(output/'per_prompt.csv',rows);csv_write(output/'summary.csv',summaries)
    _write(output/'summary.json',dict(conditions=summaries,setup_sha256=digest(output/'setup.json')))
    lines=['# Measured reference-backend cost','',
        'Diagnostics are disabled for full-budget timings. Each prompt/method receives an excluded capped warmup; '
        f"{setup['repeats']} full-budget repetitions alternate execution order. All generated tokens must exactly match completed artifacts. "
        'The headline ratio sums per-prompt median times, including prefill, projections, MLP/MoE, attention and router metadata. '
        'Model loading is excluded. This is the existing eager dense-mask backend, not an optimized sparse kernel.',
        '', '| Benchmark | Method | Prompts | E2E time / dense | Peak allocated GiB | Max router state MiB |',
        '|---|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['benchmark']} | {r['method']} | {r['prompts']} | {r['reference_e2e_time_ratio_vs_dense']:.3f} | {r['peak_allocated_bytes']/2**30:.2f} | {r['max_state_bytes']/2**20:.3f} |")
    lines+=['','`components.csv` contains separate capped-pass host-wall and CUDA-stream spans for state lookup/reset, '
        'prediction, selection/protection/coverage, observations, history updates, input preparation, output attention, and total decoder override. '
        'Spans are nested and must not be summed; stream spans include host-induced idle time and are not isolated kernel measurements. '
        'Their instrumentation is excluded from the headline timing repeats. Full-budget per-repeat memory/time measurements remain in raw shards.',
        '', 'No tile QK/PV is actually avoided in this backend. These measurements establish reference implementation overhead, '
        'not net deployable sparse savings. The frozen prompt-length-spread samples are development cost probes, '
        'not a population timing estimate or held-out quality validation. Raw timing ranges are reported without claiming statistical speedup.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


class ComponentSpans:
    """Nested spans are reported individually and must not be added together."""
    def __init__(self):self.events=[];self.stack=ExitStack()

    def wrap(self,owner,name,label):
        original=getattr(owner,name)
        @wraps(original)
        def timed(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();began=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                host=time.perf_counter()-began;end.record();self.events.append((label,start,end,host))
        self.stack.enter_context(patch.object(owner,name,timed))

    def __enter__(self):
        from . import online
        from .state import History,RequestState
        for name in ('geometry','fixed_budget','protected_set','cover_rows','observations',
                     '_prepare_attention_inputs','_finish_eager_attention'):
            self.wrap(online,name,name)
        for owner,name,label in ((History,'predict','history_predict'),(History,'update','history_update'),
                (RequestState,'get','state_lookup_reset'),(online.CacheIdentity,'_hook','cache_identity_hook'),
                (online.OnlineAttention,'__call__','decoder_attention_override_total')):
            self.wrap(owner,name,label)
        return self

    def __exit__(self,*args):return self.stack.__exit__(*args)

    def summary(self):
        torch.cuda.synchronize();totals={}
        for label,start,end,host in self.events:
            row=totals.setdefault(label,dict(component=label,calls=0,host_wall_seconds=0.,cuda_stream_span_ms=0.))
            row['calls']+=1;row['host_wall_seconds']+=host;row['cuda_stream_span_ms']+=start.elapsed_time(end)
        return list(totals.values())


def dense_generate(adapter,row):
    binding=_install_dense(adapter)
    try:
        _set_context(binding,row);return adapter.generate(_request(row))
    finally:binding.close()


def measure(adapter,row,entry):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    if entry is None:out=dense_generate(adapter,row);router=None
    else:out,router,_=generate(adapter,row,RouterConfig(**entry['config']),diagnostics=False,**router_options(entry))
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    return out,router,dict(e2e_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(stage,output,names,*,samples_per_benchmark=2,repeats=3):
    setup=load(stage);fp=fingerprint(stage)
    audit=json.loads((stage/'audit.json').read_text())
    if not audit['complete']:raise RuntimeError('completed source stage required before profiling')
    smoke=json.loads((stage/'smoke.json').read_text())
    if not smoke['passed'] or smoke['fingerprint']!=fp:raise RuntimeError('matching CUDA smoke required')
    entries={name:setup['conditions'][name] for name in names}
    if not entries or samples_per_benchmark<1 or repeats<2:raise ValueError('require candidates, samples and repeated timings')
    if any(v['config']['history_source']!='sparse' or 'isolate' in v for v in entries.values()):
        raise ValueError('profile deployable own-history candidates, not paid teacher or isolated controls')
    rows=[]
    for benchmark in sorted({v['benchmark'] for v in entries.values()}):
        pool=sorted((r for r in setup['samples'] if r['benchmark']==benchmark),key=lambda r:(r['prompt_tokens'],r['id']))
        # Fixed length-spread development examples; never select by outcomes.
        indices=sorted({round(i*(len(pool)-1)/max(samples_per_benchmark-1,1)) for i in range(min(samples_per_benchmark,len(pool)))})
        rows.extend(pool[i] for i in indices)
    spec=dict(stage=str(stage),freeze_sha256=digest(stage/'freeze.json'),code_fingerprint=fp,
        performance_source_sha256=digest(Path(__file__)),conditions=entries,samples=rows,repeats=repeats,
        backend='dense-mask emulation; no executed tile QK/PV avoidance',diagnostics=False,
        sampling='development prompt-length endpoints, independent of outcome',
        warmup='one 32-token generation per method and benchmark, excluded from timing',
        order='alternate dense-first and dense-last by repeat',
        component_scope='separate capped 32-token pass; nested host/stream spans are not additive or isolated GPU kernel time')
    output.mkdir(parents=True,exist_ok=True);path=output/'setup.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('immutable performance setup differs')
    _write(path,spec)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    warm=set();timings=[];spans=[]
    for row in rows:
        methods={'dense':None,**{n:e for n,e in entries.items() if e['benchmark']==row['benchmark']}}
        for name,entry in methods.items():
            if (row['benchmark'],name) not in warm:
                measure(adapter,dict(row,generation_budget=32),entry);warm.add((row['benchmark'],name))
        for repeat in range(repeats):
            order=list(methods.items());order=order if repeat%2==0 else order[::-1]
            for name,entry in order:
                path=output/'timings'/name/f'{sample_key(row)}_{repeat}.json'
                if path.exists():data=json.loads(path.read_text())
                else:
                    out,router,timing=measure(adapter,row,entry)
                    if entry is None:
                        cache=setup['cached_dense'][row['id']];assert digest(cache['path'])==cache['sha256']
                        reference=json.loads(Path(cache['path']).read_text())
                    else:
                        reference=json.loads((stage/'conditions'/name/'shards'/f'{sample_key(row)}.json').read_text())
                    assert out.completion_tokens==reference['completion_tokens'],'profiling changed completed generation'
                    data=dict(id=row['id'],benchmark=row['benchmark'],method=name,repeat=repeat,**timing,
                        completion_tokens=out.completion_tokens,model_evaluations=out.model_evaluations,
                        costs=router.costs if router else None,state_bytes=router.max_state_bytes if router else 0,
                        exact_reference_parity=True,diagnostics=False,code_fingerprint=fp)
                    assert fingerprint(stage)==fp;_write(path,data)
                assert data['code_fingerprint']==fp
                timings.append({k:v for k,v in data.items() if k not in ('completion_tokens','costs')})
                print(row['id'],name,repeat,data['e2e_seconds'],flush=True)
        for name,entry in methods.items():
            if entry is None:continue
            path=output/'components'/name/f'{sample_key(row)}.json'
            if path.exists():data=json.loads(path.read_text())
            else:
                capped=dict(row,generation_budget=32)
                reference,_,_=measure(adapter,capped,entry)
                with ComponentSpans() as collector:out,router,timing=measure(adapter,capped,entry)
                assert out.completion_tokens==reference.completion_tokens
                data=dict(id=row['id'],benchmark=row['benchmark'],method=name,components=collector.summary(),
                    costs=router.costs,profiled_e2e_seconds=timing['e2e_seconds'],exact_unprofiled_parity=True,
                    code_fingerprint=fp)
                _write(path,data)
            assert data['code_fingerprint']==fp
            spans.extend(dict(id=row['id'],benchmark=row['benchmark'],method=name,**v) for v in data['components'])
    csv_write(output/'timings.csv',timings);csv_write(output/'components.csv',spans)
    summarize(output)
    _write(output/'audit.json',dict(complete=True,timing_runs=len(timings),code_fingerprint=fp,
        reference_parity=True,executed_sparse_speedup_claimed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions',nargs='+',required=True);p.add_argument('--samples-per-benchmark',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    run(a.stage,a.output,a.conditions,samples_per_benchmark=a.samples_per_benchmark,repeats=a.repeats)
