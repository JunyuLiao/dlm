"""Frozen vLLM panel worker. Raw completions and binding stay private.

Request-boundary synchronization only. Real scheduler phases count denoising
forwards separately from encoder commits; repeats are NOT per-request seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_binding(binding, spec):
    if binding['protocol_id'] != spec['protocol_id']:
        raise ValueError('protocol mismatch')
    for path, expected in binding['files'].items():
        if digest(path) != expected:
            raise ValueError('frozen input/source drift')
    if binding['deploy_commit'] != Path('DEPLOY_SHA').read_text().strip():
        raise ValueError('deploy commit drift')


def add_tracked_request(engine, tracker, rid, prompt, params):
    internal_rid = engine.add_request(rid, prompt, params)
    if not isinstance(internal_rid, str) or not internal_rid:
        raise ValueError('vLLM did not return its internal request identity')
    tracker.start_request(internal_rid)


def validate_receipts(arm, receipts, n, required):
    if arm == 'dense':
        if receipts is not None:
            raise ValueError('dense unexpectedly intercepted')
        return
    a = receipts['adapter']
    if a['order_errors'] or a['begins'] != n or a['observes'] != n or a['global_calls'] != 5*n:
        counts = {k:a[k] for k in ('begins','observes','global_calls','invalidates','order_errors') if k in a}
        raise ValueError(f'adapter clock/coverage mismatch: scheduler_N={n}, adapter={counts}')
    if arm == 'method':
        m = receipts['method']
        effective = m['effective_method']
        if any(effective.get(k) != v for k, v in required.items()):
            raise ValueError('effective method drift')
        if m['active_layers'] != [5, 11, 17, 23, 29]:
            raise ValueError('GLOBAL scope drift')
        if m['gated_native_calls'] or m['layer_native_calls'] or m['unsupported_mask_refreshes']:
            raise ValueError('unexpected native fallback')
        if not m['fused_observations'] or not m['dp_routes'] or not a['split_fa4_calls']:
            raise ValueError('intended sparse path did not execute')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True)
    p.add_argument('--arm', choices=['dense', 'native', 'allkept', 'method'], required=True)
    p.add_argument('--block', type=int, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    binding = read(a.binding)
    spec = read(binding['spec'])
    validate_binding(binding, spec)
    a.run_dir.mkdir(parents=True, exist_ok=False)
    start_process = time.perf_counter()
    status = dict(protocol_id=spec['protocol_id'], arm=a.arm, block=a.block,
                  run_id=a.run_dir.parent.name+'_'+a.run_dir.name, complete=False)
    try:
        run(a, binding, spec, status)
        status['complete'] = True
    finally:
        status['gpu_reserved_seconds'] = round(time.perf_counter()-start_process, 3)
        (a.run_dir/'terminal.json').write_text(json.dumps(status, indent=2)+'\n')


def run(a, binding, spec, status):
    # Imports happen only after the frozen binding and unique destination pass.
    import torch
    import vllm
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from scripts.v27_vllm_metrics import PhaseTracker, graph_snapshot

    settings = spec['settings']
    engine_seed = spec['blocks'][a.block]['engine_seed']
    repeats = spec['blocks'][a.block]['repeats']
    rows = {}
    cells = read(binding['cells'])
    for ds, path in binding['manifests'].items():
        for row in read(path):
            rows[(ds, row['id'])] = row
    if a.preflight:
        # Longest eligible item is selected without using an answer or completion.
        cells = [max(cells, key=lambda c: len(rows[(c['dataset'], c['id'])]['prompt_tokens']))]
        repeats = [0]
    elif a.arm in spec['controls']:
        cells = [c for c in cells if c['index'] in spec['control_indices'][c['dataset']]]
    adapter = None
    config = read(binding['config'])
    import experiments.numerical_qk_reuse.vllm_adapter as va
    if a.arm != 'dense':
        hf = AutoConfig.from_pretrained(binding['model'])
        txt = getattr(hf, 'text_config', hf)
        adapter = va.VllmMethodAdapter(txt.layer_types, config=config if a.arm=='method' else None,
                    condition=config['condition'] if a.arm=='method' else None, arm=a.arm, profile=False)
        va.install_vllm_patches(adapter)
    tracker = PhaseTracker()
    tracker.install_scheduler_hook()
    tracker.install_execution_hook()
    cg = spec['arm_settings'][a.arm]['cudagraph_mode']
    kw = dict(model=binding['model'], dtype='bfloat16', max_model_len=settings['max_model_len'],
              max_num_seqs=1, max_num_batched_tokens=settings['chunk'], enable_chunked_prefill=True,
              gpu_memory_utilization=settings['gpu_memory_utilization'], enable_prefix_caching=False,
              trust_remote_code=False, seed=engine_seed, block_size=settings['block_size'])
    if cg != 'default':
        kw['compilation_config'] = {'cudagraph_mode':cg}
    llm = LLM(**kw)
    engine = llm.llm_engine
    tokenizer = llm.get_tokenizer()
    gpu_uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if gpu_uuid != binding['gpu_uuid']:
        raise ValueError('GPU identity drift')
    meta = dict(schema='v27_vllm_panel_v1', protocol_id=spec['protocol_id'], deploy_commit=binding['deploy_commit'],
                host=binding['host'], gpu_uuid=gpu_uuid, arm=a.arm, engine_seed=engine_seed, seed_applied=False,
                measurement_mode='request_boundary_sync', run_id=status['run_id'], qualification_only=a.preflight,
                adapter_sha256=digest(va.__file__) if adapter else None,
                method_fingerprint=config['fingerprint'] if a.arm=='method' else None,
                compilation_config=cg, cudagraph_mode=cg, torch=torch.__version__,vllm=vllm.__version__,**settings)
    rng = random.Random(spec['order_seed'] + a.block)
    rng.shuffle(cells)
    # Warm each shape once per fresh engine, outside timing/quality outputs.
    schedule = [(True,c,-1) for c in cells]
    for rep in repeats:
        order = list(cells)
        random.Random(spec['order_seed']+rep).shuffle(order)
        schedule.extend((False,c,rep) for c in order)
    status['expected_timed'] = sum(not warm for warm,_,_ in schedule)
    status['completed_timed'] = 0
    with (a.run_dir/'records.jsonl').open('x') as public, (a.run_dir/'completions.private.jsonl').open('x') as private:
        for ordinal,(warm,cell,rep) in enumerate(schedule):
            row = rows[(cell['dataset'],cell['id'])]
            rid = f'{a.arm}-{ordinal}'
            params = SamplingParams(max_tokens=int(row['generation_budget']), skip_special_tokens=False)
            tracker.start_request(rid)
            before = graph_snapshot()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            if adapter:
                adapter.begin_request()
            add_tracked_request(engine, tracker, rid, TokensPrompt(prompt_token_ids=list(row['prompt_tokens'])), params)
            final = None
            while engine.has_unfinished_requests():
                for o in engine.step():
                    if o.finished:
                        final = o
            torch.cuda.synchronize()
            receipts = adapter.end_request() if adapter else None
            torch.cuda.synchronize()
            end = time.perf_counter()
            phase = tracker.finalize(begin,end)
            after = graph_snapshot()
            if final is None or len(final.outputs)!=1:
                raise ValueError('missing final output')
            n = phase['denoising_forwards']
            validate_receipts(a.arm,receipts,n,spec['primary_receipt_method'])
            if warm:
                print(json.dumps(dict(event='warm',dataset=cell['dataset'],index=cell['index'],arm=a.arm)),flush=True)
                continue
            captures = after['num_cudagraph_captured']-before['num_cudagraph_captured']
            compile_delta = {k:after[k]-before[k] for k in before}
            output = final.outputs[0]
            rec = dict(meta,dataset=cell['dataset'],index=cell['index'],repeat=rep,
                       wall_s=end-begin,prefill_s=phase['prefill_s'],decode_span_s=phase['decode_span_s'],
                       denoise_forward_count=n,commit_forward_count=phase['commit_forwards'],
                       scheduler_denoise_forward_count=phase['scheduler_denoising_forwards'],
                       scheduler_commit_forward_count=phase['scheduler_commit_forwards'],
                       speculative_unused_denoising=phase['speculative_unused_denoising'],
                       execution_count_source='vllm_existing_async_cpu_snapshot',
                       scheduler_steps=phase['scheduler_steps'],prefill_steps=phase['prefill_steps'],
                       phase_boundary=phase['phase_boundary'],output_tokens=len(output.token_ids),
                       finish_reason=output.finish_reason,graph_captures_timed=captures,
                       compilation_deltas=compile_delta,receipts=receipts)
            public.write(json.dumps(rec)+'\n');public.flush()
            raw = tokenizer.decode(output.token_ids,skip_special_tokens=False)
            priv = dict(dataset=cell['dataset'],index=cell['index'],repeat=rep,arm=a.arm,run_id=status['run_id'],
                        id=cell['id'],completion=raw,finish_reason=output.finish_reason,stop_reason=output.stop_reason)
            private.write(json.dumps(priv)+'\n');private.flush()
            status['completed_timed'] += 1
            print(json.dumps(dict(event='run',dataset=cell['dataset'],index=cell['index'],repeat=rep,arm=a.arm,
                                  W=rec['wall_s'],N=n,graphs=captures)),flush=True)
            if captures or any(compile_delta[k] for k in ('num_backend_compilations','num_inductor_compiles')):
                raise ValueError('timed compile/capture; retain failed run and relaunch into a new directory')


if __name__ == '__main__':
    main()
