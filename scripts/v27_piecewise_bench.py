"""v27 substrate diagnostic: real native-adaptive requests with the decoder forward compiled PIECEWISE (the vLLM
pattern): every attention module (and the runtime-capture pre-hook of the attention binding) stays eager
behind torch._dynamo.disable, so the variable-length KV concat, the dense/sparse FA4 calls and all M1/M2/M3
state logic run unchanged; everything between them (LOCAL layers, MoE, norms, LM head) is compiled with CUDA
graphs. (All attention modules, LOCAL included, are eager boundaries -- exactly vLLM's piecewise split -- which also
keeps the plugins' Python counters out of traced code.) Per arm: per-decoder-call event span / host time by path, request receipt counters, tokens.

usage: python v27_piecewise_bench.py CONFIG_JSON MANIFEST ROW_INDEX BUDGET BACKEND ARM [ARM ...]
       BACKEND: eager | cudagraphs (dynamo CUDA graphs, eager kernels) | inductor (mode=reduce-overhead)
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict


def piecewise(model, backend):
    import torch
    import torch._dynamo as dynamo
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(dynamo.config, name):
            setattr(dynamo.config, name, 256)
    if hasattr(dynamo.config, 'accumulated_recompile_limit'):
        dynamo.config.accumulated_recompile_limit = 4096
    if os.environ.get('V27_PW_INLINE') == '0':
        # parameters stay module attributes (graph constants), not lifted graph inputs that CUDA graphs would copy
        dynamo.config.inline_inbuilt_nn_modules = False
    import copy
    import importlib
    decoder = model.model.decoder
    modeling = importlib.import_module(type(decoder).__module__)
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    local_mode = os.environ.get('V27_PW_LOCAL', 'graph')
    eager_attention = graphed_local = 0
    if local_mode == 'graph':
        # LOCAL layers already take the native SDPA path in every arm (plugins pass them through untouched);
        # binding them to that same pure function under a private key keeps plugin counters/flags out of traced
        # code, so the 25 LOCAL layers (static shapes once the sliding cache is full) join the CUDA graphs.
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        registry['v27_local_sdpa'] = sdpa_attention_forward
    for layer in decoder.layers:
        attn = layer.self_attn
        if attn.is_sliding and local_mode == 'graph':
            cfg = copy.copy(attn.config)
            cfg._attn_implementation = 'v27_local_sdpa'
            if cfg._attn_implementation != 'v27_local_sdpa':
                raise RuntimeError('could not bind LOCAL attention implementation')
            attn.config = cfg
            graphed_local += 1
        else:
            # vLLM-style piecewise boundary: this attention module runs eager (GLOBAL: our FA4 / M1-M3 path)
            attn.forward = dynamo.disable(attn.forward)
            eager_attention += 1
    base = model.model
    register = base.register_forward_pre_hook

    def register_eager(hook, *a, **k):
        return register(dynamo.disable(hook), *a, **k)
    base.register_forward_pre_hook = register_eager
    inner = model.forward
    if backend == 'inductor':
        model.forward = torch.compile(inner, mode='reduce-overhead', dynamic=False)
    else:
        model.forward = torch.compile(inner, backend='cudagraphs', dynamic=False)
    sampler_compiled = os.environ.get('V27_PW_SAMPLER', '1') == '1'
    if sampler_compiled:
        # exactly the official DiffusionGemma compiled path (_compile_functions): accept/renoise compiled once with
        # mode='reduce-overhead', fullgraph=True and cached on the model; the stopping criterion is assigned the
        # same way the official code does it
        prepare_sampler, prepare_stop = model._prepare_sampler, model._prepare_diffusion_stopping_criteria

        def compiled_sampler(*a, **k):
            sampler = prepare_sampler(*a, **k)
            if not hasattr(model, '_v27_accept'):
                model._v27_accept = torch.compile(sampler.accept_canvas, mode='reduce-overhead', fullgraph=True)
                model._v27_renoise = torch.compile(sampler.renoise_canvas, mode='reduce-overhead', fullgraph=True)
            sampler.accept_canvas, sampler.renoise_canvas = model._v27_accept, model._v27_renoise
            return sampler

        def compiled_stop(*a, **k):
            stop = prepare_stop(*a, **k)
            if stop is not None:
                if not hasattr(model, '_v27_stop'):
                    model._v27_stop = torch.compile(stop.__call__, mode='reduce-overhead', fullgraph=True)
                stop.__call__ = model._v27_stop
            return stop
        model._prepare_sampler, model._prepare_diffusion_stopping_criteria = compiled_sampler, compiled_stop
    return dict(eager_attention_modules=eager_attention, graphed_local_attention_modules=graphed_local,
                sampler_compiled=sampler_compiled)


def main(argv=None):
    argv = argv or sys.argv[1:]
    config_path, manifest, row_index, budget, backend = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4]
    arm_names = argv[5:]
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import integration, v27_fa4
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    config = json.load(open(config_path))
    arms = {a['name']: a for a in config['arms']}
    row = dict(json.load(open(manifest))[row_index], generation_budget=budget)
    adapter = create_adapter('diffusion_gemma', config['model'], device='cuda', precision='bfloat16',
                             revision=config.get('revision')).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    counts = defaultdict(int)

    def counted(name, fn):
        def wrapper(*a, **k):
            counts[name] += 1
            return fn(*a, **k)
        return wrapper
    originals = {n: getattr(integration.Attention, n) for n in ('_consume', '_dense_native', '_fused_bootstrap_observation')}
    integration.Attention._consume = counted('consume', originals['_consume'])
    integration.Attention._dense_native = counted('dense_native', originals['_dense_native'])
    integration.Attention._fused_bootstrap_observation = counted('fused_observation', originals['_fused_bootstrap_observation'])
    fa4_dense = v27_fa4.dense
    v27_fa4.dense = counted('fa4_dense_kernel', fa4_dense)
    setup = piecewise(adapter.model, backend) if backend != 'eager' else {}
    forward_inner = adapter.model.forward
    forwards = []

    def forward(*a, **k):
        counts.clear()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        s.record()
        out = forward_inner(*a, **k)
        e.record()
        forwards.append((s, e, (time.perf_counter() - t) * 1e3, dict(counts)))
        return out
    adapter.model.forward = forward
    step_inner = adapter.model._denoising_step
    steps = []

    def step(*a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        s.record()
        out = step_inner(*a, **k)
        e.record()
        steps.append((s, e, (time.perf_counter() - t) * 1e3))
        return out
    adapter.model._denoising_step = step
    try:
        for name in arm_names:
            forwards.clear()
            steps.clear()
            arm = arms[name]
            cfg = dict(arm['config'], condition=arm['condition'], plugin=arm['plugin'])
            cfg.setdefault('fingerprint', 'diagnostic_control_unfingerprinted')
            torch.cuda.synchronize()
            t = time.perf_counter()
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                receipt = _one(adapter, row, 101, cfg)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            by_class = defaultdict(list)
            for s, e, host, c in forwards:
                kind = ('fused_observation' if c.get('fused_observation') else 'consume' if c.get('consume')
                        else 'dense_native' if c.get('dense_native') else 'fa4_dense_plugin' if c.get('fa4_dense_kernel')
                        else 'other')
                by_class[kind].append((s.elapsed_time(e), host))
            summary = {k: dict(n=len(v), event_median_ms=round(statistics.median(x for x, _ in v), 2),
                               host_median_ms=round(statistics.median(h for _, h in v), 2),
                               event_sum_s=round(sum(x for x, _ in v) / 1e3, 3))
                       for k, v in by_class.items()}
            tokens = receipt.get('generated_token_ids') or receipt.get('token_ids') or []
            print(json.dumps(dict(arm=name, backend=backend, torch=torch.__version__, setup=setup,
                                  prompt_tokens=row['prompt_token_count'], calls=receipt['total_decoder_calls'],
                                  request_wall_s=round(wall, 3), forwards=summary,
                                  decode_span_s=receipt.get('generation_excluding_initial_prefill_seconds'),
                                  steps=dict(n=len(steps),
                                             event_median_ms=round(statistics.median(s.elapsed_time(e) for s, e, _ in steps), 2) if steps else None,
                                             host_median_ms=round(statistics.median(h for _, _, h in steps), 2) if steps else None,
                                             event_sum_s=round(sum(s.elapsed_time(e) for s, e, _ in steps) / 1e3, 3)),
                                  forward_event_sum_s=round(sum(s.elapsed_time(e) for s, e, _, _ in forwards) / 1e3, 3),
                                  tokens_sha=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16] if tokens else None,
                                  receipt_keys=sorted(receipt)[:40])), flush=True)
        try:
            from torch._dynamo.utils import counters
            print(json.dumps(dict(dynamo_counters={k: dict(v) for k, v in counters.items() if k in ('stats', 'graph_break', 'recompiles')})), flush=True)
        except Exception as exc:
            print(json.dumps(dict(dynamo_counters_error=str(exc)[:200])))
    finally:
        adapter.model.forward = forward_inner
        adapter.model._denoising_step = step_inner
        v27_fa4.dense = fa4_dense
        for n, f in originals.items():
            setattr(integration.Attention, n, f)


if __name__ == '__main__':
    main()
