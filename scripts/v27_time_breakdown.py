"""v27: where generation time goes, on a BOUND panel's substrate (numbers only, no text).

A. Request level (substrate as in the protocol): per arm, prompt prefill, encoder canvas appends, decoder forwards and
   the rest of each denoising step (sampler, stop test, bookkeeping), as CUDA-event sums, plus request wall.
B. Forward composition: one real decoder call (index --call) of the first arm is captured and re-run EAGER under
   torch.profiler with module-scoped ranges; device kernel time per category: GLOBAL attention (module; core
   attention kernel separately), LOCAL attention (module; core separately), dense MLP, MoE router, MoE experts,
   LM head, everything else. The compiled forward's total on the same state is reported for comparison.
C. Kernels of the COMPILED forward on the same state (CUDA-graph replay, CUPTI graph-kernel records): top kernels by
   device time, so the non-attention remainder can be read off the kernels that actually run in the protocol.
D. One whole denoising step replayed on its captured inputs (compiled forward + native sampler + stop test): step
   elapsed, forward elapsed, top non-forward kernels, and the host syncs the step issues.
usage: python -m scripts.v27_time_breakdown --run-dir DIR --host IP --gpu-uuid UUID --stage S --dataset D --index I
           --budget N [--call K] --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--call', type=int, default=5)
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_substrate
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    substrate = protocol.get('substrate', 'eager')
    if substrate != 'eager':
        v27_substrate.install(model, name=substrate)
    kernel = v27_substrate.prefill_kernel(substrate) if substrate != 'eager' else 'dense64'
    row = dict(rows[protocol['ids'][a.dataset][a.index]], generation_budget=a.budget)
    events = defaultdict(list)

    def timed(label_fn, fn):
        def wrapper(*args, **kwargs):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            events[label_fn(args, kwargs)].append((s, e))
            return out
        return wrapper
    encoder = model.model.encoder
    enc_forward, dec_forward, step_fn = encoder.forward, model.forward, model._denoising_step

    def enc_label(args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        return 'encoder_prefill' if ids is not None and ids.shape[-1] > 256 else 'encoder_append'
    captured = {}
    calls = [0]
    step_calls = [0]

    def capture_step(*args, **kwargs):
        if step_calls[0] == a.call and 'step' not in captured:
            captured['step'] = {k: (copy.deepcopy(v) if k == 'past_key_values' else
                                    v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        step_calls[0] += 1
        return step_fn(*args, **kwargs)

    def capture_forward(*args, **kwargs):
        if calls[0] == a.call and 'state' not in captured:
            captured['state'] = (args, {k: (copy.deepcopy(v) if k == 'past_key_values' else
                                            v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()})
        calls[0] += 1
        return dec_forward(*args, **kwargs)
    encoder.forward = timed(enc_label, enc_forward)
    model.forward = timed(lambda *_: 'decoder_forward', capture_forward)
    model._denoising_step = timed(lambda *_: 'denoising_step', capture_step)
    results = {}
    try:
        for i, arm in enumerate(a.arm):
            config = configs[a.dataset][arm]
            if substrate != 'eager':
                v27_substrate.set_local(model, v27_substrate.local_mode_for(config))
            for attempt in range(2):                     # first run compiles / captures
                events.clear()
                calls[0] = step_calls[0] = 0
                with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
                    receipt = _one(adapter, row, protocol['seeds'][0], config)
                torch.cuda.synchronize()
            sums = {k: dict(n=len(v), total_s=round(sum(s.elapsed_time(e) for s, e in v) / 1e3, 3),
                            median_ms=round(statistics.median(s.elapsed_time(e) for s, e in v), 2))
                    for k, v in events.items()}
            step_other = (sums.get('denoising_step', {}).get('total_s', 0) - sums.get('decoder_forward', {}).get('total_s', 0))
            results[arm] = dict(request_wall_s=round(receipt['request_wall_seconds'], 3), calls=receipt['total_decoder_calls'],
                                parts=sums, step_other_than_forward_s=round(step_other, 3))
            print(json.dumps(dict(part='A_request', arm=arm, dataset=a.dataset, prompt_tokens=row.get('prompt_token_count'),
                                  **results[arm])), flush=True)
    finally:
        encoder.forward, model.forward, model._denoising_step = enc_forward, dec_forward, step_fn
    # B: composition of one captured decoder call (first arm's state), eager, profiled by module ranges
    args, kwargs = captured['state']
    eager_forward = getattr(model, '_v27_eager_model_forward', None) or dec_forward
    import importlib
    registry = importlib.import_module(type(model.model.decoder).__module__).ALL_ATTENTION_FUNCTIONS
    open_ranges = {}

    def enter(label):
        def hook(module, *_):
            r = torch.profiler.record_function(label)
            r.__enter__()
            open_ranges.setdefault(id(module), []).append(r)
        return hook

    def leave(module, *_):
        open_ranges[id(module)].pop().__exit__(None, None, None)
    handles = []
    for layer in model.model.decoder.layers:
        kind = 'local' if layer.self_attn.is_sliding else 'global'
        for mod, label in ((layer.self_attn, f'attn_{kind}_module'), (layer.mlp, 'dense_mlp'),
                           (layer.router, 'moe_router'), (layer.experts, 'moe_experts')):
            handles += [mod.register_forward_pre_hook(enter(label)), mod.register_forward_hook(leave)]
    head = getattr(model, 'lm_head', None)
    if head is not None:
        handles += [head.register_forward_pre_hook(enter('lm_head')), head.register_forward_hook(leave)]
    core_keys = [k for k in ('sdpa', 'v27_local_sdpa') if k in registry]
    originals = {k: registry[k] for k in core_keys}

    def core(fn):
        def wrapped(module, *x, **kw):
            with torch.profiler.record_function('core_local' if getattr(module, 'is_sliding', False) else 'core_global'):
                return fn(module, *x, **kw)
        return wrapped
    from experiments.numerical_qk_reuse import v27_fast_dense as fast
    with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION):
        for k in core_keys:
            registry[k] = core(registry[k])
        try:
            with torch.inference_mode():
                for _ in range(3):
                    eager_forward(*args, **kwargs)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    eager_forward(*args, **kwargs)
                    torch.cuda.synchronize()
        finally:
            for k, f in originals.items():
                registry[k] = f
            for h in handles:
                h.remove()
    with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION), torch.inference_mode():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        compiled = []
        for _ in range(6):
            torch.compiler.cudagraph_mark_step_begin()
            s.record()
            model.forward(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            compiled.append(s.elapsed_time(e))
        compiled = compiled[2:]
    device = lambda ev: getattr(ev, 'device_time_total', getattr(ev, 'cuda_time_total', 0))
    self_device = lambda ev: getattr(ev, 'self_device_time_total', getattr(ev, 'self_cuda_time_total', 0))
    averages = prof.key_averages()
    total = sum(self_device(ev) for ev in averages) / 1e3
    cats = {ev.key: round(device(ev) / 1e3, 3) for ev in averages
            if ev.key in ('attn_global_module', 'attn_local_module', 'core_global', 'core_local', 'dense_mlp',
                          'moe_router', 'moe_experts', 'lm_head')}
    cats['other'] = round(total - sum(v for k, v in cats.items() if k not in ('core_global', 'core_local')), 3)
    print(json.dumps(dict(part='B_forward_composition_eager_ms', dataset=a.dataset, call=a.call,
                          global_prefix_keys=kwargs['past_key_values'].layers[5].keys.shape[-2]
                          if hasattr(kwargs.get('past_key_values'), 'layers') else None,
                          device_total_ms=round(total, 3), categories=cats,
                          compiled_forward_ms=round(statistics.median(compiled), 3))), flush=True)

    def kernel_table(prof, top=25):
        rows = defaultdict(lambda: [0.0, 0])
        for ev in prof.events():
            if getattr(ev, 'device_type', None) == torch.autograd.DeviceType.CUDA:
                rows[ev.name][0] += ev.time_range.elapsed_us() / 1e3
                rows[ev.name][1] += 1
        busy = sum(v[0] for v in rows.values())
        ranked = sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]
        return busy, [dict(ms=round(v[0], 3), n=v[1], kernel=k[:140]) for k, v in ranked]

    def host_syncs(prof):
        names = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize', 'cudaMemcpyAsync',
                 'cudaGraphLaunch', 'cudaLaunchKernel', 'cuLaunchKernel', 'cuLaunchKernelEx')
        counts = defaultdict(int)
        for ev in prof.events():
            if ev.name in names:
                counts[ev.name] += 1
        return dict(counts)
    # C: kernels of the compiled forward (graph replay) on the captured state
    with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION), torch.inference_mode():
        for _ in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            model.forward(*args, **kwargs)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof_c:
            torch.compiler.cudagraph_mark_step_begin()
            model.forward(*args, **kwargs)
            torch.cuda.synchronize()
    busy, table = kernel_table(prof_c, top=30)
    print(json.dumps(dict(part='C_compiled_forward_kernels', dataset=a.dataset, call=a.call,
                          kernel_busy_ms=round(busy, 3), host=host_syncs(prof_c), top=table)), flush=True)
    # D: one whole denoising step (first arm's captured inputs), compiled forward + native sampler + stop test
    if 'step' in captured:
        step_kwargs = captured['step']
        forward = step_kwargs['decoder_forward']
        fwd_events = []

        def timed_forward(*x, **kw):
            s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            out = forward(*x, **kw)
            e0.record()
            fwd_events.append((s0, e0))
            return out
        replay = dict(step_kwargs, decoder_forward=timed_forward)
        with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION), torch.inference_mode():
            walls = []
            for i in range(6):
                rk = {k: (copy.deepcopy(v) if k == 'past_key_values' else v.clone() if torch.is_tensor(v) else v)
                      for k, v in replay.items()}
                torch.cuda.synchronize()
                s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s1.record()
                if i == 5:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof_d:
                        step_fn(**rk)
                        e1.record()
                        torch.cuda.synchronize()
                else:
                    step_fn(**rk)
                    e1.record()
                    torch.cuda.synchronize()
                walls.append((s1.elapsed_time(e1), fwd_events[-1][0].elapsed_time(fwd_events[-1][1])))
        busy_d, table_d = kernel_table(prof_d, top=40)
        fwd_kernels = {r['kernel'] for r in table}
        outside = [r for r in table_d if r['kernel'] not in fwd_kernels]
        steady = walls[2:5]
        print(json.dumps(dict(part='D_step_replay', dataset=a.dataset, call=a.call,
                              step_ms=round(statistics.median(w for w, _ in steady), 3),
                              forward_ms=round(statistics.median(f for _, f in steady), 3),
                              step_minus_forward_ms=round(statistics.median(w - f for w, f in steady), 3),
                              profiled_kernel_busy_ms=round(busy_d, 3), host=host_syncs(prof_d),
                              top_non_forward_kernels=outside[:15])), flush=True)


if __name__ == '__main__':
    main()
