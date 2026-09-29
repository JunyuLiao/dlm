"""v27 in-model forward time attribution on real captured states (diagnostic only).

Captures native denoising calls of one request (canvas 0 and a later canvas), then
profiles one complete ``model.forward`` per state with torch.profiler. Each decoder
sub-module is tagged with ``record_function`` through forward pre/post hooks, and
LOCAL vs GLOBAL attention *kernels* are tagged by wrapping the attention registry
entry. GPU time is attributed to the innermost tag. Profiling adds overhead and
is not a timing result; only the shares are used.
usage: python -m scripts.v27_forward_attribution CONFIG_JSON OUT_JSON
       CONFIG: {"model","revision","manifest","id","dataset","canvases":[0,6]}
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict

import torch


def main(argv=None):
    argv = argv or sys.argv[1:]
    cfg = json.load(open(argv[0]))
    from dllm.models import create_adapter
    from scripts import v20_profile as base
    adapter = create_adapter('diffusion_gemma', cfg['model'], device='cuda', precision='bfloat16',
                             revision=cfg['revision']).load()
    model = adapter.model
    rows = base.load_rows(cfg['manifest'])
    row = rows[cfg['id']]
    targets = [dict(dataset=cfg['dataset'], id=cfg['id'], canvas=c, call_index=0) for c in cfg['canvases']]
    captured, _ = base.capture(adapter, row, targets, 101, max_calls=3)
    modeling = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_'))
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    native = registry['sdpa']

    def tagged_attention(module, *args, **kwargs):
        tag = 'attn_kernel_local' if getattr(module, 'is_sliding', False) else 'attn_kernel_global'
        with torch.autograd.profiler.record_function(tag):
            return native(module, *args, **kwargs)

    hooks = []
    def tag_of(name, mod):
        kind = type(mod).__name__.lower()
        if 'attention' in kind:
            return 'attention_block'
        if 'moe' in kind or 'expert' in kind or 'router' in kind:
            return 'moe'
        if 'mlp' in kind:
            return 'mlp'
        if 'norm' in kind:
            return 'norm'
        if name.endswith('lm_head') or 'lm_head' in name:
            return 'lm_head'
        if 'embed' in kind:
            return 'embed'
        return None
    stack = {}
    for name, mod in model.named_modules():
        tag = tag_of(name, mod)
        if tag is None:
            continue
        def pre(m, a, tag=tag, key=name):
            ctx = torch.autograd.profiler.record_function(tag)
            ctx.__enter__()
            stack.setdefault(key, []).append(ctx)
        def post(m, a, o, key=name):
            stack[key].pop().__exit__(None, None, None)
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))
    registry['sdpa'] = tagged_attention
    report = dict(schema='v27_forward_attribution_v1', id=cfg['id'], states={})
    try:
        with torch.inference_mode():
            for canvas in cfg['canvases']:
                seq = captured.get(canvas) or []
                if not seq:
                    report['states'][str(canvas)] = 'not reached'
                    continue
                step = seq[min(2, len(seq) - 1)]
                kw = base.prepare_step(step['snapshot'], None)
                for _ in range(2):
                    base.decoder_call(model, kw)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    base.decoder_call(model, kw)
                    torch.cuda.synchronize()
                totals = defaultdict(float)
                for evt in prof.key_averages():
                    if evt.key in ('attn_kernel_local', 'attn_kernel_global', 'attention_block', 'moe',
                                   'mlp', 'norm', 'lm_head', 'embed'):
                        totals[evt.key] += evt.device_time_total / 1000.0
                all_cuda = sum(e.self_device_time_total for e in prof.key_averages()) / 1000.0
                report['states'][str(canvas)] = dict(
                    keys=int(step['prefix']['absolute']) + 256, inclusive_ms=dict(totals),
                    all_kernels_ms=all_cuda,
                    note='inclusive times: attention_block includes its kernel and projections; '
                         'attn_kernel_* are the SDPA calls only; moe/mlp may nest')
                print(canvas, report['states'][str(canvas)], flush=True)
    finally:
        registry['sdpa'] = native
        for h in hooks:
            h.remove()
    with open(argv[1], 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
