"""v27 in-model diagnosis: CUDA-event time of every GLOBAL-layer attention call inside real requests,
split by path (dense bootstrap / FA4-or-c64 sparse consume / fused observation / other), plus the kept
fraction of every consumed map and the FA4 block-list cache builds. Uses the arm configs of an
existing direct-cost profile config (.config.json) and one of its targets. Numbers only.
usage: python -m scripts.v27_consume_timing CONFIG_JSON MANIFEST ROW_INDEX BUDGET ARM [ARM ...]
(the row must match the arm configs' thinking mode; BUDGET caps generation to bound the run)
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict


def main(argv=None):
    argv = argv or sys.argv[1:]
    config_path, manifest, row_index, budget, arm_names = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4:]
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
    records = defaultdict(list)
    kept_fraction = []
    forwards = []   # (start event, end event, host ms, {path: count} inside this forward)
    import time

    def timed(name, fn, method=True):
        def wrapper(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            records[name].append((s, e))
            counts[name] += 1
            if name == 'consume':
                skipped, eligible = a[4], a[5]
                kept_fraction.append((eligible & ~skipped).float().mean())
            return out
        return wrapper

    originals = {n: getattr(integration.Attention, n) for n in ('_consume', '_dense_native', '_fused_bootstrap_observation',
                                                                '__call__')}
    integration.Attention._consume = timed('consume', originals['_consume'])
    integration.Attention._dense_native = timed('dense_native', originals['_dense_native'])
    integration.Attention._fused_bootstrap_observation = timed('fused_observation', originals['_fused_bootstrap_observation'])
    integration.Attention.__call__ = timed('global_call_total', originals['__call__'])
    # kernel-level: the FA4 entry points themselves (also reached by the D_fa4 plugin control)
    fa4_originals = dict(dense=v27_fa4.dense, sparse_lists=v27_fa4.sparse_lists,
                         block_sparse_tensors=v27_fa4.block_sparse_tensors)
    v27_fa4.dense = timed('fa4_dense_kernel', fa4_originals['dense'])
    v27_fa4.sparse_lists = timed('fa4_sparse_kernel', fa4_originals['sparse_lists'])
    v27_fa4.block_sparse_tensors = timed('fa4_list_build', fa4_originals['block_sparse_tensors'])
    counts = defaultdict(int)
    model_forward = type(adapter.model).forward

    profile_at = {int(x) for x in os.environ.get('V27_TORCH_PROFILE', '').split(',') if x}
    profiled = []

    def forward(self, *a, **k):
        # one decoder call: GPU stream span (includes host launch gaps) and host enqueue time, no added sync
        counts.clear()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        prof = None
        if len(forwards) in profile_at:
            # optional torch.profiler view of this one forward: host ops, syncs, GPU kernel time
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        t = time.perf_counter()
        s.record()
        out = model_forward(self, *a, **k)
        e.record()
        host = (time.perf_counter() - t) * 1e3
        if prof is not None:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            profiled.append((len(forwards), dict(counts), host, prof.key_averages()))
        forwards.append((s, e, host, dict(counts)))
        return out
    type(adapter.model).forward = forward
    try:
        for name in arm_names:
            records.clear()
            kept_fraction.clear()
            forwards.clear()
            profiled.clear()
            arm = arms[name]
            cfg = dict(arm['config'], condition=arm['condition'], plugin=arm['plugin'])
            cfg.setdefault('fingerprint', 'diagnostic_control_unfingerprinted')
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                receipt = _one(adapter, row, 101, cfg)
            torch.cuda.synchronize()
            summary = {k: dict(n=len(v), median_ms=round(statistics.median(s.elapsed_time(e) for s, e in v), 4),
                               total_ms=round(sum(s.elapsed_time(e) for s, e in v), 1)) for k, v in records.items()}
            kf = [float(x) for x in kept_fraction]
            by_class = defaultdict(list)
            for s, e, host, c in forwards:
                kind = ('fused_observation' if c.get('fused_observation') else 'consume' if c.get('consume')
                        else 'dense_native' if c.get('dense_native') else 'fa4_dense_plugin' if c.get('fa4_dense_kernel')
                        else 'other')
                by_class[kind].append((s.elapsed_time(e), host))
            forward_summary = {k: dict(n=len(v), event_median_ms=round(statistics.median(x for x, _ in v), 2),
                                       host_median_ms=round(statistics.median(h for _, h in v), 2))
                               for k, v in by_class.items()}
            print(json.dumps(dict(arm=name, prompt_tokens=row['prompt_token_count'], calls=receipt['total_decoder_calls'],
                                  paths=summary, forwards=forward_summary,
                                  kept_fraction_median=round(statistics.median(kf), 4) if kf else None,
                                  fa4_list_builds=(receipt.get('counters') or {}).get('fa4_list_builds'))), flush=True)
            for index, c, host, averages in profiled:
                device = lambda ev: getattr(ev, 'self_device_time_total', getattr(ev, 'self_cuda_time_total', 0))
                ops = sorted(averages, key=lambda ev: ev.self_cpu_time_total, reverse=True)
                syncs = {ev.key: ev.count for ev in averages
                         if any(w in ev.key for w in ('Synchronize', 'DtoH', 'item', 'nonzero', 'local_scalar'))}
                print(json.dumps(dict(arm=name, forward=index, paths=c, host_ms_profiled=round(host, 2),
                                      cpu_self_ms=round(sum(ev.self_cpu_time_total for ev in averages) / 1e3, 2),
                                      gpu_kernel_ms=round(sum(device(ev) for ev in averages) / 1e3, 2),
                                      syncs=syncs,
                                      top_cpu=[(ev.key[:60], ev.count, round(ev.self_cpu_time_total / 1e3, 2))
                                               for ev in ops[:25]])), flush=True)
    finally:
        type(adapter.model).forward = model_forward
        for n, f in originals.items():
            setattr(integration.Attention, n, f)
        for n, f in fa4_originals.items():
            setattr(v27_fa4, n, f)


if __name__ == '__main__':
    main()
