"""v9 section 6: locate the request/replay gap on the PRODUCTION request path.

Diagnostic only -- never an accepted timing. One process, condition S by default
(production M1 via ``runner._runtime`` + one ``observe``, diagnostic off):

  pass A (in-process cold): natural request truncated after the late window;
      every Triton in-memory miss is logged via JITFunction.cache_hook /
      compiled_hook (host timestamps, constexpr shape), against a per-step host
      timeline (entry/exit, no synchronization) and encoder calls.
  pass B (same seed => same trajectory, in-memory warm): identical request with
      two short torch.profiler windows (early/late canvas, 2 steps each),
      labeled with record_function ranges. Kernels are attributed through their
      launch correlation to the innermost label enclosing the launch on the CPU
      thread. GPU-active time is the UNION of kernel/memcpy/memset intervals.

Everything left over is reported as unattributed; nothing is forced.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch.autograd.profiler import record_function


class Stop(Exception):
    pass


def union_length(intervals):
    total, end = 0.0, None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end - start
    return total


class Labels:
    """record_function ranges opened in forward pre-hooks, closed in post-hooks."""

    def __init__(self, model):
        self.handles, self.stack = [], []
        targets = {'DiffusionGemmaDecoderTextAttention': 'attn_module',
                   'DiffusionGemmaText4MLP': 'dense_mlp',
                   'DiffusionGemmaTextRouter': 'moe_router',
                   'DiffusionGemmaTextExperts': 'moe_experts',
                   'DiffusionGemmaDecoderModel': 'decoder_model',
                   'DiffusionGemmaEncoderModel': 'encoder'}
        for module in model.modules():
            label = targets.get(type(module).__name__)
            if label:
                self.handles.append(module.register_forward_pre_hook(self._enter(label)))
                self.handles.append(module.register_forward_hook(self._exit))

    def _enter(self, label):
        def hook(*_):
            rf = record_function(f'v9::{label}')
            rf.__enter__()
            self.stack.append(rf)
        return hook

    def _exit(self, *_):
        self.stack.pop().__exit__(None, None, None)

    def close(self):
        for handle in self.handles:
            handle.remove()


def wrap_functions():
    """Label selector internals and T bookkeeping; returns an undo list."""
    from experiments.numerical_qk_reuse import integration
    from experiments.value_direction_hopper import query_adaptive
    undo = []

    import inspect

    def wrap(owner, name, label):
        raw = inspect.getattr_static(owner, name)
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            with record_function(f'v9::{label}'):
                return original(*args, **kwargs)
        setattr(owner, name, staticmethod(wrapped) if isinstance(raw, staticmethod) else wrapped)
        undo.append((owner, name, raw))

    wrap(integration, 'attention', 'sel_route_fused')
    wrap(integration, 'route_only', 'sel_route')
    wrap(integration, 'preqk_attention', 'sel_preqk_pv')
    wrap(integration.Attention, 'observe_scores', 'sel_anchor_scores')
    wrap(integration.Attention, '_summary_for', 'sel_summary_lookup')
    wrap(integration.Sketches, 'get', 'sel_sketch')
    wrap(integration.Attention, '__call__', 'sel_call')
    wrap(query_adaptive.State, 'begin', 'T_begin')
    wrap(query_adaptive.State, 'observe_logits', 'T_observe_logits')
    return undo


def analyze(trace_path: Path, window_names: list[str]) -> dict[str, Any]:
    data = json.loads(trace_path.read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    launches, device, annotations, runtime = {}, [], collections.defaultdict(list), []
    windows = {}
    for e in events:
        if e.get('ph') != 'X':
            continue
        cat, name = e.get('cat', ''), e.get('name', '')
        ts, dur = float(e['ts']), float(e.get('dur', 0))
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            device.append((ts, ts + dur, name, cat, e.get('args', {}).get('correlation')))
        elif cat == 'cuda_runtime' or cat == 'cuda_driver':
            runtime.append((ts, dur, name, e['tid']))
            corr = e.get('args', {}).get('correlation')
            if corr is not None:
                launches[corr] = (ts, e['tid'])
        elif cat == 'user_annotation' and name.startswith('v9::'):
            if name.startswith('v9::window:'):
                windows[name[len('v9::window:'):]] = (ts, ts + dur)
            else:
                annotations[e['tid']].append((ts, ts + dur, name[4:]))
    for tid in annotations:
        annotations[tid].sort()
    report = {}
    for window, (w0, w1) in windows.items():
        inside = [d for d in device if d[0] >= w0 and d[1] <= w1 + 1e3]
        by_label = collections.defaultdict(list)
        unlaunched = 0
        for start, end, name, cat, corr in inside:
            launch = launches.get(corr)
            label = 'unlabeled_launch'
            if launch is None:
                unlaunched += 1
            else:
                lts, tid = launch
                spans = annotations.get(tid, [])
                best = None
                index = bisect.bisect_right(spans, (lts, float('inf'), ''))
                for a, b, lab in reversed(spans[:index]):
                    if a <= lts <= b and (best is None or b - a < best[1] - best[0]):
                        best = (a, b, lab)
                label = best[2] if best else 'outside_labels'
            by_label[label].append((start, end))
        span = w1 - w0
        gpu_union = union_length([(d[0], d[1]) for d in inside])
        in_window_runtime = [r for r in runtime if w0 <= r[0] <= w1]
        syncs = [r for r in in_window_runtime if 'Synchronize' in r[2] or r[2] == 'cudaMemcpy'
                 or 'MemcpyAsync' in r[2]]
        mallocs = [r for r in in_window_runtime if 'Malloc' in r[2] or 'cudaFree' in r[2]]
        launch_calls = [r for r in in_window_runtime if 'LaunchKernel' in r[2] or 'Launch' in r[2]]
        report[window] = dict(
            window_span_ms=span / 1e3, gpu_active_union_ms=gpu_union / 1e3,
            gpu_idle_within_window_ms=(span - gpu_union) / 1e3,
            kernels=len(inside), launch_api_calls=len(launch_calls),
            launch_api_host_ms=sum(r[1] for r in launch_calls) / 1e3,
            sync_like_calls=len(syncs), sync_like_host_ms=sum(r[1] for r in syncs) / 1e3,
            alloc_free_calls=len(mallocs), alloc_free_host_ms=sum(r[1] for r in mallocs) / 1e3,
            device_events_without_launch=unlaunched,
            by_label={label: dict(kernels=len(v), kernel_sum_ms=sum(b - a for a, b in v) / 1e3,
                                  union_ms=union_length(v) / 1e3)
                      for label, v in sorted(by_label.items())},
            note='union_ms per label are not additive across labels only if kernels overlap; '
                 'gpu_idle_within_window = span - union of all device intervals')
        top = collections.Counter()
        for start, end, name, cat, corr in inside:
            top[name[:80]] += end - start
        report[window]['top_kernels_ms'] = {k: v / 1e3 for k, v in top.most_common(15)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--id', default='aime26/2')
    parser.add_argument('--label', default='S', choices=('D', 'L', 'S'))
    parser.add_argument('--early', type=int, default=1, help='canvas of the early window (steps 0,1)')
    parser.add_argument('--late', type=int, default=8, help='canvas of the late window (steps 0,1)')
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import triton
    from triton.runtime.jit import JITFunction
    from dllm.models import GenerationRequest, create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows, _runtime
    from experiments.value_direction_hopper.query_adaptive import observe
    from scripts.v9_clean_request_timing import config_for
    row = next(r for r in _rows(args.manifest) if str(r['id']) == args.id)
    row = {k: v for k, v in row.items() if k not in GOLD_FIELDS}
    ns = SimpleNamespace(ids=[args.id], manifest=args.manifest, policy=args.policy,
                         model=args.model, revision=args.revision)
    config = config_for(ns, args.label)
    adapter = create_adapter('diffusion_gemma', config['model'], device='cuda', precision='bfloat16',
                             revision=config['revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    compiles: list[dict] = []
    clock = {'t0': None}

    def cache_hook(**kw):
        compiles.append(dict(fn=kw['fn'].name, t_before=time.perf_counter() - clock['t0'],
                             constants={k: v for k, v in kw['compile']['constants'].items()
                                        if isinstance(v, (int, float, bool))}))
        return False

    def compiled_hook(**kw):
        compiles[-1]['t_after'] = time.perf_counter() - clock['t0']
        compiles[-1]['seconds'] = compiles[-1]['t_after'] - compiles[-1]['t_before']
        return False

    JITFunction.cache_hook, JITFunction.compiled_hook = cache_hook, compiled_hook
    results = dict(schema='v9_request_trace_v1', id=args.id, label=args.label,
                   fingerprint=config['fingerprint'], triton=triton.__version__, passes={})

    def one_pass(name, profile):
        steps, encoders = [], []
        canvas = {'n': -1}
        windows = {args.early: 'early', args.late: 'late'}
        prof = {'p': None}
        labels = Labels(model) if profile else None
        undo = wrap_functions() if profile else []
        enc_handles = []
        for module in model.modules():
            if type(module).__name__ == 'DiffusionGemmaEncoderModel':
                enc_handles.append(module.register_forward_pre_hook(
                    lambda *_: encoders.append(dict(t_entry=time.perf_counter() - clock['t0']))))
                enc_handles.append(module.register_forward_hook(
                    lambda *_: encoders[-1].update(t_exit=time.perf_counter() - clock['t0'])))
        try:
            with _runtime(adapter, config['condition'], config) as runtime:
                state = runtime.get('state') if isinstance(runtime, dict) else None
                with (observe(model, state) if state is not None else torch.no_grad()):
                    inner = model._denoising_step

                    def outer(this, **kwargs):
                        cur = int(kwargs['cur_step'])
                        if cur == 48:
                            canvas['n'] += 1
                        local = 48 - cur
                        window = windows.get(canvas['n']) if local < 2 else None
                        if profile and window and local == 0:
                            torch.cuda.synchronize()
                            prof['p'] = torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA])
                            prof['p'].__enter__()
                            prof['rf'] = record_function(f'v9::window:{window}')
                            prof['rf'].__enter__()
                            prof['t'] = time.perf_counter()
                        entry = time.perf_counter() - clock['t0']
                        result = inner(**kwargs)
                        steps.append(dict(canvas=canvas['n'], local=local, cur_step=cur, t_entry=entry,
                                          t_exit=time.perf_counter() - clock['t0']))
                        if profile and window and local == 1:
                            torch.cuda.synchronize()
                            prof['rf'].__exit__(None, None, None)
                            wall = time.perf_counter() - prof['t']
                            prof['p'].__exit__(None, None, None)
                            path = args.trace_dir / f'{name}_{window}.json'
                            prof['p'].export_chrome_trace(str(path))
                            results.setdefault('windows_host_wall_ms', {})[window] = wall * 1e3
                            results.setdefault('trace_files', {})[window] = str(path)
                        if canvas['n'] == args.late and local == 1:
                            raise Stop()
                        return result

                    model._denoising_step = MethodType(outer, model)
                    torch.cuda.synchronize()
                    clock['t0'] = time.perf_counter()
                    try:
                        adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=8192,
                                                           temperature=0.0, seed=42, extra={'thinking': True}))
                    except Stop:
                        pass
                    torch.cuda.synchronize()
                    total = time.perf_counter() - clock['t0']
                    model._denoising_step = inner
        finally:
            for handle in enc_handles:
                handle.remove()
            if labels:
                labels.close()
            for owner, attr, original in undo:
                setattr(owner, attr, original)
        return dict(total_seconds_truncated=total, steps=steps, encoders=encoders)

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    results['passes']['A_cold'] = one_pass('A', False)
    results['passes']['A_cold']['compiles'] = list(compiles)
    compiles.clear()
    results['passes']['B_warm'] = one_pass('B', True)
    results['passes']['B_warm']['compiles'] = list(compiles)
    JITFunction.cache_hook = JITFunction.compiled_hook = None
    results['window_attribution'] = {w: analyze(Path(p), [w]).get(w) for w, p in results['trace_files'].items()}
    for name, p in results['passes'].items():
        steps = p['steps']
        p['step_host_ms_by_canvas'] = {}
        for c in sorted({s['canvas'] for s in steps}):
            spans = [1e3 * (s['t_exit'] - s['t_entry']) for s in steps if s['canvas'] == c]
            p['step_host_ms_by_canvas'][c] = dict(n=len(spans), sum=sum(spans), max=max(spans), min=min(spans))
        p['compile_seconds_total'] = sum(c.get('seconds', 0) for c in p['compiles'])
        p['compile_count'] = len(p['compiles'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({k: dict(total=v['total_seconds_truncated'], compiles=v['compile_count'],
                              compile_s=v['compile_seconds_total']) for k, v in results['passes'].items()}))
    print(json.dumps({w: {k: a[k] for k in ('window_span_ms', 'gpu_active_union_ms', 'gpu_idle_within_window_ms',
                                            'kernels', 'sync_like_calls')} for w, a in results['window_attribution'].items()}))


if __name__ == '__main__':
    main()
