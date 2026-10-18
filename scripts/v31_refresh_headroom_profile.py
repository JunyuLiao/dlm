"""Process-local, audit-only refresh headroom instrumentation.

Install after the native adapter is installed. Never install in a clean worker.
CUDA-event spans are inclusive stream-clock spans, not additive GPU activity or
request time. Optional private Chrome traces provide disjoint activity accounting
and union-of-intervals occupancy; they still do not identify the critical path.
No synchronization, tensor reads, model changes, or graph-mode changes are added.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import heapq
import json
import math
from pathlib import Path
import re
import threading
import time
from unittest.mock import patch


PREFIX = 'refresh_headroom.scope.'
GLOBAL_LAYERS = frozenset((5, 11, 17, 23, 29))
LAYER = re.compile(r'(?:^|\.)layers\.(\d+)\.')
PHASE_COUNTERS = (('initial', 'mage_selections'), ('refresh', 'mage_reselections'),
                  ('held', 'mage_reused_calls'), ('carried', 'mage_carried_calls'),
                  ('bootstrap_dense', 'mage_warm_dense_calls'))


def union_duration(intervals):
    """Union length in the input time unit; rejects broken trace timestamps."""
    rows = sorted((float(a), float(b)) for a, b in intervals)
    if any(not math.isfinite(a) or not math.isfinite(b) or b < a for a, b in rows):
        raise ValueError('invalid timing interval')
    total = 0.0
    if not rows:
        return total
    lo, hi = rows[0]
    for a, b in rows[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + hi - lo


def global_phase(before, after):
    changed = [name for name, key in PHASE_COUNTERS if after.get(key, 0) > before.get(key, 0)]
    return changed[0] if len(changed) == 1 else ('mixed' if changed else 'dense')


def attention_scope(layer_name, phase, before=None, after=None):
    found = LAYER.search(str(layer_name))
    if found is None or not 0 <= int(found[1]) < 30:
        return phase + '/attention_unknown'
    if int(found[1]) not in GLOBAL_LAYERS:
        return phase + '/local_attention'
    suffix = global_phase(before or {}, after or {}) if phase == 'denoise' else 'dense'
    return phase + '/global_' + suffix


def kernel_family(name):
    """Conservative fallback for compiled work without an eager Python scope.

    This is a name-based diagnostic, not proof of layer/phase attribution. Do
    not classify generic GEMMs as MoE; the remaining model work stays unknown.
    """
    name = str(name).lower()
    if any(x in name for x in ('fused_moe', 'moe_', 'moe::', 'grouped_gemm', 'groupedgemm')):
        return 'kernel_name_moe_candidate'
    if any(x in name for x in ('gemm', 'matmul', 'cublas')):
        return 'kernel_name_other_gemm'
    if 'memcpy' in name or 'memset' in name:
        return 'copy_or_memset'
    return 'unattributed_rest'


def _external(args):
    return args.get('External id', args.get('external id', args.get('external_id')))


def _category(label):
    # Detailed discovery/consumer/buffer scopes are separate exclusive buckets.
    # Their inclusive parent GLOBAL scope must never be added again.
    return label or 'unattributed_rest'


def summarize_chrome(trace, scope_labels):
    """Associate each raw CUDA activity exactly once using Chrome correlation.

    CPU launch locations are mapped to the innermost instrumented annotation
    on the same CPU thread. GPU activities are joined using External id, then
    runtime correlation when available. Missing links remain unclassified.
    CUDA overlap is reported, never converted to a sum of wall-time shares.
    """
    events = trace.get('traceEvents', []) if isinstance(trace, dict) else trace
    scopes, cpu, gpu, runtime = defaultdict(list), defaultdict(list), [], []
    for e in events:
        if e.get('ph') != 'X' or 'ts' not in e or 'dur' not in e:
            continue
        a, b = float(e['ts']), float(e['ts']) + float(e['dur'])
        if not math.isfinite(a) or not math.isfinite(b) or b < a:
            raise ValueError('invalid Chrome activity interval')
        name, cat = str(e.get('name', '')), str(e.get('cat', '')).lower()
        key = (e.get('pid'), e.get('tid'))
        args = e.get('args') or {}
        if name.startswith(PREFIX):
            ident = name[len(PREFIX):]
            scopes[key].append((a, b, scope_labels.get(ident)))
        elif any(x in cat for x in ('kernel', 'gpu_memcpy', 'gpu_memset')):
            gpu.append((e, a, b))
        elif cat in ('cpu_op', 'cuda_runtime', 'cuda_driver'):
            cpu[key].append((e, a, b))
            if cat in ('cuda_runtime', 'cuda_driver'):
                runtime.append(e)

    external_labels = {}
    for key, operations in cpu.items():
        ranges = sorted(scopes[key], key=lambda row: (row[0], row[1]))
        active, i = [], 0
        for e, a, _ in sorted(operations, key=lambda row: row[1]):
            while i < len(ranges) and ranges[i][0] <= a:
                start, end, label = ranges[i]
                heapq.heappush(active, (-start, end, i, label))
                i += 1
            while active and active[0][1] <= a:
                heapq.heappop(active)
            ext = _external(e.get('args') or {})
            if ext is not None and active and active[0][3]:
                external_labels[ext] = active[0][3]
    correlation_labels = {}
    for e in runtime:
        args = e.get('args') or {}
        ext, corr = _external(args), args.get('correlation')
        if ext in external_labels and corr is not None:
            correlation_labels[corr] = external_labels[ext]

    categories = defaultdict(lambda: dict(activities=0, activity_sum_ms=0.0, intervals=[]))
    devices = defaultdict(list)
    classified = 0
    for e, a, b in gpu:
        args = e.get('args') or {}
        label = external_labels.get(_external(args)) or correlation_labels.get(args.get('correlation'))
        if label:
            classified += 1
        label = _category(label) if label else kernel_family(e.get('name', ''))
        row = categories[label]
        row['activities'] += 1
        row['activity_sum_ms'] += (b - a) / 1000
        row['intervals'].append((a, b))
        devices[str(e.get('pid'))].append((a, b))
    for row in categories.values():
        row['activity_union_ms'] = union_duration(row.pop('intervals')) / 1000
    device_summary = {}
    for device, intervals in devices.items():
        envelope = max(b for _, b in intervals) - min(a for a, _ in intervals)
        union = union_duration(intervals)
        device_summary[device] = dict(activity_union_ms=union/1000, activity_envelope_ms=envelope/1000,
                                      no_activity_within_envelope_ms=(envelope-union)/1000)
    return dict(gpu_activity_available=bool(gpu), cuda_activities=len(gpu), scope_linked_activities=classified,
                unlinked_activities=len(gpu)-classified, exclusive_activity_categories=dict(categories),
                devices=device_summary,
                interpretation='Each raw CUDA activity occurs in one category. Concurrent activities overlap; '
                'category sums/unions are not additive request wall time. Device union counts busy intervals. '
                'No-activity gaps are not proof of host causality. MoE fallback is only a kernel-name candidate. '
                'FULL graph replay can lack eager layer/phase associations; unknown work is retained.')


@dataclass
class Span:
    ident: int
    label: str
    start_ns: int
    end_ns: int = 0
    children: list = field(default_factory=list)
    start_event: object = None
    end_event: object = None


class HeadroomCollector:
    """Audit collector; request indices include warmups and start at zero."""
    def __init__(self, output_dir, *, profile_request_indices=(), warmup_requests=1,
                 cuda_events=True, max_spans=50000, torch_module=None):
        if len(profile_request_indices) > 1:
            raise ValueError('at most one profiled request per engine')
        if any(type(i) is not int or i < warmup_requests for i in profile_request_indices):
            raise ValueError('profiler request must follow excluded warmups')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_indices = tuple(profile_request_indices)
        self.warmup_requests, self.cuda_events, self.max_spans = warmup_requests, cuda_events, max_spans
        self.torch = torch_module
        self.request_index, self.active, self.profiler = -1, False, None
        # Hooks are installed before model loading, profiling and graph capture.
        # Keep a complete inert state until a scheduled request begins.
        self.metadata = {}
        self.phase, self.seen_denoise = 'prefill', False
        self.spans, self.stacks, self.phase_calls = [], defaultdict(list), defaultdict(int)
        self.dropped_spans = 0
        self.patches = []
        self.last_receipt = None

    def begin_request(self, metadata=None):
        if self.active:
            raise RuntimeError('previous headroom request was not finalized')
        self.request_index += 1
        self.metadata = metadata or {}
        self.phase, self.seen_denoise = 'prefill', False
        self.spans, self.stacks, self.phase_calls = [], defaultdict(list), defaultdict(int)
        self.dropped_spans = 0
        self.active = self.request_index >= self.warmup_requests
        self.start_ns = time.perf_counter_ns()
        if self.active and ((self.output_dir / f'request{self.request_index:04d}.json').exists()
                            or (self.output_dir / f'request{self.request_index:04d}.trace.private.json').exists()):
            self.active = False
            raise RuntimeError('preserve previous headroom request; use a new attempt')
        if self.active and self.request_index in self.profile_indices:
            import torch
            self.torch = self.torch or torch
            self.profiler = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False, profile_memory=False)
            self.profiler.__enter__()

    def record_phase(self, encoder):
        if not self.active:
            return
        self.phase = ('commit' if self.seen_denoise else 'prefill') if encoder else 'denoise'
        self.seen_denoise |= not encoder
        self.phase_calls[self.phase] += 1

    @contextmanager
    def scope(self, label):
        if not self.active or len(self.spans) >= self.max_spans:
            if self.active:
                self.dropped_spans += 1
            yield None
            return
        if self.torch is None:
            import torch
            self.torch = torch
        span = Span(len(self.spans), label, time.perf_counter_ns())
        stack = self.stacks[threading.get_ident()]
        if stack:
            stack[-1].children.append(span)
        stack.append(span)
        self.spans.append(span)
        annotation = self.torch.profiler.record_function(PREFIX + str(span.ident)) if self.profiler else None
        if annotation:
            annotation.__enter__()
        stream = None
        if self.cuda_events:
            stream = self.torch.cuda.current_stream()
            span.start_event = self.torch.cuda.Event(enable_timing=True)
            span.end_event = self.torch.cuda.Event(enable_timing=True)
            span.start_event.record(stream)
        try:
            yield span
        finally:
            if span.end_event is not None:
                span.end_event.record(stream)
            if annotation:
                annotation.__exit__(None, None, None)
            span.end_ns = time.perf_counter_ns()
            stack.pop()

    def end_request(self, *, boundary_completed):
        if not self.active:
            return dict(warmup=True, request_index=self.request_index, diagnostic_only=True)
        self.active = False
        if not boundary_completed:
            if self.profiler:
                self.profiler.__exit__(None, None, None)
                self.profiler = None
            raise RuntimeError('headroom events require the existing completed request boundary sync')
        rows = defaultdict(lambda: dict(calls=0, cpu_inclusive_ms=0.0, cpu_exclusive_own_scopes_ms=0.0,
                                        cuda_inclusive_span_ms=0.0))
        for span in self.spans:
            if span.end_ns < span.start_ns:
                raise RuntimeError('unfinished headroom scope')
            child_ns = union_duration((c.start_ns, c.end_ns) for c in span.children)
            row = rows[span.label]
            row['calls'] += 1
            row['cpu_inclusive_ms'] += (span.end_ns-span.start_ns)/1e6
            row['cpu_exclusive_own_scopes_ms'] += max(0, span.end_ns-span.start_ns-child_ns)/1e6
            if span.start_event is not None:
                row['cuda_inclusive_span_ms'] += span.start_event.elapsed_time(span.end_event)
        expected_attention = 30 * sum(self.phase_calls.values())
        observed_attention = sum(row['calls'] for label, row in rows.items()
                                 if '/global_' in label or label.endswith('/local_attention'))
        receipt = dict(schema='v31_refresh_headroom_audit_v1', request_index=self.request_index,
                       warmup=False, diagnostic_only=True, clean_timing=False, phase_calls=dict(self.phase_calls),
                       spans=dict(rows), spans_recorded=len(self.spans), dropped_spans=self.dropped_spans,
                       expected_attention_calls_from_prepare=expected_attention,
                       observed_attention_calls=observed_attention,
                       complete_scope_inventory=(self.dropped_spans == 0 and expected_attention > 0
                                                 and observed_attention == expected_attention),
                       interpretation='CPU exclusive means exclusive among this collector scopes only. '
                       'CUDA event spans include dispatch gaps and nested work; never sum parent and child spans. '
                       'Use separate clean request time for latency. Missing FULL graph eager scopes are not zero cost.')
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            path = self.output_dir / f'request{self.request_index:04d}.trace.private.json'
            from scripts.v29_trace_profile import export_no_overwrite
            export_no_overwrite(self.profiler, path)
            receipt['profiler'] = summarize_chrome(json.loads(path.read_text()),
                                                   {str(s.ident): s.label for s in self.spans})
            self.profiler = None
        destination = self.output_dir / f'request{self.request_index:04d}.json'
        with destination.open('x') as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
            f.write('\n')
        self.last_receipt = receipt
        return receipt

    def _patch(self, owner, name, replacement):
        p = patch.object(owner, name, replacement)
        p.start()
        self.patches.append(p)

    def close(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        self.active = False
        for p in reversed(self.patches):
            p.stop()
        self.patches.clear()


def install(adapter, *, output_dir, profile_request_indices=(), warmup_requests=1,
            cuda_events=True, max_spans=50000):
    """Attach request and attention hooks to an already-installed B=1 adapter.

    ``adapter=None`` installs FA4/native-audit hooks but leaves request start/end
    explicit: the caller must invoke collector.begin_request/end_request. This
    avoids mistaking NativeGeometryAudit construction for a measured request.
    NativeGeometryAudit must be enabled to identify dense prefill/commit phases.
    """
    import os
    if os.environ.get('VALUE_CLEAN_TIMING') == '1':
        raise ValueError('headroom instrumentation is forbidden in a clean worker')
    collector = HeadroomCollector(output_dir, profile_request_indices=profile_request_indices,
                                  warmup_requests=warmup_requests, cuda_events=cuda_events, max_spans=max_spans)
    from vllm.v1.attention.backends import flash_attn as fa
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    forward = fa.FlashAttentionImpl.forward

    @wraps(forward)
    def attention(*args, **kwargs):
        if not collector.active:
            return forward(*args, **kwargs)
        layer = args[1] if len(args) > 1 else kwargs.get('layer')
        name = getattr(layer, 'layer_name', '')
        before = dict(adapter.calls) if adapter is not None else {}
        phase = collector.phase
        with collector.scope(attention_scope(name, phase)) as span:
            result = forward(*args, **kwargs)
            if span is not None:
                span.label = attention_scope(name, phase, before, adapter.calls if adapter is not None else {})
        return result
    collector._patch(fa.FlashAttentionImpl, 'forward', attention)

    if adapter is None:
        original_record = NativeGeometryAudit.record
        @wraps(original_record)
        def record(instance, encoder, draft, *args, **kwargs):
            collector.record_phase(bool(encoder) or not draft)
            return original_record(instance, encoder, draft, *args, **kwargs)
        collector._patch(NativeGeometryAudit, 'record', record)
        return collector

    original_begin, original_end, original_prepare = adapter.begin_request, adapter.end_request, adapter.on_prepare
    @wraps(original_begin)
    def begin(*args, **kwargs):
        result = original_begin(*args, **kwargs)
        collector.begin_request()
        return result
    @wraps(original_end)
    def end(*args, **kwargs):
        report = collector.end_request(boundary_completed=True)
        result = original_end(*args, **kwargs)
        result.setdefault('adapter', {})['headroom'] = report
        return result
    @wraps(original_prepare)
    def prepare(phase_encoder, *args, **kwargs):
        result = original_prepare(phase_encoder, *args, **kwargs)
        collector.record_phase(bool(phase_encoder))
        return result
    collector._patch(adapter, 'begin_request', begin)
    collector._patch(adapter, 'end_request', end)
    collector._patch(adapter, 'on_prepare', prepare)

    for name, label in (('_buffers', 'kv_buffer'), ('_select_initial_refresh', 'discovery'),
                        ('sparse_lists', 'sparse_consumer'), ('on_sample', 'sampler_hook')):
        if not hasattr(adapter, name):
            continue
        original = getattr(adapter, name)
        def wrapped(*args, _original=original, _label=label, **kwargs):
            if not collector.active:
                return _original(*args, **kwargs)
            with collector.scope(collector.phase + '/' + _label):
                return _original(*args, **kwargs)
        collector._patch(adapter, name, wrapped)
    return collector
