"""Independent 32K vLLM profiling worker; its records are not timing evidence.

Reuse the frozen worker's native sampling, actual execution receipts and boundary
sync. Add only profiler ranges in this process. Do not import this into a formal
panel worker. Public summaries contain fixed categories, never trace names/text.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
import math
from pathlib import Path
import re
from unittest.mock import patch


LABELS = frozenset(('global_attention', 'local_attention', 'attention_other',
                   'prepare_metadata', 'kv_prefix_build', 'kv_canvas_refresh',
                   'selector', 'observation', 'consumer', 'split_lists',
                   'sample', 'method_begin', 'method_observe'))
PREFIX = 'v29.cost.'
GLOBAL_LAYERS = frozenset((5, 11, 17, 23, 29))
COUNT_FIELDS = ('denoising_forwards', 'commit_forwards', 'prefill_steps',
                'scheduler_denoising_forwards', 'scheduler_commit_forwards',
                'speculative_unused_denoising')


def attention_label(layer_name):
    match = re.search(r'(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)', str(layer_name))
    if match is None:
        return 'attention_other'
    return 'global_attention' if int(match[1]) in GLOBAL_LAYERS else 'local_attention'


def buffer_label(adapter, layer, prefix, count):
    previous = adapter.buffers.get(layer)
    return ('kv_prefix_build' if previous is None or previous['prefix'] != prefix
            or previous['nk'] != prefix+count else 'kv_canvas_refresh')


def profile_counts(phase):
    counts = {name: phase[name] for name in COUNT_FIELDS}
    if any(type(value) is not int or value < 0 for value in counts.values()) or not counts['denoising_forwards']:
        raise ValueError('missing actual diagnostic execution counts')
    if counts['denoising_forwards']-counts['scheduler_denoising_forwards'] != counts['speculative_unused_denoising']:
        raise ValueError('actual and consumed diagnostic counts differ inconsistently')
    return counts


def _nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid profiler duration')
    return value


def _device(event):
    device = getattr(event, 'device_type', '')
    return str(getattr(device, 'name', device)).split('.')[-1].upper()


def nearest_label(event):
    seen = set()
    while event is not None:
        if id(event) in seen:
            raise ValueError('cyclic profiler CPU ancestry')
        seen.add(id(event))
        name = str(getattr(event, 'name', ''))
        if name.startswith(PREFIX) and name[len(PREFIX):] in LABELS:
            return name[len(PREFIX):]
        event = getattr(event, 'cpu_parent', None)
    return 'unattributed'


def host_category(name):
    lower = name.lower()
    if any(word in lower for word in ('cudastreamsynchronize', 'cudaeventsynchronize', 'cudadevicesynchronize',
                                      'custreamsynchronize', 'cueventsynchronize')):
        return 'host_sync'
    if name in ('aten::item', 'aten::_local_scalar_dense'):
        return 'host_scalar_extract'
    if 'cudamemcpy' in lower or 'cumemcpy' in lower:
        return 'host_copy_enqueue'
    if 'cudalaunch' in lower or 'cugraphlaunch' in lower or 'cudagraphlaunch' in lower:
        return 'host_cuda_launch'
    if any(word in lower for word in ('cudamalloc', 'cudafree')):
        return 'host_allocator'
    return None


def summarize_events(events, scope_calls):
    """Associate each directly attached kernel once with its nearest CPU range.

    Do not sum inclusive range device_time_total: parent ranges overlap. Raw
    CUDA activity and CPU-associated kernel totals are reported separately; they
    are not presumed equal, additive wall time, or critical-path attribution.
    """
    ranges, gpu, host = defaultdict(lambda: dict(calls=0, cpu_inclusive_ms=0.)), defaultdict(float), {}
    raw_cuda_us = 0.
    d2h_us, d2h_count, attached = 0., 0, 0
    for event in events:
        name = str(getattr(event, 'name', ''))
        if _device(event) == 'CUDA':
            raw_cuda_us += _nonnegative(event.time_range.elapsed_us())
            if any(word in name.lower() for word in ('dtoh', 'device -> host', 'device to host')):
                d2h_us += _nonnegative(event.time_range.elapsed_us())
                d2h_count += 1
            continue
        if _device(event) != 'CPU':
            continue
        label = nearest_label(event)
        if name.startswith(PREFIX) and name[len(PREFIX):] in LABELS:
            row = ranges[name[len(PREFIX):]]
            row['calls'] += 1
            row['cpu_inclusive_ms'] += _nonnegative(event.cpu_time_total)/1000
        for kernel in getattr(event, 'kernels', ()):
            gpu[label] += _nonnegative(kernel.duration)/1000
            attached += 1
        category = host_category(name)
        if category:
            row = host.setdefault(category, dict(calls=0, cpu_self_ms=0., cpu_inclusive_ms=0., by_scope_self_ms={}))
            row['calls'] += 1
            self_ms = _nonnegative(event.self_cpu_time_total)/1000
            row['cpu_self_ms'] += self_ms
            row['cpu_inclusive_ms'] += _nonnegative(event.cpu_time_total)/1000
            row['by_scope_self_ms'][label] = row['by_scope_self_ms'].get(label, 0.)+self_ms
    if any(key not in LABELS or type(value) is not int or value < 0 for key, value in scope_calls.items()):
        raise ValueError('invalid fixed scope counters')
    return dict(range_cpu_inclusive=dict(ranges), directly_associated_gpu_ms=dict(gpu),
                raw_cuda_activity_sum_ms=raw_cuda_us/1000,
                directly_associated_kernel_records=attached,
                d2h_activity=dict(calls=d2h_count, sum_ms=d2h_us/1000), host_ops=host,
                eager_scope_calls=dict(scope_calls),
                missing_eager_attention_scopes=[name for name in ('global_attention', 'local_attention')
                                                if not scope_calls.get(name)],
                interpretation='nested CPU ranges overlap; CUDA streams overlap; totals are not additive request time; '
                               'FULL graph replay may lack eager layer ranges; host sync is observed wait, not causal attribution')


class ProfileSession:
    def __init__(self, torch, ordinal):
        if type(ordinal) is not int or ordinal < 1:
            raise ValueError('profile ordinal must follow at least one unprofiled request')
        self.torch, self.ordinal = torch, ordinal
        self.seen, self.active, self.started, self.finished = 0, False, False, False
        self.profiler = None
        self.calls = defaultdict(int)
        self.summary = None
        self.phase = None

    def request_start(self):
        selected = self.seen == self.ordinal
        self.seen += 1
        if selected:
            self.profiler = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False)
            self.profiler.__enter__()
            self.active = self.started = True

    def stop(self, failed=False):
        if self.active:
            self.active = False
            self.profiler.__exit__(None, None, None)
            self.finished = not failed
            self.summary = summarize_events(self.profiler.events(), dict(self.calls))

    @contextmanager
    def scope(self, label):
        if label not in LABELS:
            raise ValueError('unknown diagnostic scope')
        if not self.active:
            yield
            return
        self.calls[label] += 1
        with self.torch.profiler.record_function(PREFIX+label):
            yield


def wrapped(session, fn, label):
    @wraps(fn)
    def call(*args, **kwargs):
        category = label(args, kwargs) if callable(label) else label
        with session.scope(category):
            return fn(*args, **kwargs)
    return call


@contextmanager
def instrument(session, panel, metrics, va, dg, fa, integration, arm):
    """Process-local wrappers, always restored; no event or device sync calls."""
    with ExitStack() as stack:
        def tag(owner, name, label):
            stack.enter_context(patch.object(owner, name, wrapped(session, getattr(owner, name), label)))

        tag(fa.FlashAttentionImpl, 'forward', lambda args, _: attention_label(getattr(args[1], 'layer_name', '')))
        tag(va.VllmMethodAdapter, 'forward', 'global_attention')
        tag(va.VllmMethodAdapter, '_buffers', lambda args, _: buffer_label(args[0], args[1], args[5], args[6]))
        tag(va.VllmMethodAdapter, 'sparse_lists', 'consumer')
        tag(va.VllmMethodAdapter, '_split', 'split_lists')
        tag(va.VllmMethodAdapter, 'on_prepare', 'method_begin')
        tag(va.VllmMethodAdapter, 'on_sample', 'method_observe')
        tag(dg, '_compiled_sample_step', 'sample')
        if integration is not None:
            for name, label in (('_route', 'selector'), ('_route_dense_prefix', 'selector'),
                                ('_observe', 'observation'), ('_fused_bootstrap_observation', 'observation'),
                                ('_consume', 'consumer')):
                tag(integration.Attention, name, label)
        # The adapter's metadata read is in its installed wrapper, not inside
        # the original vLLM prepare_attn. Wrap after adapter patch installation.
        original_install = va.install_vllm_patches
        def install(adapter):
            result = original_install(adapter)
            tag(dg.DiffusionGemmaModelState, 'prepare_attn', 'prepare_metadata')
            return result
        stack.enter_context(patch.object(va, 'install_vllm_patches', install))
        # Dense never calls install_vllm_patches. Do not double-wrap preparation
        # for the adapter arms: inclusive duplicate ranges would distort counts.
        if arm == 'dense':
            tag(dg.DiffusionGemmaModelState, 'prepare_attn', 'prepare_metadata')
        original_add, original_final = panel.add_tracked_request, metrics.PhaseTracker.finalize
        def add(*args, **kwargs):
            session.request_start()
            return original_add(*args, **kwargs)
        def finalize(*args, **kwargs):
            result = original_final(*args, **kwargs)
            if getattr(session, 'active', False):
                session.phase = profile_counts(result)
            session.stop()
            return result
        stack.enter_context(patch.object(panel, 'add_tracked_request', add))
        stack.enter_context(patch.object(metrics.PhaseTracker, 'finalize', finalize))
        try:
            yield
        finally:
            session.stop(failed=True)


def validate_32k(binding, read):
    cells = read(binding['cells'])
    if not cells:
        raise ValueError('empty diagnostic cells')
    rows = {(dataset, row['id']): row for dataset, path in binding['manifests'].items() for row in read(path)}
    for cell in cells:
        count = len(rows[(cell['dataset'], cell['id'])]['prompt_tokens'])
        if not 28000 <= count <= 40000:
            raise ValueError('diagnostic binding must contain only frozen 32K cells')
    return len(cells)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--profile-ordinal', type=int, default=1)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--arm', choices=('dense', 'native', 'allkept', 'method'), required=True)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--query-block', type=int, choices=(128, 64), default=128)
    parser.add_argument('--canvas-buffers', choices=('legacy', 'release_after_invalidate'), default='legacy')
    args = parser.parse_args(argv)
    if args.summary.exists() or args.run_dir.exists() or args.profile_ordinal < 1:
        raise ValueError('new diagnostic destinations and an ordinal after warm-up required')
    from scripts import v27_vllm_panel_run as panel, v27_vllm_metrics as metrics, v28_vllm_qualify as worker
    binding = panel.read(args.binding)
    spec = panel.read(binding['spec'])
    panel.validate_binding(binding, spec)
    if Path(__file__).resolve() not in {Path(path).resolve() for path in binding['files']}:
        raise ValueError('frozen diagnostic binding must pin this profiler source')
    validate_32k(binding, panel.read)
    from experiments.numerical_qk_reuse import v21
    config = panel.read(binding['config'])
    v21.validate_effective(config, config['condition'])
    # Existing worker performs effective-config, execution-setting and bound
    # source checks before engine construction. Diagnostic imports stay lazy.
    import torch
    import experiments.numerical_qk_reuse.vllm_adapter as va
    integration = None
    if args.arm == 'method':
        # Match the adapter's existing dependency-light HF binding stub before
        # importing the core for range attachment; do not load the HF runner.
        va._stub_runner()
        import experiments.numerical_qk_reuse.integration as integration
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.attention.backends import flash_attn as fa
    session = ProfileSession(torch, args.profile_ordinal)
    worker_args = ['--binding', args.binding, '--arm', args.arm, '--block', str(args.block),
                   '--run-dir', str(args.run_dir), '--query-block', str(args.query_block),
                   '--canvas-buffers', args.canvas_buffers, '--mode', 'benchmark']
    with instrument(session, panel, metrics, va, dg, fa, integration, args.arm):
        worker.main(worker_args)
    if not session.finished or session.summary is None or session.phase is None:
        raise ValueError('selected diagnostic request was not profiled')
    report = dict(schema='v29_32k_cost_profile_v1', arm=args.arm, profile_ordinal=args.profile_ordinal,
                  deploy_commit=binding['deploy_commit'], torch=torch.__version__,
                  profiler=session.summary, performance_claim_allowed=False, quality_evaluated=False,
                  profiled_native_counts=session.phase,
                  measurement='independent profiled request; initialization and preceding requests unprofiled',
                  native_clock='unchanged runner plus actual async execution receipts',
                  caveat='profiled worker records must never be ingested as formal timing; ranges/trace overhead unknown')
    with args.summary.open('x', encoding='utf8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return report


if __name__ == '__main__':
    main()
