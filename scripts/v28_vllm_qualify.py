"""V28 qualification/benchmark wrapper around the unchanged v27 panel loop.

Freeze a new v28_* protocol/binding/config before use. The public spec declares
adapter_settings={lifecycle: request_clear, alias_splits: 2} and the frozen
primary_receipt_method includes q_block=128 or 64. Configs are never rewritten.
Qualification uses the longest eligible item, one warm and one timed request;
benchmark uses the original frozen cells/repeats/control schedule. The existing
phase tracker counts actual async execution, including unretired denoising.
Actual paged-copy checks run only in the first warm request, on method/allkept.
Native/dense have no adapter copy; their copy check is explicitly not applicable.
This script does not score or establish accuracy/speed conclusions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

from scripts import v27_vllm_panel_run as panel


ALLOCATOR_FIELDS = frozenset(('num_alloc_retries', 'num_ooms', 'allocated_bytes',
                              'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'))


def execution_settings(query_block, lifecycle):
    if query_block not in (128, 64) or lifecycle != 'request_clear':
        raise ValueError('V28 requires q128/q64 and matched request_clear')
    return dict(query_block=query_block, lifecycle=lifecycle, alias_splits=2,
                q_regroup=False, q_carry64=False,
                allkept_query_block=128, kv_check='first_warm_request_only')


def validate_frozen_inputs(binding, spec, config, query_block, lifecycle):
    settings = execution_settings(query_block, lifecycle)
    if not str(spec.get('protocol_id', '')).startswith('v28_'):
        raise ValueError('V28 needs a new frozen protocol; do not reuse a V18 binding')
    if spec.get('adapter_settings') != dict(lifecycle=lifecycle, alias_splits=2):
        raise ValueError('frozen adapter_settings differ')
    if spec.get('arms') != ['dense', 'method'] or spec.get('controls') != ['native', 'allkept']:
        raise ValueError('matched dense/method/native/allkept arms required')
    required = spec.get('primary_receipt_method', {})
    if required.get('q_block') != query_block or config.get('q_block', 128) != query_block:
        raise ValueError('q-block differs between CLI, frozen config and receipt contract')
    if config.get('q_regroup', False) or config.get('q_carry64', False):
        raise ValueError('natural q128/q64 qualification excludes regroup/carry64 variants')
    fixed = dict(score_period=64, decision_interval=6, risk_state='dense_prefix',
                 carry_first=True, fused_observe=True, async_route=True, consumer='fa4')
    if any(required.get(key) != value for key, value in fixed.items()):
        raise ValueError('main effective-method receipt contract is incomplete or changed')
    # Pin the wrapper as well as the loop/counting/adapter sources. This prevents
    # a qualified binding from silently running a later local instrumentation.
    pinned = {Path(path).resolve() for path in binding.get('files', {})}
    root = Path(__file__).resolve().parents[1]
    sources = (Path(__file__).resolve(), Path(panel.__file__).resolve(),
               root / 'scripts/v27_vllm_metrics.py',
               root / 'experiments/numerical_qk_reuse/vllm_adapter.py')
    if any(source.resolve() not in pinned for source in sources):
        raise ValueError('frozen binding must pin V28 wrapper, v27 loop/tracker and adapter')
    return settings


def warm_request_count(args, binding, spec):
    if args.preflight:
        return 1
    cells = panel.read(binding['cells'])
    if args.arm in spec['controls']:
        cells = [cell for cell in cells
                 if cell['index'] in spec['control_indices'][cell['dataset']]]
    if not cells:
        raise ValueError('no frozen warm cells for selected arm')
    return len(cells)


def sample_tokens(prefix, n, page):
    nk = prefix + n
    if prefix < 0 or n <= 0 or page <= 0:
        raise ValueError('invalid actual KV geometry')
    return sorted({token for token in (0, page - 1, page, prefix - 1,
                                       prefix, prefix + 1, nk - 1) if 0 <= token < nk})


def affine_token_stride(shape, strides):
    """Check physical page/token merge, independent of logical page consecutivity."""
    if len(shape) != 4 or len(strides) != 4:
        raise ValueError('paged cache needs four dimensions')
    return strides[-1] == 1 and strides[0] == shape[1] * strides[1]


def allocator_snapshot():
    """Host allocator metadata only: no synchronize, flush or peak reset."""
    import torch
    stats = torch.cuda.memory_stats()
    names = dict(num_alloc_retries='num_alloc_retries', num_ooms='num_ooms',
                 allocated_bytes='allocated_bytes.all.current',
                 reserved_bytes='reserved_bytes.all.current',
                 peak_allocated_bytes='allocated_bytes.all.peak',
                 peak_reserved_bytes='reserved_bytes.all.peak')
    snapshot = {public: stats.get(source) for public, source in names.items()}
    if any(type(value) is not int or value < 0 for value in snapshot.values()):
        raise ValueError('unknown or negative native allocator counters')
    return snapshot


def allocator_receipt(before, after):
    if set(before) != ALLOCATOR_FIELDS or set(after) != ALLOCATOR_FIELDS or any(
            type(value) is not int or value < 0 for value in (*before.values(), *after.values())):
        raise ValueError('allocator boundary snapshots differ or are unknown')
    deltas = {name + '_delta': after[name] - before[name] for name in ('num_alloc_retries', 'num_ooms')}
    if any(value < 0 for value in deltas.values()):
        raise ValueError('allocator cumulative counts decreased during request')
    return dict(before=before, after=after, **deltas,
                peak_scope='process cumulative; no peak reset; not an isolated request peak')


def check_actual_copy(kv_cache, buffer, table, prefix, n, head_size):
    """Compare sampled copies with an independent physical [page,head,slot,KV] oracle.

    Return shapes/strides and maximum error only, never values or page identities.
    Sampling crosses page and prefix boundaries and covers every KV head.
    """
    import torch
    if kv_cache.ndim != 4 or kv_cache.shape[-1] != 2 * head_size:
        raise ValueError('unexpected actual vLLM KV layout')
    key, value = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    page, hk = key.shape[1:3]
    nk = prefix + n
    if buffer['nk'] != nk or tuple(buffer['k'].shape) != (1, hk, nk, head_size):
        raise ValueError('adapter copy shape differs from actual KV')
    pages = table[:math.ceil(nk / page)].long()
    tokens = sample_tokens(prefix, n, page)
    channels = sorted({0, head_size - 1})
    expected, actual = [], []
    for token in tokens:
        physical_page = pages[token // page]
        for head in range(hk):
            for channel in channels:
                expected.extend((kv_cache[physical_page, head, token % page, channel],
                                 kv_cache[physical_page, head, token % page, channel + head_size]))
                actual.extend((buffer['k'][0, head, token, channel], buffer['v'][0, head, token, channel]))
    error = float((torch.stack(expected).float() - torch.stack(actual).float()).abs().max().item())
    if not math.isfinite(error) or error != 0:
        raise ValueError('actual paged K/V copy numerical mismatch')
    consecutive = bool(((pages[1:] - pages[:-1]) == 1).all().item())
    return dict(kv_shape=list(kv_cache.shape), kv_stride=list(kv_cache.stride()),
                key_shape=list(key.shape), key_stride=list(key.stride()),
                value_stride=list(value.stride()), page_size=int(page), kv_heads=int(hk),
                logical_pages_consecutive=consecutive,
                affine_token_view_possible=affine_token_stride(key.shape, key.stride()),
                sampled_tokens=len(tokens), sampled_scalars=len(expected), max_abs_error=error)


def adapter_variant(base, settings, warm_count):
    """Apply execution options and a warm-only probe without altering the core config."""
    if type(warm_count) is not int or warm_count <= 0:
        raise ValueError('a real untimed warm request is required')
    class QualifiedAdapter(base):
        def __init__(self, *args, **kwargs):
            kwargs['lifecycle'] = settings['lifecycle']
            super().__init__(*args, **kwargs)
            if self.splits != settings['alias_splits']:
                raise ValueError('alias split drift')
            self._v28_request = -1
            self._v28_checks = []
            self._v28_checked_layers = set()
            self._v28_timed_checks = 0

        def begin_request(self):
            self._v28_request += 1
            self._v28_allocator_before = allocator_snapshot()
            super().begin_request()

        def forward(self, impl, layer_idx, query, kv_cache, metadata, output):
            result = super().forward(impl, layer_idx, query, kv_cache, metadata, output)
            if self._v28_request == 0 and layer_idx not in self._v28_checked_layers:
                if self._v28_request >= warm_count:
                    self._v28_timed_checks += 1
                    raise ValueError('KV numerical probe reached a timed request')
                check = check_actual_copy(kv_cache, self.buffers[layer_idx], metadata.block_table[0],
                                          self.step_ctx['seq_len'] - self.step_ctx['n'],
                                          self.step_ctx['n'], impl.head_size)
                self._v28_checks.append(dict(layer=int(layer_idx), **check))
                self._v28_checked_layers.add(layer_idx)
            return result

        def end_request(self):
            result = super().end_request()
            result['adapter']['allocator'] = allocator_receipt(
                self._v28_allocator_before, allocator_snapshot())
            result['adapter']['v28_config'] = dict(settings)
            result['adapter']['kv_layout_checks'] = list(self._v28_checks)
            result['adapter']['kv_probe_timed_checks'] = self._v28_timed_checks
            result['adapter']['kv_copy_check_status'] = (
                'not_applicable_native' if self.arm == 'native' else 'warm_checked')
            return result
    return QualifiedAdapter


def validate_variant_receipt(arm, receipts, n, required, settings, original=panel.validate_receipts):
    original(arm, receipts, n, required)
    if arm == 'dense':
        return
    adapter = receipts['adapter']
    if adapter.get('v28_config') != settings or adapter.get('kv_probe_timed_checks') != 0:
        raise ValueError('V28 execution settings or untimed probe boundary drift')
    allocator = adapter.get('allocator', {})
    expected_allocator = allocator_receipt(allocator.get('before', {}), allocator.get('after', {}))
    if allocator != expected_allocator:
        raise ValueError('allocator boundary receipt differs')
    if arm in ('method', 'allkept'):
        checks = adapter.get('kv_layout_checks', [])
        if sorted(check['layer'] for check in checks) != [5, 11, 17, 23, 29]:
            raise ValueError('actual KV copy checks must cover all five GLOBAL layers')
        if any(check.get('max_abs_error') != 0 or check.get('page_size') != 64 for check in checks):
            raise ValueError('actual KV numerical/layout qualification failed')
        if not adapter.get('split_fa4_calls'):
            raise ValueError('alias2 consumer did not execute')
    if arm == 'method':
        effective = receipts['method']['effective_method']
        expected = dict(q_block=settings['query_block'], q_regroup=False, q_carry64=False)
        if any(effective.get(key) != value for key, value in expected.items()):
            raise ValueError('q-block effective-method drift')
        if settings['query_block'] == 64 and any(receipts['method'].get(key, 0) <= 0
                for key in ('q64_refined_routes', 'q64_list_builds')):
            raise ValueError('q64 refined selector/list path did not execute')


@contextmanager
def instrument_runner(adapter_module, settings, warm_count):
    original_adapter, original_validate = adapter_module.VllmMethodAdapter, panel.validate_receipts
    adapter_module.VllmMethodAdapter = adapter_variant(original_adapter, settings, warm_count)
    def validate(arm, receipts, n, required):
        validate_variant_receipt(arm, receipts, n, required, settings, original_validate)
    panel.validate_receipts = validate
    try:
        yield
    finally:
        adapter_module.VllmMethodAdapter, panel.validate_receipts = original_adapter, original_validate


def summarize_closed_run(records, status, settings, qualification):
    if not records or len(records) != status.get('completed_timed') or len(records) != status.get('expected_timed'):
        raise ValueError('closed-run timed inventory differs')
    for row in records:
        if row.get('qualification_only') is not qualification:
            raise ValueError('qualification marker drift')
        if any(row.get(key) != status[key] for key in ('run_id', 'arm', 'protocol_id')):
            raise ValueError('closed-run identity differs')
        if row.get('execution_count_source') != 'vllm_existing_async_cpu_snapshot':
            raise ValueError('actual async count source missing')
        fields = ('denoise_forward_count', 'scheduler_denoise_forward_count',
                  'speculative_unused_denoising', 'commit_forward_count', 'graph_captures_timed')
        if any(type(row.get(key)) is not int or row[key] < 0 for key in fields):
            raise ValueError('unknown or negative actual count/capture')
        if row['denoise_forward_count'] <= 0 or row['denoise_forward_count'] != (
                row['scheduler_denoise_forward_count'] + row['speculative_unused_denoising']):
            raise ValueError('actual N differs from retired plus unused N')
        compile_delta = row.get('compilation_deltas', {})
        if row['graph_captures_timed'] or any(type(compile_delta.get(key)) is not int or compile_delta[key] != 0
                for key in ('num_backend_compilations', 'num_inductor_compiles')):
            raise ValueError('timed compile/capture or unknown compile count')
        for key in ('wall_s', 'decode_span_s', 'prefill_s'):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('nonpositive/nonfinite real phase timing')
    allocator = None
    if status['arm'] != 'dense':
        values = [row['receipts']['adapter']['allocator'] for row in records]
        allocator = dict(num_alloc_retries_delta_sum=sum(value['num_alloc_retries_delta'] for value in values),
                         num_ooms_delta_sum=sum(value['num_ooms_delta'] for value in values),
                         process_peak_allocated_bytes_max=max(value['after']['peak_allocated_bytes'] for value in values),
                         process_peak_reserved_bytes_max=max(value['after']['peak_reserved_bytes'] for value in values),
                         end_allocated_bytes_mean=sum(value['after']['allocated_bytes'] for value in values) / len(values),
                         end_reserved_bytes_mean=sum(value['after']['reserved_bytes'] for value in values) / len(values),
                         interpretation='Process cumulative peaks; no peak reset. Not evidence of timing superiority to legacy.')
    return dict(schema='v28_vllm_qualification_v1', protocol_id=status['protocol_id'], arm=status['arm'],
                qualification_only=qualification, v28_config=settings, timed_requests=len(records),
                wall_s_mean=sum(row['wall_s'] for row in records) / len(records),
                decode_span_s_mean=sum(row['decode_span_s'] for row in records) / len(records),
                prefill_s_mean=sum(row['prefill_s'] for row in records) / len(records),
                denoise_forward_count_sum=sum(row['denoise_forward_count'] for row in records),
                scheduler_N_sum=sum(row['scheduler_denoise_forward_count'] for row in records),
                speculative_unused_N_sum=sum(row['speculative_unused_denoising'] for row in records),
                commit_forward_count_sum=sum(row['commit_forward_count'] for row in records),
                allocator_summary=allocator, allocator_status='not_applicable_dense' if allocator is None else 'observed',
                kv_layout_checks=(records[0].get('receipts') or {}).get('adapter', {}).get('kv_layout_checks', []),
                interpretation='Qualification/benchmark only; no scoring or accuracy inference. '
                               'Alias2/request_clear shared; all-kept retains q128 all-kept lists. '
                               'S/N is amortized; repeat labels are not applied sampling seeds.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--arm', choices=['dense', 'native', 'allkept', 'method'], required=True)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--query-block', type=int, choices=[128, 64], required=True)
    parser.add_argument('--lifecycle', choices=['request_clear'], default='request_clear')
    parser.add_argument('--mode', choices=['qualification', 'benchmark'], default='qualification')
    args = parser.parse_args(argv)
    args.preflight = args.mode == 'qualification'
    binding = panel.read(args.binding)
    spec, config = panel.read(binding['spec']), panel.read(binding['config'])
    panel.validate_binding(binding, spec)
    settings = validate_frozen_inputs(binding, spec, config, args.query_block, args.lifecycle)
    from experiments.numerical_qk_reuse import v21
    v21.validate_effective(config, config['condition'])
    if not 0 <= args.block < len(spec['blocks']):
        raise ValueError('block outside frozen protocol')
    warm_count = warm_request_count(args, binding, spec)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    status = dict(protocol_id=spec['protocol_id'], arm=args.arm, block=args.block,
                  run_id=args.run_dir.parent.name + '_' + args.run_dir.name,
                  complete=False, v28_config=settings, mode=args.mode)
    started = time.perf_counter()
    try:
        import experiments.numerical_qk_reuse.vllm_adapter as adapter_module
        with instrument_runner(adapter_module, settings, warm_count):
            panel.run(args, binding, spec, status)
        records = [json.loads(line) for line in (args.run_dir / 'records.jsonl').read_text().splitlines()]
        for row in records:
            validate_variant_receipt(args.arm, row['receipts'], row['denoise_forward_count'],
                                     spec['primary_receipt_method'], settings)
        summary = summarize_closed_run(records, status, settings, args.preflight)
        (args.run_dir / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
        status['complete'] = True
    finally:
        status['gpu_reserved_seconds'] = round(time.perf_counter() - started, 3)
        (args.run_dir / 'terminal.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
