"""CPU-only qualification guards; no torch/vLLM import or GPU execution."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import v28_vllm_qualify as q


def check(layer=5):
    return dict(layer=layer, max_abs_error=0., page_size=64, kv_heads=2)


def allocator_snapshot():
    return dict(num_alloc_retries=1, num_ooms=0, allocated_bytes=100,
                reserved_bytes=200, peak_allocated_bytes=150, peak_reserved_bytes=250)


def receipt(block=64):
    settings = q.execution_settings(block, 'request_clear')
    return dict(adapter=dict(order_errors=0, begins=10, observes=10, global_calls=50,
                             split_fa4_calls=35, v28_config=settings, kv_probe_timed_checks=0,
                             allocator=q.allocator_receipt(allocator_snapshot(), allocator_snapshot()),
                             kv_layout_checks=[check(layer) for layer in (5, 11, 17, 23, 29)]),
                method=dict(effective_method=dict(q_block=block, q_regroup=False, q_carry64=False),
                            active_layers=[5, 11, 17, 23, 29], gated_native_calls=0,
                            layer_native_calls=0, unsupported_mask_refreshes=0,
                            fused_observations=5, dp_routes=5,
                            q64_refined_routes=5, q64_list_builds=5))


class QualificationGuards(unittest.TestCase):
    def inputs(self, block=64):
        root = Path(q.__file__).resolve().parents[1]
        binding = dict(files={str(path): 'toy_hash' for path in
                             (Path(q.__file__).resolve(), Path(q.panel.__file__).resolve(),
                              root / 'scripts/v27_vllm_metrics.py',
                              root / 'experiments/numerical_qk_reuse/vllm_adapter.py')})
        required = dict(q_block=block, score_period=64, decision_interval=6,
                        risk_state='dense_prefix', carry_first=True, fused_observe=True,
                        async_route=True, consumer='fa4')
        spec = dict(protocol_id='v28_toy', arms=['dense', 'method'], controls=['native', 'allkept'],
                    adapter_settings=dict(lifecycle='request_clear', alias_splits=2),
                    primary_receipt_method=required)
        config = dict(q_block=64) if block == 64 else {}
        return binding, spec, config

    def test_q128_default_config_and_explicit_q64_frozen(self):
        for block in (128, 64):
            with self.subTest(block=block):
                inputs = self.inputs(block)
                before = copy.deepcopy(inputs)
                self.assertEqual(q.validate_frozen_inputs(*inputs, block, 'request_clear')['query_block'], block)
                self.assertEqual(inputs, before)

    def test_old_protocol_and_unpinned_wrapper_rejected(self):
        binding, spec, config = self.inputs()
        spec['protocol_id'] = 'v27_vllm_lb_v18b'
        with self.assertRaisesRegex(ValueError, 'new frozen'):
            q.validate_frozen_inputs(binding, spec, config, 64, 'request_clear')
        binding, spec, config = self.inputs()
        del binding['files'][str(Path(q.__file__).resolve())]
        with self.assertRaisesRegex(ValueError, 'pin'):
            q.validate_frozen_inputs(binding, spec, config, 64, 'request_clear')

    def test_q64_drift_regroup_and_changed_method_rejected(self):
        for target, field, value in [('config', 'q_block', 128), ('config', 'q_regroup', True),
                                     ('config', 'q_carry64', True), ('required', 'score_period', 8)]:
            binding, spec, config = self.inputs()
            (config if target == 'config' else spec['primary_receipt_method'])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.validate_frozen_inputs(binding, spec, config, 64, 'request_clear')

    def test_lifecycle_alias2_match_is_frozen(self):
        for lifecycle, splits in [('legacy', 2), ('request_clear', 1)]:
            binding, spec, config = self.inputs()
            spec['adapter_settings'].update(lifecycle=lifecycle, alias_splits=splits)
            with self.subTest(lifecycle=lifecycle), self.assertRaises(ValueError):
                q.validate_frozen_inputs(binding, spec, config, 64, 'request_clear')

    def test_original_receipt_validation_and_q64_counters(self):
        settings = q.execution_settings(64, 'request_clear')
        q.validate_variant_receipt('method', receipt(), 10, {'q_block': 64}, settings)
        for field in ('q64_refined_routes', 'q64_list_builds'):
            rec = receipt()
            rec['method'][field] = 0
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'q64 refined'):
                q.validate_variant_receipt('method', rec, 10, {'q_block': 64}, settings)
        with self.assertRaisesRegex(ValueError, 'clock'):
            q.validate_variant_receipt('method', receipt(), 9, {'q_block': 64}, settings)

    def test_probe_boundary_numerics_and_layer_coverage(self):
        settings = q.execution_settings(64, 'request_clear')
        for kind in ('timed', 'missing_layer', 'error', 'wrong_page'):
            rec = receipt()
            if kind == 'timed': rec['adapter']['kv_probe_timed_checks'] = 1
            if kind == 'missing_layer': rec['adapter']['kv_layout_checks'].pop()
            if kind == 'error': rec['adapter']['kv_layout_checks'][0]['max_abs_error'] = .1
            if kind == 'wrong_page': rec['adapter']['kv_layout_checks'][0]['page_size'] = 32
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                q.validate_variant_receipt('method', rec, 10, {'q_block': 64}, settings)

    def test_dense_native_and_allkept_controls(self):
        settings = q.execution_settings(128, 'request_clear')
        q.validate_variant_receipt('dense', None, 10, {}, settings)
        for arm in ('native', 'allkept'):
            rec = receipt(128)
            rec['method'] = None
            if arm == 'native': rec['adapter']['kv_layout_checks'] = []
            q.validate_variant_receipt(arm, rec, 10, {}, settings)

    def test_sample_boundaries_and_hk2_stride_counterexample(self):
        self.assertEqual(q.sample_tokens(65, 3, 64), [0, 63, 64, 65, 66, 67])
        self.assertFalse(q.affine_token_stride((4, 64, 2, 512), (131072, 1024, 65536, 1)))
        self.assertTrue(q.affine_token_stride((4, 64, 1, 512), (65536, 1024, 65536, 1)))
        with self.assertRaises(ValueError): q.sample_tokens(-1, 3, 64)

    def status_and_row(self):
        status = dict(protocol_id='v28_toy', arm='method', run_id='toy', completed_timed=1, expected_timed=1)
        row = dict(protocol_id='v28_toy', arm='method', run_id='toy', qualification_only=True,
                   execution_count_source='vllm_existing_async_cpu_snapshot', denoise_forward_count=10,
                   scheduler_denoise_forward_count=9, speculative_unused_denoising=1,
                   commit_forward_count=2, graph_captures_timed=0, wall_s=3., prefill_s=1., decode_span_s=2.,
                   compilation_deltas=dict(num_backend_compilations=0, num_inductor_compiles=0), receipts=receipt())
        return status, row

    def test_closed_summary_actual_N_and_private_field_exclusion(self):
        status, row = self.status_and_row()
        row.update(id='private_toy', completion='private_toy', prompt='private_toy')
        result = q.summarize_closed_run([row], status, q.execution_settings(64, 'request_clear'), True)
        self.assertEqual(result['denoise_forward_count_sum'], 10)
        self.assertEqual(result['scheduler_N_sum'], 9)
        self.assertEqual(result['speculative_unused_N_sum'], 1)
        self.assertNotIn('private_toy', str(result))

    def test_closed_summary_rejects_count_capture_timing_and_identity_drift(self):
        for field, value in [('qualification_only', False), ('run_id', 'other'),
                             ('execution_count_source', 'retired'), ('denoise_forward_count', 9),
                             ('scheduler_denoise_forward_count', -1), ('graph_captures_timed', None),
                             ('graph_captures_timed', 1), ('wall_s', 0), ('decode_span_s', float('nan'))]:
            status, row = self.status_and_row()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.summarize_closed_run([row], status, q.execution_settings(64, 'request_clear'), True)
        status, row = self.status_and_row()
        row['compilation_deltas'] = {}
        with self.assertRaises(ValueError):
            q.summarize_closed_run([row], status, q.execution_settings(64, 'request_clear'), True)
        status['completed_timed'] = 0
        with self.assertRaises(ValueError):
            q.summarize_closed_run([row], status, q.execution_settings(64, 'request_clear'), True)

    def test_instrumentation_restores_originals_on_exception(self):
        module = SimpleNamespace(VllmMethodAdapter=FakeAdapter)
        original = q.panel.validate_receipts
        with self.assertRaisesRegex(RuntimeError, 'toy failure'):
            with q.instrument_runner(module, q.execution_settings(64, 'request_clear'), 1):
                self.assertIsNot(module.VllmMethodAdapter, FakeAdapter)
                raise RuntimeError('toy failure')
        self.assertIs(module.VllmMethodAdapter, FakeAdapter)
        self.assertIs(q.panel.validate_receipts, original)

    def test_adapter_probe_only_first_warm_and_explicit_receipt_settings(self):
        settings = q.execution_settings(64, 'request_clear')
        cls = q.adapter_variant(FakeAdapter, settings, 2)
        adapter = cls(arm='method')
        probe_result = check()
        del probe_result['layer']
        with patch.object(q, 'check_actual_copy', return_value=probe_result) as probe, \
                patch.object(q, 'allocator_snapshot', side_effect=allocator_snapshot) as memory:
            for request in range(4):
                adapter.begin_request()
                for layer in (5, 11, 17, 23, 29):
                    adapter.forward(SimpleNamespace(head_size=512), layer, None, None,
                                    SimpleNamespace(block_table=[[0]]), None)
                rec = adapter.end_request()
            self.assertEqual(probe.call_count, 5)
            self.assertEqual(memory.call_count, 8)
        self.assertEqual(rec['adapter']['v28_config'], settings)
        self.assertEqual(rec['adapter']['kv_probe_timed_checks'], 0)
        self.assertEqual(rec['adapter']['allocator']['num_alloc_retries_delta'], 0)
        self.assertEqual(rec['method']['effective_method'], {'q_block': 64})
        with self.assertRaises(ValueError): q.adapter_variant(FakeAdapter, settings, 0)

    def test_actual_copy_probe_random_pages_and_cross_head_mismatch(self):
        kv = FakeKV()
        table = FakeVector([3, 0, 2])
        fake_torch = SimpleNamespace(stack=lambda values: FakeVector(values))
        buffer = dict(nk=131, k=FakeBuffer(kv, table, False), v=FakeBuffer(kv, table, True))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            result = q.check_actual_copy(kv, buffer, table, 128, 3, 512)
            self.assertEqual(result['max_abs_error'], 0)
            self.assertFalse(result['logical_pages_consecutive'])
            self.assertFalse(result['affine_token_view_possible'])
            buffer['v'].wrong_head = True
            with self.assertRaisesRegex(ValueError, 'numerical mismatch'):
                q.check_actual_copy(kv, buffer, table, 128, 3, 512)

    def test_allocator_metadata_only_snapshot_and_monotonic_deltas(self):
        raw = dict(num_alloc_retries=2, num_ooms=1,
                   **{'allocated_bytes.all.current': 100, 'reserved_bytes.all.current': 200,
                      'allocated_bytes.all.peak': 150, 'reserved_bytes.all.peak': 250})
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(memory_stats=lambda: raw))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            snapshot = q.allocator_snapshot()
            before = allocator_snapshot()
            result = q.allocator_receipt(before, snapshot)
            self.assertEqual(result['num_alloc_retries_delta'], 1)
            self.assertEqual(result['num_ooms_delta'], 1)
            del raw['num_ooms']
            with self.assertRaises(ValueError): q.allocator_snapshot()
        after = allocator_snapshot()
        after['num_alloc_retries'] = 0
        with self.assertRaises(ValueError): q.allocator_receipt(allocator_snapshot(), after)
        with self.assertRaises(ValueError): q.allocator_receipt({}, {})


class FakeAdapter:
    def __init__(self, arm='method', lifecycle='legacy'):
        self.arm, self.lifecycle, self.splits = arm, lifecycle, 2
        self.buffers = {}
        self.step_ctx = dict(seq_len=65, n=1)
    def begin_request(self):
        pass
    def forward(self, impl, layer, query, kv, metadata, output):
        self.buffers[layer] = {}
        return output
    def end_request(self):
        return dict(adapter={}, method=dict(effective_method={'q_block': 64}))


class FakeVector:
    def __init__(self, values): self.values = list(values)
    def __getitem__(self, index):
        return FakeVector(self.values[index]) if isinstance(index, slice) else self.values[index]
    def long(self): return self
    def float(self): return self
    def __sub__(self, other): return FakeVector(a - b for a, b in zip(self.values, other.values))
    def __eq__(self, other): return FakeVector(value == other for value in self.values)
    def abs(self): return FakeVector(abs(value) for value in self.values)
    def max(self): return SimpleNamespace(item=lambda: max(self.values))
    def all(self): return SimpleNamespace(item=lambda: all(self.values))


class FakeKV:
    ndim = 4
    shape = (4, 2, 64, 1024)
    def stride(self): return (131072, 65536, 1024, 1)
    def transpose(self, first, second):
        if (first, second) != (1, 2): raise AssertionError('incorrect native layout')
        return self
    def split(self, size, dim):
        if (size, dim) != (512, -1): raise AssertionError('incorrect KV split')
        view = SimpleNamespace(shape=(4, 64, 2, 512), stride=lambda: (131072, 1024, 65536, 1))
        return view, view
    def __getitem__(self, index):
        page, head, slot, channel = index
        return page * 1000000 + head * 100000 + slot * 1000 + channel


class FakeBuffer:
    shape = (1, 2, 131, 512)
    def __init__(self, kv, table, value):
        self.kv, self.table, self.value, self.wrong_head = kv, table, value, False
    def __getitem__(self, index):
        _, head, token, channel = index
        if self.wrong_head: head = 1 - head
        return self.kv[self.table[token // 64], head, token % 64, channel + 512 * self.value]


if __name__ == '__main__':
    unittest.main()
