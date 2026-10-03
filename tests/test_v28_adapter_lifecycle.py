"""Request ownership and paged-stride tests using only the Python standard library."""
import ast
from contextlib import ExitStack, contextmanager
import gc
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


class Payload:
    pass


class StorageView:
    def __init__(self, storage):
        self.storage = storage


class StubAttention:
    is_sliding = False


class StubAdapter:
    def __init__(self, layer_types):
        self.model = SimpleNamespace(modules=lambda: [StubAttention()])


def load_adapter():
    """Execute actual adapter class methods without importing torch or vLLM."""
    path = Path(__file__).resolve().parents[1] / 'experiments/numerical_qk_reuse/vllm_adapter.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ('_PrefixCache', 'VllmMethodAdapter')]
    def unexpected_sync():
        raise AssertionError('non-profile lifecycle added a GPU synchronization')
    namespace = dict(SimpleNamespace=SimpleNamespace, ExitStack=ExitStack,
                     _stub_runner=lambda: None, _StubAdapter=StubAdapter,
                     _StubAttention=StubAttention, _BINDING=None,
                     torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=unexpected_sync)))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['VllmMethodAdapter'], namespace


class AdapterLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.Adapter, self.namespace = load_adapter()

    def adapter(self, lifecycle='request_clear', arm='method'):
        return self.Adapter(['full_attention'], config={} if arm == 'method' else None,
                            condition='toy' if arm == 'method' else None,
                            arm=arm, lifecycle=lifecycle)

    def populate(self, adapter, close_error=False, counter_error=False):
        payloads = [Payload() for _ in range(5)]
        refs = [weakref.ref(value) for value in payloads]
        kv, dp, paged, canvas, split = payloads
        adapter.buffers[0] = StorageView(kv)
        adapter.cache.layers[0] = StorageView(kv)
        adapter.binding = SimpleNamespace(runtime=SimpleNamespace(
            attention_override=SimpleNamespace(dp_states={0: dp})))
        adapter.paged, adapter.canvas = paged, canvas
        adapter._split_cache.append(split)
        adapter.bound, adapter.pending_sample, adapter.step_ctx = True, True, {}
        adapter.calls['global_calls'] = 17
        closed = []
        def close():
            closed.append(True)
            if close_error:
                raise RuntimeError('toy close failure')
        def counters():
            if counter_error:
                raise RuntimeError('toy counter failure')
            return {'toy_count': 19}
        adapter._stack = ExitStack()
        adapter._stack.callback(close)
        adapter.runtime = {'counters': counters}
        return refs, closed

    def assert_cleared(self, adapter):
        for field in ('binding', 'cache', 'paged', 'canvas', '_stack', 'runtime', 'stub', 'step_ctx'):
            self.assertIsNone(getattr(adapter, field))
        self.assertFalse(adapter.buffers)
        self.assertFalse(adapter._split_cache)
        self.assertFalse(adapter._events)
        self.assertFalse(adapter.bound)
        self.assertFalse(adapter.pending_sample)

    def test_request_clear_releases_views_binding_and_split_storage(self):
        adapter = self.adapter()
        refs, closed = self.populate(adapter)
        self.assertTrue(all(ref() is not None for ref in refs))
        receipt = adapter.end_request()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(closed, [True])
        self.assertEqual(receipt['adapter']['global_calls'], 17)
        self.assertEqual(receipt['method'], {'toy_count': 19})
        self.assertIsNone(receipt['timing'])
        self.assert_cleared(adapter)

    def test_legacy_default_retains_existing_references(self):
        adapter = self.Adapter(['full_attention'], config={}, condition='toy')
        self.assertEqual(adapter.lifecycle, 'legacy')
        refs, closed = self.populate(adapter)
        receipt = adapter.end_request()
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(closed, [True])
        self.assertFalse(adapter.buffers)
        self.assertEqual(receipt['method'], {'toy_count': 19})

    def test_close_exception_still_releases_all_request_references(self):
        adapter = self.adapter()
        refs, closed = self.populate(adapter, close_error=True)
        with self.assertRaisesRegex(RuntimeError, 'close failure'):
            adapter.end_request()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(closed, [True])
        self.assert_cleared(adapter)

    def test_counter_exception_still_closes_and_releases(self):
        adapter = self.adapter()
        refs, closed = self.populate(adapter, counter_error=True)
        with self.assertRaisesRegex(RuntimeError, 'counter failure'):
            adapter.end_request()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(closed, [True])
        self.assert_cleared(adapter)

    def test_external_cache_and_binding_no_longer_retain_payload(self):
        adapter = self.adapter()
        refs, _ = self.populate(adapter)
        cache, binding = adapter.cache, adapter.binding
        adapter.end_request()
        gc.collect()
        self.assertFalse(cache.layers)
        self.assertIsNone(binding.runtime.attention_override)
        self.assertTrue(all(ref() is None for ref in refs))

    def fake_core(self):
        parent = ModuleType('experiments.numerical_qk_reuse')
        parent.__path__ = []
        v21 = ModuleType('experiments.numerical_qk_reuse.v21')
        @contextmanager
        def install(stub, config, condition):
            binding = self.namespace['_BINDING']
            binding.runtime.attention_override = SimpleNamespace(dp_states={})
            yield {'counters': lambda: {'toy_count': 0}}
        v21.install = install
        parent.v21 = v21
        return patch.dict(sys.modules, {'experiments.numerical_qk_reuse': parent,
                                      'experiments.numerical_qk_reuse.v21': v21})

    def test_two_begin_end_cycles_all_arms_preserve_receipt_snapshot(self):
        for arm in ('method', 'native', 'allkept'):
            with self.subTest(arm=arm), self.fake_core():
                adapter = self.adapter(arm=arm)
                receipts = []
                for count in (7, 11):
                    adapter.begin_request()
                    self.assertEqual(adapter.calls['global_calls'], 0)
                    self.assertEqual(adapter.cache.layers, [None])
                    self.assertFalse(adapter._split_cache)
                    adapter.calls['global_calls'] = count
                    receipts.append(adapter.end_request())
                    self.assert_cleared(adapter)
                self.assertEqual([r['adapter']['global_calls'] for r in receipts], [7, 11])
                self.assertEqual(adapter.lifecycle, 'request_clear')

    def test_opt_in_rejects_double_begin_for_controls(self):
        for arm in ('native', 'allkept'):
            with self.subTest(arm=arm):
                adapter = self.adapter(arm=arm)
                adapter.begin_request()
                with self.assertRaisesRegex(RuntimeError, 'previous request'):
                    adapter.begin_request()
                adapter.end_request()

    def test_invalid_lifecycle_rejected_before_layer_or_gpu_access(self):
        with self.assertRaisesRegex(ValueError, 'lifecycle'):
            self.Adapter(None, arm='native', lifecycle='canvas_clear')


class PagedStrideTests(unittest.TestCase):
    def test_contiguous_pages_hk2_are_not_a_flattenable_token_view(self):
        # Physical [pages, HK, PAGE, 2D], then transpose(1,2)/split(D).
        # Even with logical page_table=[0,1], head0 token63 -> token64
        # jumps over head1 of page0; a uniform token stride cannot do this.
        hk, page, d = 2, 64, 512
        def address(token, head):
            return (token // page) * hk * page * 2 * d + head * page * 2 * d + (token % page) * 2 * d
        self.assertEqual(address(63, 0) - address(62, 0), 1024)
        self.assertEqual(address(64, 0) - address(63, 0), 66560)
        self.assertNotEqual(address(63, 0) + 1024, address(64, 0))
        self.assertEqual(address(63, 0) + 1024, address(0, 1))

    def test_single_head_special_case_does_have_uniform_token_stride(self):
        page, d = 64, 512
        def address(token):
            return (token // page) * page * 2 * d + (token % page) * 2 * d
        self.assertEqual(address(64) - address(63), address(63) - address(62))


if __name__ == '__main__':
    unittest.main()
