"""Canvas ownership/stream guards with public stdlib fakes, no GPU imports."""
import gc
import math
from types import SimpleNamespace
import unittest
import weakref

from tests.test_v28_adapter_lifecycle import load_adapter


class Tensor:
    def __init__(self, shape, trace, owner=None):
        self.shape, self.trace, self.owner = tuple(shape), trace, owner
        self.device, self.dtype = 'cuda:0', 'toy_bf16'

    def __getitem__(self, index):
        shape = list(self.shape)
        if isinstance(index, int):
            shape = shape[1:]
        elif isinstance(index, tuple):
            for dim, part in enumerate(index):
                if isinstance(part, slice) and part.stop is not None:
                    shape[dim] = min(shape[dim], part.stop)
        return Tensor(shape, self.trace, self)

    def long(self):
        return self

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // math.prod(x for x in shape if x != -1)
        return Tensor(shape, self.trace, self)

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return Tensor(shape, self.trace, self)

    def copy_(self, other):
        self.trace.append('copy')
        return self

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2


class CanvasReleaseTests(unittest.TestCase):
    def setUp(self):
        self.Adapter, self.namespace = load_adapter()
        self.trace = []
        self.stream = SimpleNamespace(device='cuda:0', cuda_stream=10)

        def current_stream(device=None):
            self.trace.append('stream')
            return self.stream

        def empty(shape, **kwargs):
            self.trace.append('allocate')
            return Tensor(shape, self.trace)

        cuda = self.namespace['torch'].cuda
        cuda.current_stream = current_stream
        self.namespace['torch'].empty = empty
        self.namespace['torch'].empty_like = lambda tensor: empty(tensor.shape)

    def adapter(self, arm='method', canvas_buffers='release_after_invalidate', lifecycle='legacy'):
        return self.Adapter(['full_attention', 'full_attention'],
                            config={} if arm == 'method' else None,
                            condition='toy' if arm == 'method' else None,
                            arm=arm, lifecycle=lifecycle, canvas_buffers=canvas_buffers)

    def buffers(self, adapter, layer=0, prefix=64):
        key = Tensor((3, 64, 2, 512), self.trace)
        value = Tensor((3, 64, 2, 512), self.trace)
        table = Tensor((3,), self.trace)
        b = adapter._buffers(layer, key, value, table, prefix, 64)
        adapter.cache.layers[layer] = SimpleNamespace(keys=b['pk'], values=b['pv'])
        return weakref.ref(b['k']), weakref.ref(b['v'])

    def encoder(self, adapter, callback=None):
        def invalidate():
            self.trace.append('invalidate')
            if callback:
                callback()
        adapter.stub = SimpleNamespace(model=SimpleNamespace(encoder=invalidate))

    def prepare(self, adapter):
        adapter.on_prepare(True, 0, 128, 64)

    def test_release_after_hook_preserves_layer_list_and_drops_prefix_views(self):
        adapter = self.adapter()
        refs = self.buffers(adapter)
        layer_list = adapter.cache.layers
        source = {'k': adapter.buffers[0]['pk'], 'v': adapter.buffers[0]['pv']}

        def hook():
            self.assertTrue(adapter.buffers)
            self.assertIsNotNone(adapter.cache.layers[0])
            self.assertTrue(all(ref() is not None for ref in refs))
            source.clear()  # the core and Sketches encoder hooks clear these owners

        self.encoder(adapter, hook)
        self.trace.clear()
        self.prepare(adapter)
        gc.collect()
        self.assertEqual(self.trace, ['invalidate', 'stream'])
        self.assertIs(adapter.cache.layers, layer_list)
        self.assertEqual(layer_list, [None, None])
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertFalse(adapter.buffers)
        self.assertFalse(adapter._buffer_streams)
        self.assertEqual(adapter.calls['canvas_release_calls'], 1)
        self.assertEqual(adapter.calls['canvas_release_layers'], 1)
        self.assertEqual(adapter.calls['canvas_release_bytes'], 2 * 1 * 2 * 128 * 512 * 2)
        self.assertEqual(adapter.calls['canvas_release_epoch'], 1)

    def test_failed_invalidate_retains_every_owner_and_epoch(self):
        adapter = self.adapter()
        refs = self.buffers(adapter)

        def failure():
            raise RuntimeError('toy invalidate failed')

        self.encoder(adapter, failure)
        self.trace.clear()
        with self.assertRaisesRegex(RuntimeError, 'invalidate failed'):
            self.prepare(adapter)
        self.assertEqual(self.trace, ['invalidate'])
        self.assertTrue(adapter.buffers)
        self.assertIsNotNone(adapter.cache.layers[0])
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(adapter._canvas_invalidate_epoch, 0)
        self.assertEqual(adapter.calls['canvas_release_calls'], 0)

    def test_stream_switch_rejects_all_layers_before_any_release(self):
        adapter = self.adapter()
        refs = self.buffers(adapter, 0) + self.buffers(adapter, 1)
        adapter._buffer_streams[1] = ('cuda:0', 11)
        self.encoder(adapter)
        with self.assertRaisesRegex(RuntimeError, 'stream changed'):
            self.prepare(adapter)
        self.assertEqual(set(adapter.buffers), {0, 1})
        self.assertTrue(all(source is not None for source in adapter.cache.layers))
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(adapter.calls['canvas_release_calls'], 0)

    def test_device_switch_also_rejects(self):
        adapter = self.adapter()
        self.buffers(adapter)
        self.encoder(adapter)
        self.stream.device = 'cuda:1'
        with self.assertRaisesRegex(RuntimeError, 'stream changed'):
            self.prepare(adapter)
        self.assertTrue(adapter.buffers)

    def test_legacy_has_no_stream_queries_new_counters_or_canvas_release(self):
        adapter = self.adapter(canvas_buffers='legacy')
        refs = self.buffers(adapter)
        self.encoder(adapter)
        self.trace.clear()
        self.prepare(adapter)
        self.assertEqual(self.trace, ['invalidate'])
        self.assertTrue(adapter.buffers)
        self.assertIsNotNone(adapter.cache.layers[0])
        self.assertFalse(adapter._buffer_streams)
        self.assertNotIn('canvas_release_calls', adapter.calls)
        self.assertTrue(all(ref() is not None for ref in refs))

    def test_each_arm_accepts_common_setting_and_releases(self):
        for arm in ('method', 'allkept', 'native'):
            with self.subTest(arm=arm):
                adapter = self.adapter(arm=arm)
                refs = self.buffers(adapter)
                if arm == 'method':
                    self.encoder(adapter)
                self.prepare(adapter)
                self.assertEqual(adapter.canvas_buffers, 'release_after_invalidate')
                self.assertFalse(adapter.buffers)
                self.assertEqual(adapter.calls['canvas_release_calls'], 1)
                self.assertTrue(all(ref() is None for ref in refs))

    def test_consecutive_canvases_rebuild_with_fresh_prefix_and_stream_metadata(self):
        adapter = self.adapter()
        self.encoder(adapter)
        for prefix in (64, 128):
            refs = self.buffers(adapter, prefix=prefix)
            self.assertEqual(adapter._buffer_streams, {0: ('cuda:0', 10)})
            self.assertEqual(adapter.buffers[0]['prefix'], prefix)
            self.prepare(adapter)
            self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(adapter.calls['canvas_release_calls'], 2)
        self.assertEqual(adapter.calls['prefix_copies'], 2)
        self.assertEqual(adapter.calls['canvas_invalidate_epoch'], 2)
        self.assertEqual(adapter.calls['canvas_release_epoch'], 2)

    def test_empty_encoder_no_allocation_or_stream_query(self):
        adapter = self.adapter(arm='native')
        self.prepare(adapter)
        self.assertFalse(self.trace)
        self.assertEqual(adapter.calls['canvas_release_calls'], 0)
        self.assertEqual(adapter.calls['canvas_invalidate_epoch'], 1)

    def test_carry_and_split_maps_are_preserved(self):
        adapter = self.adapter()
        self.buffers(adapter)
        carry, split, canvas = object(), object(), object()
        adapter.binding = SimpleNamespace(runtime=SimpleNamespace(
            attention_override=SimpleNamespace(_carry={0: carry})))
        adapter._split_cache.append(split)
        adapter.canvas = canvas
        self.encoder(adapter)
        self.prepare(adapter)
        self.assertIs(adapter.binding.runtime.attention_override._carry[0], carry)
        self.assertEqual(adapter._split_cache, [split])
        self.assertIs(adapter.canvas, canvas)

    def test_cached_buffer_rejects_different_reader_stream_before_copy(self):
        adapter = self.adapter()
        self.buffers(adapter)
        self.stream.cuda_stream = 11
        self.trace.clear()
        with self.assertRaisesRegex(RuntimeError, 'refusing use'):
            self.buffers(adapter)
        self.assertEqual(self.trace, ['stream'])

    def test_untracked_cache_owner_rejected_without_partial_release(self):
        adapter = self.adapter()
        self.buffers(adapter)
        adapter.cache.layers[1] = object()
        self.encoder(adapter)
        with self.assertRaisesRegex(RuntimeError, 'no guarded buffer'):
            self.prepare(adapter)
        self.assertTrue(adapter.buffers)
        self.assertTrue(all(source is not None for source in adapter.cache.layers))

    def test_no_invalidate_or_repeated_epoch_rejected(self):
        adapter = self.adapter(arm='native')
        with self.assertRaisesRegex(RuntimeError, 'fresh successful'):
            adapter._release_canvas_buffers(0)
        self.prepare(adapter)
        with self.assertRaisesRegex(RuntimeError, 'fresh successful'):
            adapter._release_canvas_buffers(1)

    def test_end_request_metadata_reset_and_receipt_counters_retained(self):
        for lifecycle in ('legacy', 'request_clear'):
            with self.subTest(lifecycle=lifecycle):
                adapter = self.adapter(arm='native', lifecycle=lifecycle)
                self.buffers(adapter)
                self.prepare(adapter)
                self.buffers(adapter, prefix=128)
                receipt = adapter.end_request()
                self.assertFalse(adapter._buffer_streams)
                self.assertEqual(adapter._canvas_invalidate_epoch, 0)
                self.assertIsNone(adapter._canvas_release_epoch)
                self.assertEqual(receipt['adapter']['canvas_release_calls'], 1)
                adapter.begin_request()
                self.assertEqual(adapter.calls['canvas_release_calls'], 0)

    def test_close_error_also_resets_metadata(self):
        adapter = self.adapter(arm='native')
        self.buffers(adapter)

        def close():
            raise RuntimeError('toy close failed')

        adapter._stack = SimpleNamespace(close=close)
        with self.assertRaisesRegex(RuntimeError, 'close failed'):
            adapter.end_request()
        self.assertFalse(adapter._buffer_streams)
        self.assertEqual(adapter._canvas_invalidate_epoch, 0)

    def test_bad_setting_rejected_before_layer_or_device_access(self):
        with self.assertRaisesRegex(ValueError, 'canvas_buffers'):
            self.Adapter(None, arm='native', canvas_buffers='automatic')
        self.assertFalse(self.trace)


if __name__ == '__main__':
    unittest.main()
