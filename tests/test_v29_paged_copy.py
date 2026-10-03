import unittest
from unittest.mock import patch
from types import SimpleNamespace
from experiments.numerical_qk_reuse.v29_paged_copy import validate_geometry, cpu_address


class PagedCopyTests(unittest.TestCase):
    shape = (9, 64, 2, 8)
    strides = (2048, 16, 1024, 1)  # physical [page, head, slot, K+V]

    def test_native_interleaved_layout(self):
        self.assertEqual(validate_geometry(self.shape, self.strides, self.strides,
                                          (1, 2, 321, 8), 6, 65, 256), (64, 2, 8, 321))

    def test_random_pages_heads_and_boundary(self):
        table = [7, 2, 8, 0, 4, 1]
        for token in (0, 63, 64, 65, 127, 128, 255, 256, 320):
            for head in range(2):
                for d in range(8):
                    src, dst = cpu_address(token, head, d, table, 64, self.strides, 321, 2, 8)
                    self.assertEqual(src, ((table[token // 64] * 2 + head) * 64 + token % 64) * 16 + d)
                    self.assertEqual(dst, (head * 321 + token) * 8 + d)

    def test_tail_destination_keeps_prefix(self):
        positions = {cpu_address(t, h, d, [2, 0, 1], 64, self.strides, 129, 2, 8)[1]
                     for t in range(65, 129) for h in range(2) for d in range(8)}
        self.assertEqual(len(positions), 64 * 2 * 8)
        self.assertFalse(any((h * 129 + t) * 8 + d in positions
                             for t in range(65) for h in range(2) for d in range(8)))

    def test_full_range(self):
        validate_geometry(self.shape, self.strides, self.strides, (1, 2, 129, 8), 3, 0, 129)

    def test_strided_kv_supported(self):
        validate_geometry(self.shape, self.strides, (4096, 32, 2048, 2), (1, 2, 128, 8), 2, 0, 128)

    def test_out_of_range(self):
        for start, count in ((-1, 2), (0, 0), (100, 30), (129, 1)):
            with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                validate_geometry(self.shape, self.strides, self.strides, (1, 2, 129, 8), 3, start, count)

    def test_bad_table(self):
        with self.assertRaises(ValueError):
            validate_geometry(self.shape, self.strides, self.strides, (1, 2, 129, 8), 2, 0, 129)

    def test_bad_layout(self):
        for target in ((2, 2, 129, 8), (1, 1, 129, 8), (1, 2, 129, 9)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_geometry(self.shape, self.strides, self.strides, target, 3, 0, 129)

    def test_bad_strides(self):
        with self.assertRaises(ValueError):
            validate_geometry(self.shape, (0, 16, 1024, 1), self.strides, (1, 2, 129, 8), 3, 0, 129)


class AdapterCopyTests(unittest.TestCase):
    def test_unknown_backend_rejected(self):
        from tests.test_v28_adapter_lifecycle import load_adapter
        adapter, _ = load_adapter()
        with self.assertRaises(ValueError):
            adapter(['full_attention'], arm='allkept', kv_copy_backend='guess')

    def test_full_then_tail_counter_and_storage(self):
        from tests.test_v28_adapter_lifecycle import load_adapter
        from tests.test_v28_canvas_release import Tensor
        adapter, namespace = load_adapter()
        trace, copies = [], []
        def empty(shape, **kwargs):
            return Tensor(shape, trace)
        namespace['torch'].empty = empty
        namespace['torch'].empty_like = lambda x: empty(x.shape)
        a = adapter(['full_attention'], arm='allkept', kv_copy_backend='triton')
        key, value, table = Tensor((9, 64, 2, 8), trace), Tensor((9, 64, 2, 8), trace), Tensor((6,), trace)
        fake = SimpleNamespace(copy_paged_kv=lambda *args: copies.append(args))
        with patch.dict('sys.modules', {'experiments.numerical_qk_reuse.v29_paged_copy': fake}):
            b = a._buffers(0, key, value, table, 65, 256)
            self.assertIs(a._buffers(0, key, value, table, 65, 256), b)
        self.assertEqual([args[-2:] for args in copies], [(0, 321), (65, 256)])
        self.assertEqual(a.calls['triton_kv_copy_calls'], 2)
        self.assertEqual(a.calls['triton_kv_copy_elements'], 2 * 2 * 8 * (321 + 256))
        self.assertEqual(a.calls['prefix_copies'], 1)
        self.assertEqual(a.calls['canvas_refreshes'], 1)
        self.assertEqual(trace, [])


if __name__ == '__main__':
    unittest.main()
