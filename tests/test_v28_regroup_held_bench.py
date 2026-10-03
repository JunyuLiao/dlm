"""CPU-only checks; no torch or GPU required."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.v28_regroup_held_bench import (SEARCH, break_even, nominal_bin,
    gather_query, original_row_support, qualified_error, select_accepted, support_for_order)
from scripts.v28_regroup_screen import natural_order, screen_need


def toy(pt=10):
    need = np.zeros((1, 128, pt), dtype=bool)
    need[0, 0, :5] = True
    need[0, 1, 5:10] = True
    return need


def save(path, need):
    h, q, pt = need.shape
    np.savez_compressed(path, need=np.packbits(need, -1), pt=np.array(pt),
                        kept128=np.packbits(need.reshape(h, q // 128, 128, pt).any(2), -1))


class HeldRegroupTests(unittest.TestCase):
    def test_gather_operates_in_token_major_layout_and_preserves_per_head_orders(self):
        # Exercise the real helper through a NumPy-backed torch subset, keeping
        # the CPU suite independent of torch while observing dim and strides.
        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            @property
            def T(self):
                return Tensor(self.array.T)

            def __getitem__(self, index):
                return Tensor(self.array[index])

            def transpose(self, left, right):
                return Tensor(self.array.swapaxes(left, right))

            def expand_as(self, target):
                return Tensor(np.broadcast_to(self.array, target.array.shape))

        observed = []

        def gather(input_tensor, dim, index):
            observed.append((dim, input_tensor.array.shape))
            return Tensor(np.ascontiguousarray(np.take_along_axis(input_tensor.array, index.array, dim)))

        token_major = np.arange(1 * 128 * 2 * 3).reshape(1, 128, 2, 3)
        order = np.stack((np.roll(np.arange(128), 1), np.roll(np.arange(128), 5)))
        with patch.dict(sys.modules, {'torch': SimpleNamespace(gather=gather)}):
            result = gather_query(Tensor(token_major.swapaxes(1, 2)), Tensor(order))
        expected = token_major[0, order.T, np.arange(2)[None, :]][None]
        self.assertEqual(observed, [(1, (1, 128, 2, 3))])
        self.assertTrue(np.array_equal(result.array.swapaxes(1, 2), expected))
        self.assertTrue(result.array.swapaxes(1, 2).flags.c_contiguous)

    def test_historical_nominal_bins_tolerate_generated_canvas_growth(self):
        for pt, expected in ((480, 32768), (544, 32768), (1117, 65536), (1459, 98304)):
            self.assertEqual(nominal_bin(np.zeros((1, 256, pt), dtype=bool)), expected)

    def test_full_canvas_and_original_order_reference_preserve_needed_bits(self):
        need = toy()
        result = screen_need(need, SEARCH)
        self.assertTrue(result['accepted'])
        order = result['orders']['gated']
        groups = support_for_order(need, order)
        self.assertEqual(groups.shape, (1, 2, 12))
        self.assertTrue(groups[..., 10:].all())
        restored = original_row_support(need, order)
        self.assertFalse(np.any(need & ~restored[..., :10]))
        permuted = np.repeat(groups, 64, 1)
        self.assertTrue(np.array_equal(restored[np.arange(1)[:, None], order], permuted))
        # The two independent union-mask references need not be equal.
        self.assertFalse(np.array_equal(restored, original_row_support(need, natural_order(need))))

    def test_scatter_inverse_is_per_head_and_does_not_transpose_heads(self):
        need = np.tile(toy(), (2, 1, 1))
        need[1] = np.roll(need[1], 64, 0)
        order = screen_need(need, SEARCH)['orders']['gated']
        original = np.arange(2 * 128 * 3).reshape(2, 128, 3)
        gathered = original[np.arange(2)[:, None], order]
        output_layout = gathered.transpose(1, 0, 2)
        restored = np.empty_like(output_layout)
        restored[order.T, np.arange(2)[None, :]] = output_layout
        self.assertTrue(np.array_equal(restored, original.transpose(1, 0, 2)))

    def test_selection_accepts_at_most_one_each_bin_and_skips_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Paths are private implementation details and never returned.
            for i in range(4):
                save(root / f'private_{i}.npz', toy())
            with patch('scripts.v28_regroup_held_bench.nominal_bin', side_effect=[32768, 32768, 65536, 65536]):
                picked = select_accepted(root)
            self.assertEqual([entry[0] for entry in picked], [32768, 65536])
            self.assertTrue(all(result['accepted'] for _, _, result in picked))
            with patch('scripts.v28_regroup_held_bench.screen_need', return_value={'accepted': False}):
                self.assertEqual(select_accepted(root), [])
            with self.assertRaisesRegex(ValueError, 'no validated'):
                select_accepted(root / 'absent')

    def test_moresearch_gate_is_fixed_and_not_relaxed(self):
        self.assertEqual(SEARCH.max_swaps_per_head, 12)
        self.assertEqual(SEARCH.candidates_per_group, 16)
        self.assertEqual(SEARCH.min_proxy_gain, .05)
        self.assertEqual(SEARCH.max_total_increase, .01)
        self.assertEqual(SEARCH.max_moved_fraction, .125)

    def test_optimistic_break_even_arithmetic_and_nonfinite_rejection(self):
        passed = break_even(1, .7, .2)
        self.assertTrue(passed['optimistic_break_even'])
        self.assertAlmostEqual(passed['optimistic_margin_ms'], .1)
        self.assertFalse(break_even(1, .9, .2)['optimistic_break_even'])
        self.assertFalse(break_even(1, 1.1, .01)['optimistic_break_even'])
        for invalid in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                break_even(1, invalid, .2)

    def test_numerical_error_guard_rejects_nonfinite_zero_and_excess_error(self):
        self.assertEqual(qualified_error(.01, 1)['relative_max'], .01)
        for error, reference in ((float('nan'), 1), (float('inf'), 1),
                                 (.01, float('nan')), (.01, float('inf')),
                                 (.01, 0), (-.01, 1), (.03, 1)):
            with self.assertRaises(RuntimeError):
                qualified_error(error, reference)


if __name__ == '__main__':
    unittest.main()
