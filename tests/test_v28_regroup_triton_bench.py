"""CPU address/contract tests; GPU exact/FP32 oracles remain mandatory."""
import unittest
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

from scripts.v28_regroup_triton_bench import (TritonPermutation, interpretation,
    inverse_order, linear_addresses, main, validate_permutation, _kernel,
    held_report, permutation_report)


class PermutationTests(unittest.TestCase):
    def test_per_head_bijection_and_inverse(self):
        order = np.array([[4, 0, 2, 1, 3], [1, 3, 0, 4, 2]])
        validate_permutation(order, 2, 5)
        inverse = inverse_order(order)
        for head in range(2):
            self.assertTrue(np.array_equal(order[head, inverse[head]], np.arange(5)))
        for bad in (order[:, :4], order.astype(float), np.array([[0, 0, 2, 3, 4], [0, 1, 2, 3, 4]]),
                    np.array([[-1, 1, 2, 3, 4], [0, 1, 2, 3, 4]])):
            with self.assertRaises(ValueError):
                validate_permutation(bad, 2, 5)

    def test_address_gather_scatter_against_independent_numpy_indexing(self):
        # Non-power-of-two rows/heads/dim and launch tails expose axis mistakes.
        for q, h, d in ((5, 3, 7), (65, 2, 9), (1, 1, 1), (256, 16, 3)):
            order = np.stack([np.roll(np.arange(q), head+1) for head in range(h)])
            original = np.arange(q*h*d).reshape(q, h, d)
            offsets = np.arange(q*h*d+17)
            source, target, valid = linear_addresses(offsets, order, q, h, d, (h*d, d, 1))
            gathered = np.empty_like(original).reshape(-1)
            gathered[target[valid]] = original.reshape(-1)[source[valid]]
            expected = original[order.T, np.arange(h)[None, :]]
            self.assertTrue(np.array_equal(gathered.reshape(q, h, d), expected))
            source, target, valid = linear_addresses(offsets, order, q, h, d, (h*d, d, 1), True)
            restored = np.empty_like(gathered)
            restored[target[valid]] = gathered[source[valid]]
            self.assertTrue(np.array_equal(restored.reshape(q, h, d), original))
            self.assertEqual(np.unique(target[valid]).size, q*h*d)
            self.assertFalse(valid[-17:].any())

    def test_noncontiguous_source_strides_and_negative_offset_mask(self):
        q, h, d = 5, 2, 3
        order = np.array([[4, 3, 2, 1, 0], [1, 2, 3, 4, 0]])
        storage = np.arange(h*q*d).reshape(h, q, d)
        logical = storage.transpose(1, 0, 2)
        source, target, valid = linear_addresses(np.arange(-1, q*h*d+1), order, q, h, d, (d, q*d, 1))
        output = np.empty(q*h*d, dtype=int)
        output[target[valid]] = storage.reshape(-1)[source[valid]]
        self.assertTrue(np.array_equal(output.reshape(q, h, d), logical[order.T, np.arange(h)[None, :]]))
        self.assertFalse(valid[0]); self.assertFalse(valid[-1])

    def test_invalid_dimensions_strides_fail(self):
        for dim, strides in ((0, (2, 1, 1)), (3, (0, 1, 1)), (3, (1, 1))):
            with self.assertRaises(ValueError):
                linear_addresses([0], np.array([[0]]), 1, 1, dim, strides)

    def test_both_kernel_qualifications_required(self):
        backend = object.__new__(TritonPermutation)
        for gathered, scattered in ((False, False), (True, False), (False, True)):
            backend.gather_qualified, backend.scatter_qualified = gathered, scattered
            with self.assertRaises(RuntimeError):
                backend.require_qualified()
        backend.gather_qualified = backend.scatter_qualified = True
        backend.require_qualified()

    def test_kernel_numerics_are_not_model_quality_or_cross_host_margin(self):
        self.assertIn('different-host', interpretation('permutation'))
        self.assertIn('not online performance or model accuracy', interpretation('held'))

    def test_invalid_repeats_fail_before_gpu_import(self):
        with patch('scripts.v28_regroup_triton_bench.permutation_report') as gpu:
            with self.assertRaisesRegex(ValueError, 'four repeats'):
                main(['absent_private', 'unused_new_report.json', '--repeats', '3'])
            gpu.assert_not_called()

    def test_lazy_jit_receives_global_language_and_actual_constexpr_annotations(self):
        language = ModuleType('triton.language')
        language.constexpr = object()
        triton = ModuleType('triton')
        triton.language = language
        triton.jit = lambda function: function
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            kernel = _kernel()
        self.assertIs(kernel.__globals__['tl'], language)
        for parameter in ('Q', 'H', 'D', 'SQ', 'SH', 'SD', 'SCATTER', 'BLOCK'):
            self.assertIs(kernel.__annotations__[parameter], language.constexpr)
        self.assertNotIn('tl', kernel.__code__.co_freevars)

    def test_mode_defaults_and_disclosed_allocation_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode, expected in (('permutation', 100), ('held', 32)):
                with patch('scripts.v28_regroup_triton_bench.'+mode+'_report', return_value={}) as gpu:
                    result = main(['absent_private', str(Path(tmp)/(mode+'.json')), '--mode', mode])
                self.assertEqual(gpu.call_args.args[1], expected)
                self.assertEqual(result['timing_parameters']['warm'], 8)
                self.assertIn('each call allocates', result['allocation_policy'])

    def test_missing_fixed_bins_fail_before_gpu_import(self):
        with patch('scripts.v28_regroup_triton_bench.held.select_accepted', return_value=[]):
            for run in (held_report, permutation_report):
                with self.assertRaisesRegex(ValueError, 'three original held'):
                    run('absent_private', 4)


if __name__ == '__main__':
    unittest.main()
