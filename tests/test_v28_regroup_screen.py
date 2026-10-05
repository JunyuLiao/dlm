"""CPU checks for support preservation and explicitly non-timing alias2 proxies."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.v28_regroup_screen import (Options, aggregate_results, alias2_loads,
    cost_from_counts, gate_candidate, group_support, load_need, main, natural_order,
    optimize_local_swaps, screen_need, validate_need, validate_order)


def unbalanced_need(heads=1, queries=128):
    need = np.zeros((heads, queries, 10), dtype=bool)
    # Two distinct rare rows make group 0 heavy. A swap into group 1 balances
    # both alias2 loads without adding any tile or changing the fixed grid.
    need[:, 0, :5] = True
    need[:, 1, 5:] = True
    return need


def save_dump(path, row_need, **overrides):
    h, q, pt = row_need.shape
    values = dict(need=np.packbits(row_need, axis=-1), pt=np.array(pt),
                  kept128=np.packbits(row_need.reshape(h, q // 128, 128, pt).any(2), axis=-1))
    values.update(overrides)
    np.savez_compressed(path, **values)


class Alias2ScreenTests(unittest.TestCase):
    def test_alias2_cost_matches_floor_and_ceil_including_empty_and_odd_lists(self):
        counts = np.array([[0, 1, 2, 5]], dtype=np.int64)
        self.assertEqual(alias2_loads(counts).tolist(), [0, 0, 1, 2, 0, 1, 1, 3])
        cost = cost_from_counts(counts)
        self.assertEqual(cost['total_tiles'], 8)
        self.assertEqual(cost['cta_count'], 8)
        self.assertEqual(cost['max_cta_tiles'], 3)
        self.assertEqual(cost['p95_cta_tiles'], 3)
        self.assertEqual(cost['balanced_proxy_cost'], 3)
        self.assertEqual(cost_from_counts(counts, sm_count=1)['balanced_proxy_cost'], 8)

    def test_q64_baseline_is_the_natural_union_cost(self):
        need = unbalanced_need(heads=2, queries=256)
        self.assertEqual(group_support(need, natural_order(need)).sum(-1).tolist(),
                         [[10, 0, 0, 0], [10, 0, 0, 0]])
        result = screen_need(need)
        self.assertEqual(result['costs']['q64']['total_tiles'], 20)
        self.assertEqual(result['costs']['q64']['cta_count'], 16)
        self.assertEqual(result['costs']['q64']['balanced_proxy_cost'], 5)

    def test_swaps_are_bijections_with_64_rows_per_group_and_head_block_isolation(self):
        need = unbalanced_need(heads=2, queries=256)
        # Different head needs must remain independent despite similar row IDs.
        need[1] = np.roll(need[1], 128, axis=0)
        before = need.copy()
        order, swaps = optimize_local_swaps(need)
        self.assertGreater(swaps, 0)
        self.assertEqual(order.reshape(2, 4, 64).shape, (2, 4, 64))
        for head in range(2):
            self.assertEqual(sorted(order[head].tolist()), list(range(256)))
        self.assertTrue(np.all(order // 128 == np.arange(256)[None, :] // 128))
        self.assertTrue(np.array_equal(need, before))
        self.assertFalse(np.array_equal(order[0], order[1]))

    def test_no_needed_bit_is_lost_after_regroup_and_inverse_scatter(self):
        need = unbalanced_need(heads=2, queries=256)
        order, _ = optimize_local_swaps(need)
        groups = group_support(need, order)
        supports_in_permuted_order = np.repeat(groups, 64, axis=1)
        original_supports = np.zeros_like(need)
        for head in range(need.shape[0]):
            original_supports[head, order[head]] = supports_in_permuted_order[head]
        self.assertFalse(np.any(need & ~original_supports))
        # Since Q128 membership is unchanged, groups cannot add support outside
        # their parent Q128 union, even if that union varies between blocks.
        parent = np.repeat(need.reshape(2, 2, 128, 10).any(2), 128, axis=1)
        self.assertFalse(np.any(original_supports & ~parent))

    def test_balanced_candidate_passes_gate_and_keeps_fixed_cta_count(self):
        result = screen_need(unbalanced_need())
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['costs']['local_swap']['total_tiles'], 10)
        self.assertEqual(result['costs']['local_swap']['max_cta_tiles'], 3)
        self.assertEqual(result['costs']['gated']['cta_count'], result['costs']['q64']['cta_count'])
        self.assertTrue(np.array_equal(result['orders']['gated'], result['orders']['local_swap']))

    def test_gate_rejects_insufficient_gain_and_returns_exact_baseline(self):
        result = screen_need(unbalanced_need(), Options(min_proxy_gain=.9))
        self.assertFalse(result['accepted'])
        self.assertIn('insufficient_proxy_gain', result['reasons'])
        self.assertTrue(np.array_equal(result['orders']['gated'], natural_order(unbalanced_need())))
        self.assertEqual(result['costs']['gated'], result['costs']['q64'])

    def test_gate_rejects_sum_increase_p95_regression_and_excess_movement(self):
        options = Options()
        baseline = cost_from_counts(np.array([10, 0]))
        candidate = cost_from_counts(np.array([6, 6]))
        accepted, reasons = gate_candidate(baseline, candidate, 2, 128, options)
        self.assertFalse(accepted)
        self.assertIn('total_work_increase', reasons)
        # Better maximum but worse p95: two isolated heavy CTAs become six.
        baseline = cost_from_counts(np.array([20] + [2] * 19))
        candidate = cost_from_counts(np.array([10] * 3 + [0] * 17))
        accepted, reasons = gate_candidate(baseline, candidate, 2, 128, options)
        self.assertFalse(accepted)
        self.assertIn('p95_regression', reasons)
        accepted, reasons = gate_candidate(cost_from_counts(np.array([10, 0])),
                                          cost_from_counts(np.array([5, 5])), 64, 128, options)
        self.assertFalse(accepted)
        self.assertIn('permutation_budget', reasons)

    def test_no_motion_or_all_zero_needs_are_not_accepted(self):
        options = Options(max_swaps_per_head=0)
        result = screen_need(unbalanced_need(), options)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['swaps'], 0)
        zero = screen_need(np.zeros((1, 128, 8), dtype=bool))
        self.assertFalse(zero['accepted'])
        self.assertEqual(zero['costs']['q64']['balanced_proxy_cost'], 0)
        report = aggregate_results([zero])
        json.dumps(report, allow_nan=False)
        self.assertIsNone(report['variants']['q64']['proxy_ratio_to_q64'])

    def test_motion_budget_is_enforced_inside_optimizer_and_search_is_deterministic(self):
        need = unbalanced_need()
        options = Options(max_moved_fraction=0)
        order, swaps = optimize_local_swaps(need, options)
        self.assertEqual(swaps, 0)
        self.assertTrue(np.array_equal(order, natural_order(need)))
        a, n = optimize_local_swaps(need)
        b, m = optimize_local_swaps(need)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(n, m)

    def test_uniform_tail_assumption_changes_counts_but_not_grid(self):
        result = screen_need(unbalanced_need(), Options(tail_tiles=3))
        base = result['costs']['q64']
        self.assertEqual(base['total_tiles'], 16)
        self.assertEqual(base['max_cta_tiles'], 7)
        self.assertEqual(base['cta_count'], 4)

    def test_invalid_orders_counts_options_and_query_padding_fail_closed(self):
        need = unbalanced_need(queries=256)
        duplicate = natural_order(need)
        duplicate[0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'bijection'):
            validate_order(need, duplicate)
        crossed = np.roll(natural_order(need), 128, axis=1)
        with self.assertRaisesRegex(ValueError, 'Q128 block'):
            validate_order(need, crossed)
        head_global_indices = np.arange(512).reshape(2, 256)
        with self.assertRaisesRegex(ValueError, 'bijection'):
            validate_order(unbalanced_need(heads=2, queries=256), head_global_indices)
        for q in (0, 64, 192, 255):
            with self.subTest(q=q), self.assertRaisesRegex(ValueError, 'no padded query'):
                validate_need(np.zeros((1, q, 8), dtype=bool))
        with self.assertRaises(ValueError):
            alias2_loads(np.array([-1]))
        with self.assertRaises(ValueError):
            alias2_loads(np.array([1.5]))
        with self.assertRaises(ValueError):
            Options(sm_count=0)
        with self.assertRaises(ValueError):
            Options(min_proxy_gain=float('nan'))
        with self.assertRaisesRegex(ValueError, 'CTA count'):
            gate_candidate(cost_from_counts(np.array([1])), cost_from_counts(np.array([1, 1])), 2, 128)

    def test_snapshot_loader_checks_extent_padding_and_router_support(self):
        need = unbalanced_need()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private_row.npz'
            save_dump(path, need)
            self.assertTrue(np.array_equal(load_need(path), need))
            packed = np.packbits(need, axis=-1)
            packed[0, 0, -1] |= 1     # pt=10: six forbidden trailing padding bits
            save_dump(path, need, need=packed)
            with self.assertRaisesRegex(ValueError, 'padding'):
                load_need(path)
            save_dump(path, need, pt=np.array(17))
            with self.assertRaisesRegex(ValueError, 'extent'):
                load_need(path)
            wrong = np.zeros((1, 1, 2), dtype=np.uint8)
            save_dump(path, need, kept128=wrong)
            with self.assertRaisesRegex(ValueError, 'kept128'):
                load_need(path)

    def test_cli_report_is_anonymous_aggregate_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_dump(root / 'private_sensitive_id_seed42.npz', unbalanced_need())
            save_dump(root / 'another_private_id.npz', unbalanced_need(heads=2))
            output = root / 'anonymous_report.json'
            report = main([str(root), str(output)])
            encoded = output.read_text(encoding='utf-8')
            self.assertNotIn('private_sensitive', encoded)
            self.assertNotIn('another_private', encoded)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn('orders', report)
            self.assertEqual(report['snapshots'], 2)
            self.assertEqual(report['accepted'], 2)
            self.assertEqual(report['variants']['q64']['cta_count'], 12)
            self.assertIn('not GPU time', report['measurement'])
            with self.assertRaises(FileExistsError):
                main([str(root), str(output)])
            with self.assertRaisesRegex(ValueError, 'no snapshots'):
                aggregate_results([])


if __name__ == '__main__':
    unittest.main()
