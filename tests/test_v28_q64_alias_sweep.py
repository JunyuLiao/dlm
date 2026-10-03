"""Dependency-light CPU checks for sweep isolation, support and aggregation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts.v28_q64_alias_sweep import (SPLITS, assert_held_builds, make_adapters,
    main, rotated_splits, selected_paths, support, timing_ratios, validate_split_batch)
from scripts.v28_regroup_screen import load_need


class FakeAdapter:
    def __init__(self, layer_types, **kwargs):
        self.layer_types, self.kwargs = layer_types, kwargs
        self._split_cache = []
        self.calls = {'split_list_builds': 0}


def save_dump(path, pt):
    need = np.zeros((2, 256, pt), dtype=bool)
    need[0, 0, 0] = True
    need[1, 128, -1] = True
    np.savez_compressed(path, need=np.packbits(need, -1), pt=np.array(pt),
                        kept128=np.packbits(need.reshape(2, 2, 128, pt).any(2), -1))


class AliasSweepTests(unittest.TestCase):
    def test_adapter_instances_and_split_caches_are_independent(self):
        paged = object()
        adapters = make_adapters(FakeAdapter, paged)
        self.assertEqual(tuple(adapters), SPLITS)
        self.assertEqual([a.splits for a in adapters.values()], [1, 2, 4])
        self.assertTrue(all(a.paged is paged for a in adapters.values()))
        self.assertTrue(all(a.kwargs['arm'] == 'allkept' for a in adapters.values()))
        adapters[1]._split_cache.append(('held', 'split1'))
        self.assertEqual(adapters[2]._split_cache, [])
        self.assertEqual(adapters[4]._split_cache, [])
        one = FakeAdapter([], arm='allkept')
        with self.assertRaisesRegex(ValueError, 'independent'):
            make_adapters(lambda *args, **kwargs: one, paged)
        one._split_cache.append(('stale', 'split2'))
        with self.assertRaisesRegex(ValueError, 'fresh'):
            make_adapters(lambda *args, **kwargs: one, paged)

    def test_split_batch_mismatch_is_rejected_instead_of_reusing_s2_cache(self):
        split = SimpleNamespace(full_block_cnt=np.zeros((2, 16, 4)), full_block_idx=np.zeros((2, 16, 4, 8)))
        validate_split_batch(split, 2)
        for count in (1, 4):
            with self.assertRaisesRegex(RuntimeError, 'alias count'):
                validate_split_batch(split, count)
        split.full_block_idx = np.zeros((4, 16, 4, 8))
        with self.assertRaises(RuntimeError):
            validate_split_batch(split, 2)

    def test_held_build_counters_fail_on_any_timed_rebuild(self):
        adapters = make_adapters(FakeAdapter, None)
        before = {s: 0 for s in SPLITS}
        assert_held_builds(adapters, before)
        adapters[4].calls['split_list_builds'] = 1
        with self.assertRaisesRegex(RuntimeError, 'rebuilt'):
            assert_held_builds(adapters, before)

    def test_rotation_balances_all_three_positions(self):
        self.assertEqual([rotated_splits(i) for i in range(3)], [(1, 2, 4), (2, 4, 1), (4, 1, 2)])
        for position in range(3):
            self.assertEqual(sorted(rotated_splits(i)[position] for i in range(3)), list(SPLITS))

    def test_selection_is_one_each_nominal_bin_despite_canvas_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, pt in enumerate((544, 480, 1117, 935, 1459, 1325)):
                save_dump(root / f'private_{i}.npz', pt)
            selected = selected_paths(root, per_bin=1)
            self.assertEqual(len(selected), 3)
            self.assertEqual([load_need(path).shape[-1] for path in selected], [544, 1117, 1459])
            for path in selected:
                need = load_need(path)
                kept = support(need, 64)
                self.assertEqual(kept.shape, (2, 4, need.shape[-1]+4))
                self.assertTrue(kept[..., -4:].all())
                self.assertFalse(np.any(need & ~np.repeat(kept[..., :-4], 64, 1)))

    def test_ratios_use_s2_control_and_reject_invalid_timing(self):
        self.assertEqual(timing_ratios({1: 3., 2: 2., 4: 2.5}), {'1': 1.5, '2': 1., '4': 1.25})
        for values in ({1: 1., 2: 0., 4: 1.}, {1: float('nan'), 2: 1., 4: 1.},
                       {1: 1., 2: 1., 4: float('inf')}, {1: 1., 2: 1.}):
            with self.assertRaises(ValueError):
                timing_ratios(values)

    def test_preflight_requires_all_bins_and_refuses_report_overwrite_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_dump(root / 'private.npz', 480)
            report = root / 'anonymous.json'
            with self.assertRaisesRegex(ValueError, 'all three'):
                main([str(root), str(report)])
            self.assertFalse(report.exists())
            report.write_text('preserve', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'new report'):
                main([str(root), str(report)])
            self.assertEqual(report.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
