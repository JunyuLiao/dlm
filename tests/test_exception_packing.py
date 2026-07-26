import unittest

import torch

from blasst.exception_packing import (
    active_counts,
    exception_tiles_per_row_histogram,
    grouped_exception_totals,
    summarize_packability,
)
from blasst.runtime import structural_blasst_reference


class ExceptionPackingTest(unittest.TestCase):
    def test_active_counts_handle_partial_query_tile(self):
        keep = torch.zeros(2, 5, 3, dtype=torch.bool)
        keep[0, :3, 1] = True
        counts = active_counts(keep, query_tile_size=4)
        self.assertEqual(counts.shape, (2, 2, 3))
        self.assertEqual(int(counts[0, 0, 1]), 3)
        self.assertEqual(int(counts[0, 1, 1]), 0)

    def test_grouping_never_mixes_kv_heads(self):
        counts = torch.zeros(4, 2, 1, dtype=torch.int64)
        counts[:, :, 0] = torch.tensor([[2, 2], [3, 0], [4, 0], [1, 1]])
        q_totals, _ = grouped_exception_totals(counts, 4, grouping="q-head", num_kv_heads=2)
        kv_totals, _ = grouped_exception_totals(counts, 4, grouping="kv-head", num_kv_heads=2)
        self.assertEqual(sorted(q_totals.tolist()), [2, 3, 4, 4])
        self.assertEqual(sorted(kv_totals.tolist()), [6, 7])

    def test_packability_and_v_reuse_are_exact(self):
        counts = torch.full((1, 16, 1), 4)
        metrics = summarize_packability(
            counts,
            tau=4,
            pack_m=64,
            grouping="q-head",
            num_kv_heads=1,
            sequence_length=64,
            query_tile_size=4,
            kv_tile_size=2,
            head_dim=2,
        )
        self.assertEqual(metrics.complete_packs, 1)
        self.assertEqual(metrics.partial_packs, 0)
        self.assertEqual(metrics.original_v_loads, 16)
        self.assertEqual(metrics.packed_v_loads, 1)
        self.assertEqual(metrics.weighted_utilization, 1.0)

    def test_row_contention_histogram_counts_candidate_votes(self):
        keep = torch.tensor([[[1, 0], [1, 1], [0, 1], [0, 0]]]).bool()
        counts = active_counts(keep, query_tile_size=2)
        hist = exception_tiles_per_row_histogram(keep, counts, tau=1, query_tile_size=2)
        # Tile 0/KV1 and tile 1/KV1 are one-vote candidates; KV0 in tile 0 has c=2.
        self.assertEqual(hist[:2].tolist(), [2, 2])

    def test_retained_minority_tile_updates_negative_voters_too(self):
        q = torch.ones(1, 128, 1, 1)
        q[:, 0] = -1
        # Reverse traversal visits +4 first. The later -4 tile is required by
        # only row 0, but physical retention evaluates its P@V for every row.
        k = torch.cat((torch.full((1, 64, 1, 1), -4.0), torch.full((1, 64, 1, 1), 4.0)), dim=1)
        v = torch.cat((torch.full((1, 64, 1, 1), 10.0), torch.zeros(1, 64, 1, 1)), dim=1)
        result = structural_blasst_reference(
            q, k, v, blasst_lambda=0.5, kv_block_size=64, skip_group_rows=128
        )
        self.assertTrue(bool(result.row_skip_by_head[0][0, 1:, 0].all()))
        self.assertFalse(bool(result.row_skip_by_head[0][0, 0, 0]))
        # Active-voter-only packing would leave these rows at exactly zero.
        self.assertGreater(float(result.output[0, 1, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
