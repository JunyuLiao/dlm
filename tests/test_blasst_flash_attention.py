import math
import unittest

import torch

from blasst.calibration import (
    PhysicalTileScoreCollector,
    lambda_for_physical_sparsity,
    measured_physical_sparsity,
)
from blasst.flash_attention import blasst_flash_attn_func, collect_blasst_stats


class BlasstReferenceTest(unittest.TestCase):
    def test_zero_threshold_matches_dense_attention(self):
        torch.manual_seed(7)
        q = torch.randn(2, 17, 4, 8)
        k = torch.randn(2, 17, 2, 8)
        v = torch.randn(2, 17, 2, 8)
        actual = blasst_flash_attn_func(q, k, v, blasst_lambda=0.0, q_block_size=5, kv_block_size=5)
        kr, vr = k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, kr) / math.sqrt(8)
        expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), vr)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_blasst_skips_low_score_late_block(self):
        collect_blasst_stats(reset=True)
        q = torch.ones(1, 4, 1, 2)
        # Reverse traversal sees the high-score final tile first, then skips the low tile.
        k = torch.cat((torch.ones(1, 4, 1, 2) * -5, torch.ones(1, 4, 1, 2) * 5), dim=1)
        v = torch.randn(1, 8, 1, 2)
        blasst_flash_attn_func(q, k, v, blasst_lambda=1e-4, q_block_size=4, kv_block_size=4)
        stats = collect_blasst_stats(reset=True)
        self.assertEqual((stats.total_row_blocks, stats.skipped_row_blocks), (8, 4))
        self.assertEqual((stats.total_blocks, stats.skipped_blocks, stats.sparsity_ratio), (2, 1, 0.5))

    def test_non_unanimous_vote_keeps_the_whole_physical_tile(self):
        q = torch.tensor([[[[1.0]], [[-1.0]]]])
        k = torch.tensor([[[[-1.0]], [[1.0]]]])
        v = torch.tensor([[[[3.0]], [[7.0]]]])
        dense = blasst_flash_attn_func(q, k, v, blasst_lambda=0.0, q_block_size=2, kv_block_size=1)
        collect_blasst_stats(reset=True)
        sparse = blasst_flash_attn_func(q, k, v, blasst_lambda=0.5, q_block_size=2, kv_block_size=1)
        stats = collect_blasst_stats(reset=True)
        self.assertEqual(stats.skipped_row_blocks, 1)
        self.assertEqual(stats.skipped_blocks, 0)
        torch.testing.assert_close(sparse, dense)

    def test_collects_physical_score_and_calibrates_its_quantile(self):
        collector = PhysicalTileScoreCollector()
        q = torch.ones(2, 2, 1, 1)
        k = torch.tensor([[[[-2.0]], [[2.0]]], [[[-1.0]], [[1.0]]]])
        v = torch.ones_like(k)
        blasst_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=0.0,
            q_block_size=2,
            kv_block_size=1,
            physical_score_callback=lambda q_tile, kv_tile, scores: collector(0, q_tile, kv_tile, scores),
        )
        grouped = collector.bucketed_scores(["high", "low"])
        # Reverse traversal makes the first score unskippable (inf). The
        # second scores are exp(-4) and exp(-2), respectively.
        self.assertEqual(grouped[("high",)].numel(), 2)
        self.assertTrue(torch.isinf(grouped[("high",)]).any())
        threshold = lambda_for_physical_sparsity(grouped[("high",)], 0.5)
        self.assertGreater(threshold, math.exp(-4))
        self.assertLess(threshold, 1.0)
        self.assertEqual(measured_physical_sparsity(grouped[("high",)], threshold), 0.5)

    def test_per_sequence_lambda_controls_physical_skip(self):
        collect_blasst_stats(reset=True)
        q = torch.ones(2, 2, 1, 1)
        k = torch.tensor([[[[-2.0]], [[2.0]]], [[[-2.0]], [[2.0]]]])
        v = torch.ones_like(k)
        blasst_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=torch.tensor([0.0, 0.1]),
            q_block_size=2,
            kv_block_size=1,
        )
        stats = collect_blasst_stats(reset=True)
        self.assertEqual(stats.skipped_blocks_per_sequence, (0, 1))
        self.assertEqual(stats.total_blocks_per_sequence, (2, 2))


if __name__ == "__main__":
    unittest.main()
