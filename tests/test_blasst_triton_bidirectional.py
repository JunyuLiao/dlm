import unittest

import torch

from blasst.triton_bidirectional import (
    DiffusionLambdaSchedule,
    blasst_bidirectional_flash_attn_func,
    get_kernel_stats,
    reset_kernel_stats,
)


class DiffusionScheduleTest(unittest.TestCase):
    def test_noise_aware_thresholds(self):
        schedule = DiffusionLambdaSchedule()
        self.assertEqual(schedule.threshold(0.9), 0.04858582466840744)
        self.assertEqual(schedule.threshold(0.5), 0.4334796965122223)
        self.assertEqual(schedule.threshold(0.15), 1.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BidirectionalBlasstKernelTest(unittest.TestCase):
    def test_lambda_zero_matches_dense_sdpa(self):
        torch.manual_seed(7)
        q = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        reset_kernel_stats(q.device)
        actual = blasst_bidirectional_flash_attn_func(q, k, v, blasst_lambda=0.0)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        stats = get_kernel_stats()
        self.assertEqual(stats.skipped_tiles, 0)
        self.assertEqual(stats.total_tiles, 16)

    def test_2d_tile_vote_skips_only_unanimous_tile(self):
        q = torch.ones(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        # Reverse traversal sees the high-score tile first, then the low tile.
        k = torch.cat((-torch.ones_like(q[:, :64]) * 4, torch.ones_like(q[:, :64]) * 4), dim=1)
        v = torch.randn_like(q)
        reset_kernel_stats(q.device)
        blasst_bidirectional_flash_attn_func(q, k, v, blasst_lambda=1e-3)
        stats = get_kernel_stats()
        self.assertEqual(stats.total_tiles, 2)
        self.assertEqual(stats.skipped_tiles, 1)

    def test_per_sequence_thresholds(self):
        q = torch.ones(2, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.cat((-torch.ones_like(q[:, :64]) * 4, torch.ones_like(q[:, :64]) * 4), dim=1)
        v = torch.randn_like(q)
        reset_kernel_stats(q.device)
        blasst_bidirectional_flash_attn_func(
            q, k, v, blasst_lambda=torch.tensor([0.0, 1e-3], device="cuda")
        )
        stats = get_kernel_stats()
        self.assertEqual(stats.total_tiles, 4)
        self.assertEqual(stats.skipped_tiles, 1)


if __name__ == "__main__":
    unittest.main()
