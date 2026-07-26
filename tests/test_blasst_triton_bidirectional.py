import unittest

import torch

from blasst.triton_bidirectional import (
    DiffusionLambdaSchedule,
    blasst_bidirectional_flash_attn_func,
    get_kernel_stats,
    reset_kernel_stats,
)
from blasst.flash_attention import blasst_flash_attn_func
from blasst.pre_qk import PreQKKernelConfig, PreQKMetadataBuffers, noise_bucket


class PreQKControlTest(unittest.TestCase):
    def test_feature_and_kernel_switches_are_independent(self):
        self.assertFalse(PreQKKernelConfig().use_pre_qk_kernel)
        self.assertFalse(PreQKKernelConfig().enable_pre_skipping)
        metadata_only = PreQKKernelConfig(use_pre_qk_kernel=True, enable_pre_skipping=False)
        self.assertTrue(metadata_only.use_pre_qk_kernel)
        self.assertFalse(metadata_only.enable_pre_skipping)
        with self.assertRaisesRegex(ValueError, "requires"):
            PreQKKernelConfig(enable_pre_skipping=True)

    def test_double_buffer_does_not_alias(self):
        buffers = PreQKMetadataBuffers(2, 1, 3, 256, "cpu")
        buffers.current[0, 0, 0, 0, 0] = -4.0
        buffers.commit()
        self.assertEqual(float(buffers.previous[0, 0, 0, 0, 0]), -4.0)
        self.assertTrue(torch.isinf(buffers.current).all())

    def test_noise_buckets(self):
        self.assertEqual(noise_bucket(0.9), "high")
        self.assertEqual(noise_bucket(0.5), "mid")
        self.assertEqual(noise_bucket(0.1), "low")


class DiffusionScheduleTest(unittest.TestCase):
    def test_noise_aware_thresholds(self):
        schedule = DiffusionLambdaSchedule()
        self.assertEqual(schedule.threshold(0.9), 0.04858582466840744)
        self.assertEqual(schedule.threshold(0.5), 0.4334796965122223)
        self.assertEqual(schedule.threshold(0.15), 1.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BidirectionalBlasstKernelTest(unittest.TestCase):
    def test_active_voter_full_matches_reference_recurrence(self):
        q = torch.ones(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        q[:, 96:] = -1
        k = torch.cat(
            (-torch.ones(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16) * 4,
             torch.ones(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16) * 4),
            dim=1,
        )
        v = torch.cat((torch.ones_like(k[:, :64]) * 10, torch.ones_like(k[:, :64]) * 2), dim=1)
        reset_kernel_stats(q.device)
        actual = blasst_bidirectional_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="full", max_active_rows=32
        )
        expected = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="full",
            max_active_rows=32, q_block_size=128, kv_block_size=64,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        stats = get_kernel_stats()
        self.assertEqual(stats.active_voter_candidate_tiles, 1)
        self.assertEqual(stats.active_voter_selected_rows, 32)
        self.assertEqual(stats.active_voter_removed_rows, 96)

    def test_row_mask_dump_matches_parent_tile_counters(self):
        torch.manual_seed(13)
        q = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        keep = torch.empty(1, 2, 256, 4, device="cuda", dtype=torch.uint8)
        reset_kernel_stats(q.device)
        blasst_bidirectional_flash_attn_func(
            q, k, v, blasst_lambda=0.3, row_keep_mask_out=keep
        )
        stats = get_kernel_stats()
        physical_keep = keep.bool().reshape(1, 2, 2, 128, 4).any(3)
        self.assertEqual(int((~physical_keep).sum()), stats.skipped_tiles)
        self.assertEqual(physical_keep.numel(), stats.total_tiles)

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

    def test_rejects_disabled_structural_optimizations(self):
        q = torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        disabled = (
            {"skip_group_rows": 32, "dense_fallback_threshold": 3},
            {"query_tile_rows": 64, "skip_group_rows": 32},
            {"query_permutation": torch.arange(128, device="cuda")[None]},
            {"kv_tile_order": torch.tensor([[[1, 0]]], device="cuda", dtype=torch.int32)},
        )
        for options in disabled:
            with self.subTest(options=tuple(options)):
                with self.assertRaisesRegex(NotImplementedError, "disabled"):
                    blasst_bidirectional_flash_attn_func(q, k, v, **options)

    def test_pre_qk_gate_avoids_one_qk_tile(self):
        q = torch.ones(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.cat((-torch.ones_like(q[:, :64]) * 4, torch.ones_like(q[:, :64]) * 4), dim=1)
        v = torch.randn_like(q)
        previous = torch.full((1, 1, 1, 2), torch.inf, device="cuda", dtype=torch.float16)
        previous[..., 0] = -20.0
        current = torch.full_like(previous, torch.inf)
        proxy_threshold = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        reset_kernel_stats(q.device)
        actual = blasst_bidirectional_flash_attn_func(
            q, k, v, blasst_lambda=0.0,
            enable_pre_qk=True,
            previous_proxy_log_scores=previous,
            current_proxy_log_scores=current,
            proxy_log_thresholds=proxy_threshold,
            proxy_warmup_tiles=1,
            proxy_periodic_refresh=0,
            proxy_anchor_local=False,
            proxy_anchor_sink=False,
        )
        # Only the right-hand tile remains, so every query receives its V-only
        # attention result.  This also proves the warmup tile prevents NaNs.
        expected = v[:, 64:].float().mean(dim=1, keepdim=True).expand_as(actual).to(actual.dtype)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        stats = get_kernel_stats()
        self.assertEqual(stats.pre_qk_skipped_tiles, 1)
        self.assertEqual(stats.proxy_metadata_tiles, 2)
        self.assertTrue(torch.isinf(current[..., 0]).all())


if __name__ == "__main__":
    unittest.main()
