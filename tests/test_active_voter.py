import math
import unittest

import torch

from blasst.flash_attention import (
    blasst_flash_attn_func,
    collect_active_voter_stats,
)


def explicit_row_masked_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tile_lambda: float,
    row_lambda: float,
    q_block_size: int,
    kv_block_size: int,
    max_active_rows: int,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent explicit mask construction followed by one dense softmax."""
    batch, query_len, heads, dim = q.shape
    scale = dim**-0.5
    allowed = torch.zeros(batch, heads, query_len, k.shape[1], dtype=torch.bool)
    tile_log = math.log(tile_lambda) if tile_lambda else -math.inf
    row_log = math.log(row_lambda) if row_lambda else -math.inf
    for q_start in range(0, query_len, q_block_size):
        q_end = min(query_len, q_start + q_block_size)
        q_tile = q[:, q_start:q_end].transpose(1, 2).float()
        running_max = torch.full((batch, heads, q_end - q_start), -torch.inf)
        last_k = ((k.shape[1] - 1) // kv_block_size) * kv_block_size
        for k_start in range(last_k, -1, -kv_block_size):
            k_end = min(k.shape[1], k_start + kv_block_size)
            k_tile = k[:, k_start:k_end].transpose(1, 2).float()
            scores = q_tile @ k_tile.transpose(-1, -2) * scale
            if causal:
                q_pos = torch.arange(q_start, q_end)[:, None]
                k_pos = torch.arange(k_start, k_end)[None, :]
                scores.masked_fill_(k_pos > q_pos, -torch.inf)
            local_max = scores.amax(-1)
            valid = torch.isfinite(local_max)
            gap = local_max - running_max
            baseline_vote = valid & (gap >= tile_log)
            physical_keep = baseline_vote.any(-1)
            row_vote = valid & (gap >= row_log)
            count = row_vote.sum(-1)
            candidate = physical_keep & (count > 0) & (count <= max_active_rows)
            active = torch.where(candidate[..., None], row_vote, valid)
            active &= physical_keep[..., None]
            allowed[:, :, q_start:q_end, k_start:k_end] = active[..., None]
            running_max = torch.where(active, torch.maximum(running_max, local_max), running_max)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    if causal:
        q_pos = torch.arange(query_len)[:, None]
        k_pos = torch.arange(k.shape[1])[None, :]
        allowed &= k_pos <= q_pos
    scores.masked_fill_(~allowed, -torch.inf)
    probabilities = torch.softmax(scores, -1)
    probabilities = torch.where(torch.isfinite(probabilities), probabilities, 0.0)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float()).to(q.dtype)
    return output, allowed


class ActiveVoterReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        collect_active_voter_stats(reset=True)

    @staticmethod
    def mixed_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.tensor([1.0, 1.0, 1.0, -1.0]).reshape(1, 4, 1, 1)
        k = torch.tensor([-4.0, -4.0, 4.0, 4.0]).reshape(1, 4, 1, 1)
        v = torch.tensor([10.0, 10.0, 0.0, 0.0]).reshape(1, 4, 1, 1)
        return q, k, v

    def test_all_rows_keep_matches_existing_blasst(self) -> None:
        torch.manual_seed(4)
        q = torch.randn(1, 7, 2, 3)
        k, v = torch.randn_like(q), torch.randn_like(q)
        baseline = blasst_flash_attn_func(q, k, v, blasst_lambda=0.0, q_block_size=4, kv_block_size=2)
        active = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.0, row_lambda=0.0,
            row_mask_variant="full", max_active_rows=4,
            q_block_size=4, kv_block_size=2,
        )
        torch.testing.assert_close(active, baseline)

    def test_no_rows_keep_matches_ordinary_physical_skip(self) -> None:
        q = torch.ones(1, 4, 1, 1)
        k = torch.tensor([-4.0, -4.0, 4.0, 4.0]).reshape(1, 4, 1, 1)
        v = torch.randn_like(k)
        baseline = blasst_flash_attn_func(q, k, v, blasst_lambda=0.5, q_block_size=4, kv_block_size=2)
        active = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="full",
            max_active_rows=2, q_block_size=4, kv_block_size=2,
        )
        torch.testing.assert_close(active, baseline)

    def test_mixed_voters_update_only_selected_rows_and_running_max(self) -> None:
        q, k, v = self.mixed_case()
        active = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="full",
            max_active_rows=2, q_block_size=4, kv_block_size=2,
        )
        expected, allowed = explicit_row_masked_dense(
            q, k, v, tile_lambda=0.5, row_lambda=0.5,
            q_block_size=4, kv_block_size=2, max_active_rows=2,
        )
        torch.testing.assert_close(active, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(bool(allowed[0, 0, :3, :2].any()))
        self.assertTrue(bool(allowed[0, 0, 3, :2].all()))
        stats = collect_active_voter_stats()
        self.assertEqual(stats.candidate_tiles, 1)
        self.assertEqual(stats.selected_rows, 1)
        self.assertEqual(stats.removed_rows, 3)

    def test_output_only_is_diagnostic_and_denominator_is_inconsistent(self) -> None:
        q, k, v = self.mixed_case()
        v[:, 2:] = 2.0
        full = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="full",
            max_active_rows=2, q_block_size=4, kv_block_size=2,
        )
        output_only = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, row_mask_variant="output",
            max_active_rows=2, q_block_size=4, kv_block_size=2,
        )
        self.assertGreater(float((full - output_only).abs().max()), 1e-4)
        self.assertGreater(float(full[0, 0, 0, 0]), float(output_only[0, 0, 0, 0]))

    def test_multiple_tiles_partial_boundaries_and_dense_mask_agreement(self) -> None:
        torch.manual_seed(9)
        q = torch.randn(1, 5, 2, 3)
        k, v = torch.randn_like(q), torch.randn_like(q)
        actual = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.7, row_lambda=0.35,
            row_mask_variant="full", max_active_rows=3,
            q_block_size=4, kv_block_size=2,
        )
        expected, _ = explicit_row_masked_dense(
            q, k, v, tile_lambda=0.7, row_lambda=0.35,
            q_block_size=4, kv_block_size=2, max_active_rows=3,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_causal_invalid_rows_and_sequence_boundaries(self) -> None:
        torch.manual_seed(12)
        q = torch.randn(1, 7, 1, 4)
        k, v = torch.randn_like(q), torch.randn_like(q)
        actual = blasst_flash_attn_func(
            q, k, v, causal=True, blasst_lambda=0.4, row_lambda=0.2,
            row_mask_variant="full", max_active_rows=3,
            q_block_size=4, kv_block_size=3,
        )
        expected, allowed = explicit_row_masked_dense(
            q, k, v, causal=True, tile_lambda=0.4, row_lambda=0.2,
            q_block_size=4, kv_block_size=3, max_active_rows=3,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertFalse(bool(torch.triu(allowed[0, 0], diagonal=1).any()))

    def test_bidirectional_mode_can_attend_to_future_tokens(self) -> None:
        q = torch.ones(1, 3, 1, 1)
        k = torch.tensor([0.0, 0.0, 10.0]).reshape(1, 3, 1, 1)
        v = torch.tensor([0.0, 0.0, 7.0]).reshape(1, 3, 1, 1)
        output = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.0, row_mask_variant="full",
            max_active_rows=3, q_block_size=2, kv_block_size=2,
        )
        self.assertGreater(float(output[0, 0, 0, 0]), 6.9)


if __name__ == "__main__":
    unittest.main()
