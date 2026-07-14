import math

import torch

from blasst.flash_attention import blasst_flash_attn_func, collect_blasst_stats


def test_zero_threshold_matches_dense_attention():
    torch.manual_seed(7)
    q = torch.randn(2, 17, 4, 8)
    k = torch.randn(2, 17, 2, 8)
    v = torch.randn(2, 17, 2, 8)
    actual = blasst_flash_attn_func(q, k, v, blasst_lambda=0.0, q_block_size=5, kv_block_size=5)
    kr, vr = k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, kr) / math.sqrt(8)
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), vr)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_blasst_skips_low_score_late_block():
    collect_blasst_stats(reset=True)
    q = torch.ones(1, 4, 1, 2)
    # Reverse traversal sees the high-score final tile first, then skips the low tile.
    k = torch.cat((torch.ones(1, 4, 1, 2) * -5, torch.ones(1, 4, 1, 2) * 5), dim=1)
    v = torch.randn(1, 8, 1, 2)
    blasst_flash_attn_func(q, k, v, blasst_lambda=1e-4, q_block_size=4, kv_block_size=4)
    stats = collect_blasst_stats(reset=True)
    assert (stats.total_row_blocks, stats.skipped_row_blocks) == (8, 4)
    assert (stats.total_blocks, stats.skipped_blocks, stats.sparsity_ratio) == (2, 1, 0.5)
