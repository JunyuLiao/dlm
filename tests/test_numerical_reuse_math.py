import math

import pytest
import torch

from experiments.numerical_qk_reuse.reference import attention


# Frozen before GPU qualification. Routing cases stay at least .19 log-risk
# away from the threshold; BF16 tensor-core PV is compared at output dtype.
BF16_OUTPUT_ATOL = .05
BF16_OUTPUT_RTOL = .03


def test_all_kept_uses_cached_scores_and_current_v_with_gqa_partial_tiles():
    generator = torch.Generator().manual_seed(71)
    scores = torch.randn((1, 4, 129, 65), generator=generator)
    scores[:, :, :, 0] = -math.inf
    v = torch.randn((1, 2, 65, 3), generator=generator).to(torch.bfloat16)
    z = torch.randn((1, 2, 65, 2), generator=generator)
    result = attention(scores, v, z, torch.ones(1, 2), log_threshold=-math.inf)
    for h in range(4):
        expected = torch.softmax(scores[0, h], -1) @ v[0, h // 2].float()
        torch.testing.assert_close(result.output[0, h].float(), expected, atol=.03, rtol=.03)
    assert result.skipped.shape == (1, 4, 2, 2)
    assert not result.skipped.any()
    assert not result.invalid_scores.any()
    shifted = attention(scores, v * 2, z, torch.ones(1, 2), log_threshold=-math.inf)
    assert not torch.equal(result.output, shifted.output)
    different_current_scores = scores + torch.linspace(0, 2, 65)
    assert not torch.allclose(result.output.float(),
                              torch.softmax(different_current_scores[0], -1) @ v[0, 0].float())


def test_first_support_ties_and_empty_rows():
    scores = torch.full((1, 1, 3, 192), -math.inf)
    scores[0, 0, 0, 0] = 0
    scores[0, 0, 0, 64] = 0
    scores[0, 0, 1, 64] = 0
    scores[0, 0, 1, 128] = 0
    v = torch.ones((1, 1, 192, 2), dtype=torch.bfloat16)
    z = torch.zeros((1, 1, 192, 2))
    result = attention(scores, v, z, torch.ones(1, 1), log_threshold=0.)
    assert result.eligible[0, 0, 0].tolist() == [True, True, True]
    assert result.skipped[0, 0, 0].tolist() == [False, False, True]
    assert result.output[0, 0, 2].tolist() == [0., 0.]
    assert result.log_normalizer[0, 0, 2] == -math.inf
    # A zero risk is -inf, so strict comparison makes -inf < -inf false.
    no_drop = attention(scores, v, z, torch.ones(1, 1), log_threshold=-math.inf)
    assert not no_drop.skipped.any()


def test_strict_threshold_row_weight_before_physical_max_and_held_bitmap():
    scores = torch.zeros((1, 1, 2, 128))
    v = torch.ones((1, 1, 128, 2), dtype=torch.bfloat16)
    z = torch.zeros((1, 1, 128, 1))
    z[0, 0, 64:, 0] = 2.
    base = attention(scores, v, z, torch.ones(1, 1), log_threshold=0.)
    # First tile has to be retained; second has alpha=.5 and risk=1.
    assert base.risk[0, 0, 0, 1] == pytest.approx(0.)
    assert not base.skipped[0, 0, 0, 1]
    loose = attention(scores, v, z, torch.ones(1, 1), log_threshold=.01)
    assert loose.skipped[0, 0, 0, 1]
    weighted = attention(scores, v, z, torch.ones(1, 1),
                         sensitivity=torch.tensor([[1., 4.]]), log_threshold=.01)
    assert not weighted.skipped[0, 0, 0, 1]
    held = attention(scores, v * 3, skipped=loose.skipped)
    assert torch.equal(held.skipped, loose.skipped)
    assert held.projected_state.shape[-1] == 0
    assert held.output[0, 0, 0].tolist() == [3., 3.]


def test_nonfinite_scores_are_flagged_and_force_keep():
    scores = torch.full((1, 1, 2, 128), -math.inf)
    scores[0, 0, 0, 0] = 0.
    scores[0, 0, 0, 64] = float('nan')
    scores[0, 0, 1, 64] = float('inf')
    v = torch.ones((1, 1, 128, 2), dtype=torch.bfloat16)
    z = torch.zeros((1, 1, 128, 1))
    result = attention(scores, v, z, torch.ones(1, 1), log_threshold=math.inf)
    assert not result.skipped[0, 0, 0, 1]
    assert result.invalid_scores[0, 0].tolist() == [True, True]
    assert torch.count_nonzero(result.output) == 0
    assert torch.isneginf(result.log_normalizer).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_cached_score_executor_matches_frozen_reference():
    from experiments.numerical_qk_reuse.cached_executor import attention as cuda_attention

    generator = torch.Generator().manual_seed(2901)
    scores = torch.randn((1, 4, 129, 65), generator=generator, dtype=torch.float32).cuda()
    scores[:, :, :, 0] = -math.inf
    v = torch.randn((1, 2, 65, 256), generator=generator).to(torch.bfloat16).cuda()
    z = torch.randn((1, 2, 65, 32), generator=generator).cuda()
    ref = torch.ones((1, 2), device='cuda')
    expected = attention(scores, v, z, ref, log_threshold=-math.inf)
    actual = cuda_attention(scores, v, z, ref, log_threshold=-math.inf, trace=True)
    assert torch.equal(actual.skipped, expected.skipped)
    assert torch.equal(actual.eligible, expected.eligible)
    assert not actual.invalid_scores.any()
    torch.testing.assert_close(actual.output.float(), expected.output.float(),
                               atol=BF16_OUTPUT_ATOL, rtol=BF16_OUTPUT_RTOL)
    torch.testing.assert_close(actual.log_normalizer, expected.log_normalizer, atol=2e-3, rtol=2e-3)
    assert torch.equal(actual.counters[..., 0], actual.counters[..., 1])
    assert torch.equal(actual.counters[..., 1], actual.counters[..., 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_route_and_held_bitmap_skip_pv():
    from experiments.numerical_qk_reuse.cached_executor import attention as cuda_attention

    scores = torch.zeros((1, 1, 128, 128), device='cuda', dtype=torch.float32)
    v = torch.ones((1, 1, 128, 256), device='cuda', dtype=torch.bfloat16)
    z = torch.zeros((1, 1, 128, 32), device='cuda')
    z[:, :, 64:, 0] = 4.
    ref = torch.ones((1, 1), device='cuda')
    # Second-tile log-risk is log(2)=.693, separated from both thresholds.
    keep = cuda_attention(scores, v, z, ref, log_threshold=.5)
    drop = cuda_attention(scores, v, z, ref, log_threshold=.9, trace=True)
    assert not keep.skipped.any()
    assert drop.skipped[0, 0, 0].tolist() == [False, True]
    expected = attention(scores, v, z, ref, log_threshold=.9)
    assert torch.equal(drop.skipped, expected.skipped)
    held = cuda_attention(scores, v*3, skipped=drop.skipped, eligible=drop.eligible, trace=True)
    assert torch.equal(held.skipped, drop.skipped)
    assert torch.equal(held.output, torch.full_like(held.output, 3.))
    assert torch.all(held.counters[..., 1] == 1)
