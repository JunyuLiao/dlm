import dataclasses

import pytest
import torch

from experiments.numerical_qk_reuse.v31_value_decision import (
    OmittedCache, bounded_proposal, c_gate, fixed_topk, mass_neighborhood,
    mass_score, moments, online_risks, squared_rows,
)


def fixture(seed=3, dtype=torch.float64, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    tensors = [torch.randn(shape, generator=g, dtype=dtype).to(device)
               for shape in [(5, 8), (39, 8), (39, 8), (5, 8)]]
    return tensors


def direct(q, k, v, selected, block=4):
    prefix = len(selected) * block
    keys = torch.cat((selected.repeat_interleave(block), torch.ones(len(k) - prefix, dtype=torch.bool, device=k.device)))
    return (q @ k[keys].T / 8**.5).softmax(-1) @ v[keys]


@pytest.mark.parametrize('prefix', [0, 2, 16, 33, 39])
def test_moments_match_direct_including_mandatory_tail(prefix):
    q, k, v, direction = fixture()
    m = moments(q, k, v, prefix, 8**-.5, direction, block=4, chunk=3)
    selected = torch.arange(prefix // 4) % 2 == 0
    torch.testing.assert_close(m.output(selected), direct(q, k, v, selected), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(m.dense(), (q @ k.T / 8**.5).softmax(-1) @ v, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('prefix', [0, 16, 33])
def test_directional_derivative_matches_central_difference(prefix):
    q, k, v, direction = fixture()
    m = moments(q, k, v, prefix, 8**-.5, direction, block=4)
    selected = torch.arange(prefix // 4) % 2 == 0
    eps = 1e-5
    finite = (direct(q + eps * direction, k, v, selected) - direct(q - eps * direction, k, v, selected)) / (2 * eps)
    torch.testing.assert_close(m.tangent(selected), finite, atol=1e-9, rtol=1e-7)


def test_mass_neighbors_use_rank_not_physical_prefix():
    q, k, v, _ = fixture()
    m = moments(q, k, v, 36, 8**-.5, block=4)
    m.mass = torch.tensor([1., 3., 8., 2., 10., 9., 5., 4., 7.])[:, None].expand(-1, 5)
    anchor = fixed_topk(mass_score(m), 2)
    pool = mass_neighborhood(m, anchor, extra=2)
    assert pool.nonzero().flatten().tolist() == [2, 4, 5, 8]


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('lifetime,online', [(False, False), (True, False), (False, True)])
def test_proposals_preserve_budget_and_combined_guards(seed, lifetime, online):
    q, k, v, direction = fixture(seed)
    m = moments(q, k, v, 36, 8**-.5, direction * .1, block=4)
    base = fixed_topk(mass_score(m), 3)
    result, info = bounded_proposal(m, base, swaps=2, extra=3, lifetime=lifetime, online=online)
    assert int(result.sum()) == 3
    assert int((result & ~base).sum()) == info['swaps'] <= 2
    assert not bool((result & ~mass_neighborhood(m, base, 3)).any())
    before, after = m.output(base) - m.dense(), m.output(result) - m.dense()
    assert after.square().sum(-1).mean() <= before.square().sum(-1).mean() + 1e-6
    assert after.square().sum(-1).max() <= before.square().sum(-1).max() * 1.02**2 + 1e-6
    assert (m.selected(base)[1] - m.selected(result)[1]).max() <= .025 + 1e-6


@pytest.mark.parametrize('kind', ['residual', 'moments', 'tangent'])
def test_reconstruction_is_exact_at_anchor(kind):
    q, k, v, dq = fixture()
    m = moments(q, k, v, 32, 8**-.5, dq, block=4)
    mask = fixed_topk(mass_score(m), 3)
    cache = OmittedCache.build(m, mask)
    torch.testing.assert_close(cache.reconstruct(m, mask, kind), m.dense(), atol=1e-12, rtol=1e-12)


def test_cached_moments_normalization_matches_unnormalized_oracle():
    q, k, v, dq = fixture()
    a = moments(q, k, v, 32, 8**-.5, dq, block=4)
    b = moments(q + .5 * dq, k, v, 32, 8**-.5, block=4)
    mask = fixed_topk(mass_score(a), 3)
    cache = OmittedCache.build(a, mask)
    selected = torch.cat((mask.repeat_interleave(4), torch.ones(7, dtype=torch.bool)))
    logits = q @ k.T / 8**.5
    logits[:, selected] = ((q + .5 * dq) @ k.T / 8**.5)[:, selected]
    torch.testing.assert_close(cache.reconstruct(b, mask), logits.softmax(-1) @ v, atol=1e-12, rtol=1e-12)
    # Arbitrary common offsets must not change normalized reconstruction.
    shift = 1000.
    cache.log_scale += shift
    b.log_scale = b.log_scale + shift
    torch.testing.assert_close(cache.reconstruct(b, mask), logits.softmax(-1) @ v, atol=1e-12, rtol=1e-12)


def test_projection_gram_is_exact_single_head_metric():
    e = torch.randn(6, 8, dtype=torch.float64)
    w = torch.randn(19, 8, dtype=torch.float64)
    torch.testing.assert_close(squared_rows(e, w.T @ w), (e @ w.T).square().sum(-1))


def test_cache_rejects_support_changes_and_noop_keeps_budget():
    q, k, v, _ = fixture()
    m = moments(q, k, v, 32, 8**-.5, block=4)
    mask = fixed_topk(mass_score(m), 3)
    cache = OmittedCache.build(m, mask)
    with pytest.raises(ValueError, match='cache reset'):
        cache.reconstruct(m, ~mask)
    allkept = torch.ones_like(mask)
    result, info = bounded_proposal(m, allkept)
    assert torch.equal(result, allkept) and info['swaps'] == 0


def test_cgate_uses_row_identity_without_budget_change():
    accepted = torch.tensor([True, False, False, True])
    changed = torch.tensor([False, False, True, True])
    reversed_ = torch.tensor([False, True, True, False])
    assert c_gate(accepted, changed, reversed_).tolist() == [1., 3., 4., 2.]


def test_online_threshold_changes_running_reference():
    q, k, v, _ = fixture()
    m = moments(q, k, v, 32, 8**-.5, block=4)
    lo, hi = online_risks(m, 0), online_risks(m, 100)
    assert torch.equal(lo[:2], hi[:2])
    assert not torch.allclose(lo[2:], hi[2:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA gate requires H100')
def test_cuda_moments_and_fixed_budget_match_float64_reference():
    q, k, v, dq = fixture()
    cpu = moments(q, k, v, 32, 8**-.5, dq, block=4)
    gpu = moments(q.cuda().float(), k.cuda().float(), v.cuda().float(), 32, 8**-.5, dq.cuda().float(), block=4)
    base = fixed_topk(mass_score(cpu), 3)
    torch.testing.assert_close(gpu.output(base.cuda()).cpu().double(), cpu.output(base), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gpu.tangent(base.cuda()).cpu().double(), cpu.tangent(base), atol=2e-6, rtol=2e-5)
    for lifetime in (False, True):
        a, _ = bounded_proposal(cpu, base, lifetime=lifetime)
        b, _ = bounded_proposal(gpu, base.cuda(), lifetime=lifetime)
        assert torch.equal(a, b.cpu())
