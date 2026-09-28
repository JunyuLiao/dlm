"""v26 M2 (pooled mu) tests: independent reference semantics, kernel parity, config guards."""
import math

import pytest
import torch

from experiments.numerical_qk_reuse.geometry_diagnostic import _select


def _inputs(seed=3, h=8, q=5, k=96, hk=1):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(1, h, q, k, generator=g)
    projected = torch.randn(1, hk, k, 32, generator=g)
    return scores, projected, torch.ones(1, hk), torch.ones(1, q)


def test_constant_tile_value_makes_pool_equal_exact():
    scores, _, ref, sens = _inputs()
    projected = torch.randn(1, 1, 96 // 32, 32).repeat_interleave(32, dim=2)   # Z constant inside each K32 tile
    for thr in (-1., 0., .5, 1.5):
        exact = _select(scores, projected, ref, sens, heads=1, queries=1, keys=32, threshold=thr)
        pooled = _select(scores, projected, ref, sens, heads=1, queries=1, keys=32, threshold=thr, pool=True)
        assert torch.equal(exact['bits'], pooled['bits'])


def test_pool_ignores_attention_weights_and_can_underestimate():
    # Tile 0: zero projected V (the retained state starts at 0). Tile 1: two keys with
    # opposite projected values, attention strongly prefers the first one.
    scores = torch.tensor([[[[0., 0., 8., -8.]]]])
    projected = torch.zeros(1, 1, 4, 32)
    projected[0, 0, 2, 0], projected[0, 0, 3, 0] = 1., -1.
    ref, sens = torch.ones(1, 1), torch.ones(1, 1)
    exact = _select(scores, projected, ref, sens, heads=1, queries=1, keys=2, threshold=-50., pool=False)
    pooled = _select(scores, projected, ref, sens, heads=1, queries=1, keys=2, threshold=-50., pool=True)
    # Exact mu of tile 1 is ~(1, 0, ...); pooled mu is 0, so pooled risk is lower (here -inf).
    assert torch.isfinite(exact['risk'][0, 0, 0, 1])
    assert pooled['risk'][0, 0, 0, 1] < exact['risk'][0, 0, 0, 1]


def test_pool_excludes_illegal_and_padding_keys():
    scores, projected, ref, sens = _inputs(k=70)
    base = _select(scores, projected, ref, sens, heads=1, queries=1, keys=64, threshold=.5, pool=True)
    poisoned = projected.clone()
    scores2 = scores.clone()
    scores2[..., 60:] = -math.inf          # illegal keys: their projected values must not matter
    poisoned[:, :, 60:] = 1e6
    a = _select(scores2, projected, ref, sens, heads=1, queries=1, keys=64, threshold=.5, pool=True)
    b = _select(scores2, poisoned, ref, sens, heads=1, queries=1, keys=64, threshold=.5, pool=True)
    assert torch.equal(a['bits'], b['bits'])
    assert base['bits'].shape == a['bits'].shape


def test_v21_config_rejects_pooled_without_bootstrap_and_bad_period():
    from experiments.numerical_qk_reuse import v21
    with pytest.raises(ValueError):
        v21.effective_config({'diagnostic': False}, 'M1_R1_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL',
                             output_score_precision='fp32_scores_bf16_pv', mu_mode='pooled')


def test_reserve_physical_keeps_storage_shared_with_another_entry():
    from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
    ident = lambda layer: Identity(0, 0, 0, layer, 1, 4, 2, 8, 64, 16, 100, 0, 64, .1,
                                   'torch.float32', 'cpu', (False, None, None), 1)
    cache = ScoreCache(8, 3, 10**9)
    shared = torch.zeros(1, 4, 8, 64)
    cache.publish_scores(ident(5), 0, shared)
    cache.publish_scores(ident(11), 0, shared)
    need = shared.numel() * 4
    # Replacing layer 5 frees nothing: layer 11 still holds the storage.
    assert cache.reserve_physical(ident(5), need) == need + need


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')
@pytest.mark.parametrize('variant', ['static', 'generic'])
def test_pooled_route_only_matches_independent_reference(variant):
    pytest.importorskip('triton')
    from experiments.numerical_qk_reuse.cached_executor import route_only
    g = torch.Generator().manual_seed(11)
    scores = torch.randn(1, 8, 256, 200, generator=g)
    scores[..., 190:] = -math.inf
    projected = torch.randn(1, 2, 200, 32, generator=g)
    ref = torch.rand(1, 2, generator=g) + .5
    sens = torch.rand(1, 256, generator=g) + .5
    for thr in (-2., -.5, .5):
        want = _select(scores, projected, ref, sens, heads=1, queries=128, keys=64, threshold=thr, pool=True)
        got = route_only(scores.cuda().contiguous(), projected.cuda().contiguous(), ref.cuda().contiguous(),
                         sensitivity=sens.cuda().contiguous(), log_threshold=thr, variant=variant, pool=True)
        keep = (got.eligible & ~got.skipped).cpu()
        assert torch.equal(keep, want['bits'][:, :, ::128, :]), thr
        exact = route_only(scores.cuda().contiguous(), projected.cuda().contiguous(), ref.cuda().contiguous(),
                           sensitivity=sens.cuda().contiguous(), log_threshold=thr, variant=variant, pool=False)
        want_exact = _select(scores, projected, ref, sens, heads=1, queries=128, keys=64, threshold=thr)
        assert torch.equal((exact.eligible & ~exact.skipped).cpu(), want_exact['bits'][:, :, ::128, :])
