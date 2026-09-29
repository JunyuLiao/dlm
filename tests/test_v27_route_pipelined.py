"""v27 pipelined summary-LOAD selector: bit-identical decisions to the generic selector (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')

H, HK, D, NQ = 16, 2, 512, 256


def _setup(keys, seed, mode):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only, tile_pool
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(1, H, NQ, D, device='cuda', generator=g) * .6
    k = torch.randn(1, HK, keys, D, device='cuda', generator=g) * .6
    scores = (torch.einsum('bhqd,bhkd->bhqk', q, k.repeat_interleave(H // HK, 1)) * D ** -.5).contiguous()
    z = torch.randn(1, HK, keys, 32, device='cuda', generator=g)
    ref = torch.rand(1, HK, device='cuda', generator=g) + .5
    sens = torch.rand(1, NQ, device='cuda', generator=g) + .5
    pt = (keys - NQ) // 64
    qb, kt = math.ceil(NQ / 128), math.ceil(keys / 64)
    compact = mode == 'compact'
    summary = allocate_summary(1, H, qb, kt, pt, 0 if compact else 32, 'cuda', (mode, keys, seed))
    kw = {}
    if compact:
        pooled, count = tile_pool(z, torch.ones(1, HK, keys, dtype=torch.bool, device='cuda'))
        kw = dict(pool='compact', pooled=pooled, pool_count=count)
    elif mode == 'pooled':
        kw = dict(pool=True)
    # anchor call publishes the prefix summary
    route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=summary, store_summary=True,
               variant='generic', **kw)
    # a later decision call: live canvas V, reference and sensitivity changed
    z2 = z.clone()
    z2[:, :, pt * 64:] += torch.randn(1, HK, keys - pt * 64, 32, device='cuda', generator=g)
    ref2 = ref * 1.1
    sens2 = torch.rand(1, NQ, device='cuda', generator=g) + .5
    if compact:
        pooled, count = tile_pool(z2, torch.ones(1, HK, keys, dtype=torch.bool, device='cuda'))
        kw = dict(pool='compact', pooled=pooled, pool_count=count)
    return scores, z2, ref2, sens2, summary, pt, kw


def _same(a, b):
    assert torch.equal(a.skipped, b.skipped)
    assert torch.equal(a.eligible, b.eligible)
    assert torch.equal(a.invalid_tiles, b.invalid_tiles)


@pytest.mark.parametrize('mode', ['exact', 'compact', 'pooled'])
@pytest.mark.parametrize('keys', [1100, 2049, 4133])
@pytest.mark.parametrize('threshold', [-.5, -1.5])
def test_pipelined_load_is_bit_identical(mode, keys, threshold):
    from experiments.numerical_qk_reuse.cached_executor import route_only
    scores, z, ref, sens, summary, pt, kw = _setup(keys, keys + int(-threshold * 10), mode)
    want = route_only(scores, z, ref, sensitivity=sens, log_threshold=threshold, summary=summary,
                      variant='generic', **kw)
    skipped = want.skipped[want.eligible].float().mean().item()
    assert 0 < skipped < 1, 'inputs must exercise both decisions'
    for stages in (1, 3):
        got = route_only(scores, z, ref, sensitivity=sens, log_threshold=threshold, summary=summary,
                         variant='generic', pipelined=True, num_stages=stages, **kw)
        _same(got, want)
        if mode == 'compact':
            assert torch.equal(got.pool_mismatch, want.pool_mismatch)


@pytest.mark.parametrize('mode', ['exact', 'compact'])
def test_pipelined_tail_scores_match(mode):
    from experiments.numerical_qk_reuse.cached_executor import route_only
    scores, z, ref, sens, summary, pt, kw = _setup(2049, 7, mode)
    tail = scores[..., pt * 64:].contiguous()
    want = route_only(tail, z, ref, sensitivity=sens, log_threshold=-.5, summary=summary, variant='generic',
                      key_offset=pt * 64, **kw)
    got = route_only(tail, z, ref, sensitivity=sens, log_threshold=-.5, summary=summary, variant='generic',
                     key_offset=pt * 64, pipelined=True, num_stages=3, **kw)
    _same(got, want)


def test_pipelined_flags_invalid_tail_scores_like_generic():
    from experiments.numerical_qk_reuse.cached_executor import route_only
    scores, z, ref, sens, summary, pt, kw = _setup(1100, 3, 'exact')
    scores[0, 3, 17, pt * 64 + 5] = float('nan')
    scores[0, 9, 200, pt * 64 + 70] = float('inf')
    want = route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=summary, variant='generic')
    got = route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=summary, variant='generic',
                     pipelined=True, num_stages=3)
    _same(got, want)
    assert got.invalid_tiles.sum() == 2


def test_pipelined_refuses_store_and_static():
    from experiments.numerical_qk_reuse.cached_executor import route_only
    scores, z, ref, sens, summary, pt, kw = _setup(1100, 4, 'exact')
    with pytest.raises(ValueError):
        route_only(scores, z, ref, sensitivity=sens, summary=summary, store_summary=True, variant='generic',
                   pipelined=True)
    with pytest.raises(ValueError):
        route_only(scores, z, ref, sensitivity=sens, summary=None, variant='generic', pipelined=True)
