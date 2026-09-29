"""v27 fused observation vs the separate producer + route STORE path (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


@pytest.mark.parametrize('keys,splits,mu', [(1100, 1, True), (1100, 2, True), (2049, 2, True), (1100, 2, False)])
def test_fused_observe_matches_store_path(keys, splits, mu):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
    from experiments.numerical_qk_reuse.integration import Attention
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
    g = torch.Generator(device='cuda').manual_seed(keys)
    nq, h, hk, d = 256, 16, 2, 512
    q = torch.randn(1, h, nq, d, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    z = torch.randn(1, hk, keys, 32, device='cuda', generator=g)
    ref = torch.rand(1, hk, device='cuda', generator=g) + .5
    sens = torch.rand(1, nq, device='cuda', generator=g) + .5
    scale = d ** -.5
    pt = (keys - nq) // 64
    qb, kt = math.ceil(nq / 128), math.ceil(keys / 64)
    rank = 32 if mu else 0
    pool_kw = {} if mu else None
    scores = (torch.einsum('bhqd,bhkd->bhqk', q.float(), k.repeat_interleave(8, 1).float()) * scale).contiguous()
    old = allocate_summary(1, h, qb, kt, pt, rank, 'cuda', ('a',))
    new = allocate_summary(1, h, qb, kt, pt, rank, 'cuda', ('b',))
    if mu:
        want = route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=old,
                          store_summary=True, variant='generic')
    else:
        from experiments.numerical_qk_reuse.cached_executor import tile_pool
        valid = torch.ones(1, hk, keys, dtype=torch.bool, device='cuda')
        pooled, count = tile_pool(z, valid)
        pool_kw = dict(pool='compact', pooled=pooled, pool_count=count)
        want = route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=old,
                          store_summary=True, variant='generic', **pool_kw)
    out, tail = fused_observe(q, k, v, z, scale, pt, new, splits=splits, mu=mu)
    assert tail.shape == (1, h, nq, keys - pt * 64)
    torch.testing.assert_close(tail, scores[..., pt * 64:], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(new.z, old.z, atol=1e-2, rtol=1e-3)
    assert torch.equal(new.active, old.active) and torch.equal(new.bad, old.bad)
    if mu:
        torch.testing.assert_close(new.mu, old.mu, atol=1e-2, rtol=1e-2)
    got = route_only(tail.contiguous(), z, ref, sensitivity=sens, log_threshold=-.5, summary=new,
                     store_summary=False, variant='generic', key_offset=pt * 64, **(pool_kw or {}))
    agree = (got.skipped == want.skipped).float().mean().item()
    assert agree > .995 and torch.equal(got.eligible, want.eligible)
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.repeat_interleave(8, 1).float(), v.repeat_interleave(8, 1).float(), scale=scale).transpose(1, 2)
    torch.testing.assert_close(out.float(), dense, atol=2e-2, rtol=2e-2)
