"""v27 fused observation vs the separate producer + route STORE path (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


@pytest.mark.parametrize('keys,splits,mu', [(1100, 1, True), (1100, 2, True), (2049, 2, True), (1100, 2, False)])
def test_fused_observe_matches_store_path(keys, splits, mu):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
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


# Frozen V29 correctness protocol. Main explicitly selects bf16, unlike the
# original four cases above. Retain the original tolerances; never tune them
# after observing GPU failures. pt=0 tests the primitive, not core eligibility.
@pytest.mark.parametrize('prefix', [0, 31, 65, 844, 1793])
@pytest.mark.parametrize('splits', [1, 2])
@pytest.mark.parametrize('output', [True, False], ids=['dense-output', 'observe-only'])
def test_main_bf16_fused_observe_fp32_oracle(prefix, splits, output):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe

    nq, h, hk, d, rank = 256, 16, 2, 512, 32
    keys = prefix + nq
    pt = prefix // 64
    g = torch.Generator(device='cuda').manual_seed(2902 + prefix)
    q = torch.randn(1, h, nq, d, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    sketch = torch.randn(1, hk, keys, rank, device='cuda', dtype=torch.float32, generator=g)
    scale = d ** -.5
    summary = allocate_summary(1, h, nq // 128, math.ceil(keys / 64), pt, rank, 'cuda', ('v29',))

    # Independent eager math, no SDPA backend or route STORE/fused helper.
    # Disable TF32 for these FP32 matmuls and restore process settings even on
    # assertion failure. BF16 input values themselves remain unchanged.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        kh = k.repeat_interleave(h // hk, 1).float()
        vh = v.repeat_interleave(h // hk, 1).float()
        zh = sketch.repeat_interleave(h // hk, 1)
        scores = torch.matmul(q.float(), kh.transpose(-1, -2)) * scale
        dense = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2)
        out, tail = fused_observe(q, k, v, sketch, scale, pt, summary,
                                  splits=splits, mu=True, mu_precision='bf16', output=output)
        assert tail.shape == (1, h, nq, keys - pt * 64)
        torch.testing.assert_close(tail, scores[..., pt * 64:], atol=1e-4, rtol=1e-4)
        if output:
            assert out.shape == (1, nq, h, d)
            torch.testing.assert_close(out.float(), dense, atol=2e-2, rtol=2e-2)
        else:
            assert out is None
        if pt == 0:
            assert summary is None
        for tile in range(pt):
            lo, hi = tile * 64, (tile + 1) * 64
            tile_scores = scores[..., lo:hi]
            want_z = torch.logsumexp(tile_scores, dim=-1)
            want_mu = torch.matmul(torch.softmax(tile_scores, dim=-1), zh[..., lo:hi, :])
            got_z = summary.z[:, :, :, tile, :].reshape(1, h, nq)
            got_mu = summary.mu[:, :, :, tile, :, :].reshape(1, h, nq, rank)
            torch.testing.assert_close(got_z, want_z, atol=1e-2, rtol=1e-3)
            torch.testing.assert_close(got_mu, want_mu, atol=1e-2, rtol=1e-2)
            assert bool(summary.active[:, :, :, tile, :].eq(1).all())
            assert bool(summary.bad[:, :, :, tile, :].eq(0).all())
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
