"""v31 chunked dense-prefix build matches the sequential v27 build (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton kernels need CUDA')


def _summary(pt, h=16, hk=2, nq=256, seed=0, rank=32):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    g = torch.Generator(device='cuda').manual_seed(seed)
    qb = -(-nq // 128)
    s = allocate_summary(1, h, qb, pt + 4, pt, rank, 'cuda', ('t', seed))
    s.z.copy_(torch.randn(s.z.shape, device='cuda', generator=g) * 4)
    s.z[..., 3 % pt, :] -= 200.                                      # an almost massless tile
    if rank:
        s.mu.copy_(torch.randn(s.mu.shape, device='cuda', generator=g))
    s.active.fill_(1)
    s.active[0, 1, 0, 7 % pt, :50] = 0                               # partially inactive tile
    s.active[0, 2, 1, 9 % pt] = 0                                    # fully inactive (ineligible) tile
    s.bad.zero_()
    s.bad[0, 3, 0, 11 % pt, 5] = 1                                   # a bad row
    return s


@pytest.mark.parametrize('pt,compact', [(1020, False), (77, False), (300, True), (5, False)])
def test_chunked_matches_sequential(pt, compact):
    from experiments.numerical_qk_reuse import v27_dense_prefix as dp
    from experiments.numerical_qk_reuse.v31_dp_chunked import build_chunked
    s = _summary(pt, seed=pt, rank=0 if compact else 32)
    pooled = torch.randn(1, 2, pt + 4, 32, device='cuda') if compact else None
    ref = dp.build(s, 256, pt + 4, 2, pooled=pooled)
    new = build_chunked(s, 256, pt + 4, 2, pooled=pooled)
    assert torch.equal(ref.eligible, new.eligible) and torch.equal(ref.bad, new.bad)
    finite = torch.isfinite(ref.lognorm)
    assert torch.equal(finite, torch.isfinite(new.lognorm))
    assert torch.equal(ref.lognorm[~finite], new.lognorm[~finite])
    torch.testing.assert_close(new.lognorm[finite], ref.lognorm[finite], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new.previous, ref.previous, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(new.projected, ref.projected, rtol=1e-4, atol=1e-5)
