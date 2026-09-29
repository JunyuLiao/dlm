"""v27 M1-DP: kernels vs an independent torch reference of the dense-prefix risk (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')

H, HK, D, NQ = 16, 2, 512, 256


def _inputs(keys, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(1, H, NQ, D, device='cuda', generator=g) * .6
    k = torch.randn(1, HK, keys, D, device='cuda', generator=g) * .6
    scores = (torch.einsum('bhqd,bhkd->bhqk', q, k.repeat_interleave(H // HK, 1)) * D ** -.5).contiguous()
    z = torch.randn(1, HK, keys, 32, device='cuda', generator=g)
    ref = torch.rand(1, HK, device='cuda', generator=g) + .5
    sens = torch.rand(1, NQ, device='cuda', generator=g) + .5
    return scores, z, ref, sens


def _tiles(scores, z, compact):
    """Per (h, qb, tile, row): block log-mass, mu, active; from the scores directly (fp64)."""
    b, h, nq, nk = scores.shape
    kt, qb = math.ceil(nk / 64), math.ceil(nq / 128)
    s = torch.nn.functional.pad(scores[0].double(), (0, kt * 64 - nk), value=-math.inf)
    s = torch.nn.functional.pad(s, (0, 0, 0, qb * 128 - nq), value=-math.inf).view(h, qb, 128, kt, 64)
    zz = torch.nn.functional.pad(z[0].double(), (0, 0, 0, kt * 64 - nk)).view(HK, kt, 64, 32)
    zz = zz.repeat_interleave(h // HK, 0)                       # [h, kt, 64, 32]
    mx = s.amax(-1)
    active = mx > -math.inf
    w = torch.exp(s - torch.where(active, mx, 0.)[..., None])
    ell = w.sum(-1)
    z_blk = torch.where(active, mx + ell.clamp_min(1e-30).log(), -math.inf)
    if compact:
        count = torch.nn.functional.pad(torch.ones(nk, device='cuda'), (0, kt * 64 - nk)).view(kt, 64).sum(-1)
        pooled = zz.sum(2) / count.clamp_min(1)[None, :, None]   # [h, kt, 32]
        mu = pooled[:, None, None, :, :].expand(h, qb, 128, kt, 32)
    else:
        mu = torch.einsum('hqrtk,htkc->hqrtc', w / ell.clamp_min(1e-30)[..., None], zz)
    return z_blk, mu, active


def _reference(scores, z, ref, sens, threshold, compact=False, dense=True):
    """dense=True: M1-DP. dense=False: the kept-state M1 rule (for agreement only)."""
    b, h, nq, nk = scores.shape
    z_blk, mu, active = _tiles(scores, z, compact)
    qb, kt = z_blk.shape[1], z_blk.shape[3]
    rows = (torch.arange(qb * 128, device='cuda') < nq).view(qb, 128)
    log_ref = ref[0].double().clamp_min(1e-12).log().repeat_interleave(h // HK)[:, None]
    log_t = torch.nn.functional.pad(sens[0].double(), (0, qb * 128 - nq), value=1.).view(qb, 128).log()
    previous = torch.full((h, qb, 128), -math.inf, device='cuda', dtype=torch.float64)
    projected = torch.zeros(h, qb, 128, 32, device='cuda', dtype=torch.float64)
    drop = torch.zeros(h, qb, kt, dtype=torch.bool, device='cuda')
    elig = torch.zeros_like(drop)
    for j in range(kt):
        a = active[..., j] & rows
        e = a.any(-1)
        combined = torch.logaddexp(previous, torch.where(a, z_blk[..., j], -math.inf))
        safe = torch.where(combined > -math.inf, combined, 0.)
        alpha = torch.where(a, torch.exp(z_blk[..., j] - safe), 0.)
        norm = (alpha[..., None] * (mu[..., j, :] - projected)).norm(dim=-1)
        risk = norm.log() - log_ref[..., None] + log_t
        risk = torch.where(a, torch.where(previous > -math.inf, risk, math.inf), -math.inf)
        d = e & (risk.amax(-1) < threshold)
        drop[..., j], elig[..., j] = d, e
        join = e if dense else e & ~d
        old = torch.where(previous > -math.inf, torch.exp(previous - safe), 0.)
        projected = torch.where(join[..., None, None], old[..., None] * projected + alpha[..., None] * mu[..., j, :],
                                projected)
        previous = torch.where(join[..., None], combined, previous)
    return drop, elig


def _kernel(scores, z, ref, sens, threshold, compact, tail):
    from experiments.numerical_qk_reuse import v27_dense_prefix as dp
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only, tile_pool
    nk = scores.shape[-1]
    pt, qb, kt = (nk - NQ) // 64, math.ceil(NQ / 128), math.ceil(nk / 64)
    kw = {}
    if compact:
        pooled, count = tile_pool(z, torch.ones(1, HK, nk, dtype=torch.bool, device='cuda'))
        kw = dict(pool='compact', pooled=pooled, pool_count=count)
    summary = allocate_summary(1, H, qb, kt, pt, 0 if compact else 32, 'cuda', ('dp', nk))
    route_only(scores, z, ref, sensitivity=sens, log_threshold=threshold, summary=summary, store_summary=True,
               variant='generic', **kw)
    state = dp.build(summary, NQ, kt, HK, pooled=kw.get('pooled'))
    if tail:
        return dp.route(scores[..., pt * 64:].contiguous(), z, ref, state, sensitivity=sens,
                        log_threshold=threshold, key_offset=pt * 64, **kw)
    return dp.route(scores, z, ref, state, sensitivity=sens, log_threshold=threshold, **kw)


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('keys,threshold,tail', [(1100, -.5, False), (2049, -1.5, True), (4133, -.5, True)])
def test_dense_prefix_matches_reference(keys, threshold, tail, compact):
    scores, z, ref, sens = _inputs(keys, keys)
    got = _kernel(scores, z, ref, sens, threshold, compact, tail)
    want_drop, want_elig = _reference(scores, z, ref, sens, threshold, compact)
    assert torch.equal(got.eligible[0], want_elig)
    agree = (got.skipped[0] == want_drop).float().mean().item()
    assert agree >= .995, agree
    frac = want_drop[want_elig].float().mean().item()
    assert 0 < frac < 1


def test_dense_prefix_vs_kept_state_agreement_is_reported():
    scores, z, ref, sens = _inputs(4133, 11)
    dense, elig = _reference(scores, z, ref, sens, -.5, dense=True)
    kept, _ = _reference(scores, z, ref, sens, -.5, dense=False)
    agree = (dense == kept)[elig].float().mean().item()
    print(f'M1-DP vs M1 decision agreement on eligible tiles: {agree:.4f}; '
          f'skip {dense[elig].float().mean():.4f} vs {kept[elig].float().mean():.4f}')
    assert agree > .5
