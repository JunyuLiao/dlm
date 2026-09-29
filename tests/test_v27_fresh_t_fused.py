"""v27 fused fresh-T consumer: parity with an independent torch reference of the same greedy rule (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')

H, HK, D = 16, 2, 512


def _inputs(keys, nq=256, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(1, H, nq, D, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
    z = torch.randn(1, HK, keys, 32, device='cuda', generator=g)
    ref = torch.rand(1, HK, device='cuda', generator=g) + .5
    sens = torch.rand(1, nq, device='cuda', generator=g) + .5
    return q, k, v, z, ref, sens


def _reference(q, k, v, z, ref, sens, scale, threshold, splits, held=None):
    """Greedy retained-state rule per (head, 64-row block, KV split), vectorized over programs.
    Returns (output [1,Q,H,D] fp32, kept [H,MB,KT] bool, visited [H,MB,KT] bool)."""
    nq, nk = q.shape[2], k.shape[2]
    mb, kt = math.ceil(nq / 64), math.ceil(nk / 64)
    pad_q, pad_k = mb * 64 - nq, kt * 64 - nk
    group = H // HK
    qf = torch.nn.functional.pad(q[0].float(), (0, 0, 0, pad_q)).view(H, mb, 64, D)
    kf = torch.nn.functional.pad(k[0].float(), (0, 0, 0, pad_k)).repeat_interleave(group, 0)
    zf = torch.nn.functional.pad(z[0].float(), (0, 0, 0, pad_k)).repeat_interleave(group, 0)
    rows = (torch.arange(mb * 64, device='cuda') < nq).view(mb, 64)
    cols = torch.arange(kt * 64, device='cuda') < nk
    sf = torch.nn.functional.pad(sens[0].float(), (0, pad_q), value=1.).view(mb, 64)
    reff = ref[0].float().clamp_min(1e-12).repeat_interleave(group)[:, None, None]
    kept = torch.zeros(H, mb, kt, dtype=torch.bool, device='cuda')
    visited = torch.zeros_like(kept)
    per = math.ceil(kt / splits)
    for sp in range(splits):
        previous = torch.full((H, mb, 64), -math.inf, device='cuda')
        projected = torch.zeros(H, mb, 64, 32, device='cuda')
        for j in range(sp * per, min(sp * per + per, kt)):
            go = torch.ones(H, mb, dtype=torch.bool, device='cuda') if held is None else held[:, :, j]
            visited[:, :, j] = go
            kj = kf[:, j * 64:(j + 1) * 64]
            s = torch.einsum('hmrd,hkd->hmrk', qf, kj) * scale
            s = s.masked_fill(~(rows[None, :, :, None] & cols[None, None, None, j * 64:(j + 1) * 64]), -math.inf)
            tile_max = s.amax(-1)
            active = tile_max > -math.inf
            p = torch.exp(s - torch.where(active, tile_max, 0.)[..., None])
            ell = p.sum(-1)
            block_z = torch.where(active, tile_max + ell.clamp_min(1e-30).log(), -math.inf)
            mu = torch.einsum('hmrk,hkc->hmrc', p / ell.clamp_min(1e-30)[..., None], zf[:, j * 64:(j + 1) * 64])
            combined = torch.logaddexp(previous, block_z)
            safe = torch.where(combined > -math.inf, combined, 0.)
            alpha = torch.where(active, torch.exp(block_z - safe), 0.)
            norm = (alpha[..., None] * (mu - projected)).norm(dim=-1)
            risk = (norm / reff).log() + sf.log()[None]
            risk = torch.where(active, torch.where(previous > -math.inf, risk, math.inf), -math.inf)
            keep = go & (risk.amax(-1) >= threshold)
            kept[:, :, j] = keep
            old = torch.where(previous > -math.inf, torch.exp(previous - safe), 0.)
            upd = keep[..., None]
            projected = torch.where(upd[..., None], old[..., None] * projected + alpha[..., None] * mu, projected)
            previous = torch.where(upd, combined, previous)
    mask = kept.repeat_interleave(64, 1).repeat_interleave(64, 2)[:, :nq, :nk]
    vr = v[0].float().repeat_interleave(group, 0)
    s = torch.einsum('hqd,hkd->hqk', q[0].float(), k[0].float().repeat_interleave(group, 0)) * scale
    out = torch.einsum('hqk,hkd->hqd', s.masked_fill(~mask, -math.inf).softmax(-1), vr)
    return out.transpose(0, 1).unsqueeze(0), kept, visited


def _split_counts(flags, splits):
    h, mb, kt = flags.shape
    per = math.ceil(kt / splits)
    return torch.stack([flags[:, :, sp * per:sp * per + per].sum(-1) for sp in range(splits)], -1)


@pytest.mark.parametrize('keys,nq,splits', [(1537, 256, 2), (4096, 256, 1), (3000, 200, 4)])
def test_all_kept_equals_dense(keys, nq, splits):
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64_fresh_t, dense64
    q, k, v, z, ref, sens = _inputs(keys, nq)
    got, cnt = consume64_fresh_t(q, k, v, z, ref, sens, D ** -.5, -math.inf, splits=splits)
    dense = dense64(q, k, v, D ** -.5, splits=splits)
    torch.testing.assert_close(got.float(), dense.float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(cnt[..., 0], cnt[..., 1])
    kt = math.ceil(keys / 64)
    assert int(cnt[0, 0, :, 1].sum()) == kt


def test_infinite_threshold_keeps_first_support_per_split():
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64_fresh_t
    q, k, v, z, ref, sens = _inputs(2048, 256, seed=1)
    got, cnt = consume64_fresh_t(q, k, v, z, ref, sens, D ** -.5, math.inf, splits=2)
    want, kept, _ = _reference(q, k, v, z, ref, sens, D ** -.5, math.inf, 2)
    assert (cnt[..., 0] == 1).all()
    assert torch.equal(cnt[..., 0], _split_counts(kept, 2).int())
    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('threshold', [-1.5, -0.75])
@pytest.mark.parametrize('splits', [1, 2])
def test_greedy_decisions_match_reference(threshold, splits):
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64_fresh_t
    q, k, v, z, ref, sens = _inputs(1600, 256, seed=2)
    got, cnt = consume64_fresh_t(q, k, v, z, ref, sens, D ** -.5, threshold, splits=splits,
                                 mu_precision='tf32x3')
    want, kept, _ = _reference(q, k, v, z, ref, sens, D ** -.5, threshold, splits)
    ref_counts = _split_counts(kept, splits).int()
    total = kept.numel()
    frac = kept.float().mean().item()
    assert .05 < frac < .95, f'threshold {threshold} does not exercise skipping ({frac:.3f} kept)'
    same = (cnt[..., 0] == ref_counts).all(-1)
    assert same.float().mean() >= .98, f'{int((~same).sum())} of {same.numel()} programs differ'
    per_row = same.repeat_interleave(64, 1)[:, :q.shape[2]].transpose(0, 1)[None]
    torch.testing.assert_close(got.float()[per_row[..., None].expand_as(got)],
                               want[per_row[..., None].expand_as(want)], atol=2e-2, rtol=2e-2)
    assert int(cnt[..., 1].sum()) == total


def test_held_bitmap_restricts_visits_and_matches_consume64():
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64, consume64_fresh_t
    q, k, v, z, ref, sens = _inputs(2000, 256, seed=3)
    kt = math.ceil(2000 / 64)
    g = torch.Generator(device='cuda').manual_seed(4)
    skipped = torch.rand((1, H, 2, kt), device='cuda', generator=g) > .5
    skipped[..., 0] = False
    elig = torch.ones_like(skipped)
    got, cnt = consume64_fresh_t(q, k, v, z, ref, sens, D ** -.5, -math.inf, splits=2,
                                 skipped=skipped, eligible=elig)
    base = consume64(q, k, v, skipped, elig, D ** -.5, splits=2)
    torch.testing.assert_close(got.float(), base.float(), atol=2e-2, rtol=2e-2)
    held = (~skipped[0]).repeat_interleave(2, 1)  # [H, MB64, KT] from the Q128 bitmap
    assert torch.equal(cnt[..., 1], _split_counts(held, 2).int())
    got2, cnt2 = consume64_fresh_t(q, k, v, z, ref, sens, D ** -.5, -2.0, splits=2,
                                   skipped=skipped, eligible=elig, mu_precision='tf32x3')
    want2, kept2, visited2 = _reference(q, k, v, z, ref, sens, D ** -.5, -2.0, 2, held=held)
    assert torch.equal(cnt2[..., 1], _split_counts(visited2, 2).int())
    same = (cnt2[..., 0] == _split_counts(kept2, 2).int()).all(-1)
    assert same.float().mean() >= .98


def test_bf16_mu_runs_and_rejects_bad_shapes():
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64_fresh_t
    q, k, v, z, ref, sens = _inputs(1024, 128, seed=5)
    got, cnt = consume64_fresh_t(q, k, v, z, ref, None, D ** -.5, -2.0, mu_precision='bf16')
    assert got.shape == (1, 128, H, D) and torch.isfinite(got.float()).all()
    assert (cnt[..., 0] <= cnt[..., 1]).all() and (cnt[..., 0] >= 1).all()
    with pytest.raises(ValueError):
        consume64_fresh_t(q, k, v, z[..., :16], ref, None, D ** -.5, 0.)
    with pytest.raises(ValueError):
        consume64_fresh_t(q, k, v, z, ref, None, D ** -.5, 0., skipped=torch.zeros(1, dtype=torch.bool,
                                                                                    device='cuda'))
