"""v31 pooled residual: with tile-constant prefix keys the centroid approximation is exact, so sparse + residual must
equal dense attention for any kept set (math on torch; end to end through the adapter's paged FA4 split path)."""
import math

import pytest
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _problem(seed, H=16, HK=2, D=128, n=200, pt=30, dtype=torch.float32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    prefix = 64 * pt
    nk = prefix + n
    q = torch.randn(1, H, n, D, device=DEV, generator=g) * 0.5
    k = torch.randn(1, HK, nk, D, device=DEV, generator=g) * 0.5
    first = k[0, :, :prefix].reshape(HK, pt, 64, D)[:, :, :1].clone()             # one key per prefix tile
    k[0, :, :prefix] = first.expand(HK, pt, 64, D).reshape(HK, prefix, D)
    v = torch.randn(1, HK, nk, D, device=DEV, generator=g)
    qb, kt = -(-n // 128), -(-nk // 64)
    kept = torch.rand(H, qb, kt, device=DEV, generator=g) < 0.3
    kept[:, :, pt:] = True                                              # canvas / boundary tiles always kept
    kept[:, -1, :pt] = True                                             # the last block keeps every prefix tile
    return q.to(dtype), k.to(dtype), v.to(dtype), kept, prefix, pt


def _attend(q, k, v, scale, allow=None):
    """fp32 reference: o [n, H, D], lse [H, n]; allow [H, n, nk] restricts the keys."""
    H, HK = q.shape[1], k.shape[1]
    kk = k[0].float().repeat_interleave(H // HK, 0)
    vv = v[0].float().repeat_interleave(H // HK, 0)
    s = q[0].float() @ kk.transpose(1, 2) * scale
    if allow is not None:
        s = s.masked_fill(~allow, float('-inf'))
    lse = s.logsumexp(-1)
    return (torch.softmax(s, -1) @ vv).transpose(0, 1), lse


def test_centroid_residual_is_exact_for_tile_constant_keys():
    from experiments.numerical_qk_reuse import v31_residual as rs
    q, k, v, kept, prefix, pt = _problem(31)
    n, scale = q.shape[2], 0.125
    rows = torch.arange(n, device=DEV) // 128
    allow = kept[:, rows].repeat_interleave(64, -1)[..., :k.shape[2]]                # [H, n, nk]
    dense, _ = _attend(q, k, v, scale)
    o_k, lse_k = _attend(q, k, v, scale, allow)
    kb, vb = rs.tile_means(k, v, pt)
    o_d, lse_d = rs.residual_partial(q, kb, vb, kept, pt, scale)
    got = rs.merge(torch.stack([o_k, o_d[0]]), torch.stack([lse_k, lse_d[0]]), torch.float32)[0]
    assert torch.allclose(got, dense, atol=1e-4, rtol=1e-4)
    assert (o_k - dense).abs().max() > 1e-2                                          # the residual matters
    assert bool(torch.isneginf(lse_d[0, :, 128:]).all())                             # all-kept block: weight 0
    assert bool((o_d[0, 128:] == 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')
def test_adapter_sparse_lists_with_residual_matches_dense():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    H, HK, D, n, pt = 16, 2, 512, 256, 40
    q, k, v, kept, prefix, _ = _problem(32, H=H, HK=HK, D=D, n=n, pt=pt, dtype=torch.bfloat16)
    nk, page = prefix + n, 64
    pages = nk // page
    table = torch.randperm(pages, device='cuda').to(torch.int32)
    cache = torch.zeros(pages, HK, page, 2 * D, device='cuda', dtype=torch.bfloat16)    # vLLM layout
    kc, vc = cache.transpose(1, 2).split(D, dim=-1)                                       # [pages, page, HK, D]
    kc[table.long()] = k[0].transpose(0, 1).reshape(pages, page, HK, D)
    vc[table.long()] = v[0].transpose(0, 1).reshape(pages, page, HK, D)
    lists = v27_fa4.block_sparse_tensors(kept[None])
    scale = D ** -0.5
    out = {}
    for mode in (None, 'centroid'):
        a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='mage', residual=mode)
        a._kept_prefix, a._prefix_total = None, 0
        a.paged = dict(k=kc, v=vc, table=table, nk=nk, prefix=prefix)
        a._cur_layer = 5
        out[mode] = a.sparse_lists(None, q, k, v, lists, scale)[0].float()             # [n, H, D]
    dense, _ = _attend(q, k, v, scale)
    err_plain = (out[None] - dense).abs().max().item()
    err_res = (out['centroid'] - dense).abs().max().item()
    assert err_res < 3e-2 < err_plain, (err_res, err_plain)
    assert a.calls['residual_calls'] == 1 and a.calls['residual_centroid_builds'] == 1
    assert a._residual_tiles == H * kept.shape[1] * pt / 64.0
    allk = v27_fa4._allkept(H, kept.shape[1], kept.shape[2], q.device)                # dense-routed: no residual
    a.sparse_lists(None, q, k, v, allk, scale)
    assert a.calls['residual_calls'] == 1
