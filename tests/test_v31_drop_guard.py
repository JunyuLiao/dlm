"""v31 dropped-mass guard: the estimated dropped share is exact for tile-constant prefix keys, and through the
adapter's paged FA4 path a (head, block) whose dropped tiles hold too much mass falls back to its whole prefix."""
import pytest
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _scores(q, k, scale):
    H, HK = q.shape[1], k.shape[1]
    return q[0].float() @ k[0].float().repeat_interleave(H // HK, 0).transpose(1, 2) * scale        # [H, n, nk]


def test_dropped_share_is_exact_for_tile_constant_keys():
    from experiments.numerical_qk_reuse import v31_residual as rs
    g = torch.Generator(device=DEV).manual_seed(41)
    H, HK, D, n, pt = 8, 2, 64, 200, 20
    prefix = 64 * pt
    q = torch.randn(1, H, n, D, device=DEV, generator=g)
    k = torch.randn(1, HK, prefix + n, D, device=DEV, generator=g) * 0.3
    first = k[0, :, :prefix].reshape(HK, pt, 64, D)[:, :, :1].clone()
    k[0, :, :prefix] = first.expand(HK, pt, 64, D).reshape(HK, prefix, D)
    v = torch.randn_like(k)
    qb, kt = 2, -(-(prefix + n) // 64)
    kept = torch.rand(H, qb, kt, device=DEV, generator=g) < 0.4
    kept[:, :, pt:] = True
    scale = 0.2
    got = rs.dropped_share(q, k, rs.tile_means(k, v, pt)[0], kept, pt, scale)
    p = torch.softmax(_scores(q, k, scale), -1)                                               # [H, n, nk]
    rows = torch.arange(n, device=DEV) // 128
    dropped = (~kept[:, rows]).repeat_interleave(64, -1)[..., :prefix + n]
    dropped[..., prefix:] = False
    exact = (p * dropped).sum(-1)                                                             # [H, n]
    want = torch.stack([exact[:, :128].mean(-1), exact[:, 128:].mean(-1)], -1)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')
def test_guard_falls_back_only_for_diffuse_units():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    g = torch.Generator(device='cuda').manual_seed(42)
    H, HK, D, n, pt = 16, 2, 512, 256, 40
    prefix = 64 * pt
    nk = prefix + n
    q = torch.randn(1, H, n, D, device='cuda', generator=g) * 0.02
    k = torch.randn(1, HK, nk, D, device='cuda', generator=g) * 0.02
    v = torch.randn(1, HK, nk, D, device='cuda', generator=g)
    u = torch.randn(D, device='cuda', generator=g)
    u = u / u.norm()
    q[0, :8] += u * 12                                   # KV group 0: ~97% of its mass on tiles 3 and 9
    for t in (3, 9):
        k[0, 0, t * 64:(t + 1) * 64] += u * 12
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)               # group 1: near uniform
    kept = torch.zeros(H, 2, -(-nk // 64), dtype=torch.bool, device='cuda')
    kept[:, :, pt:] = True
    kept[:, :, [3, 9, 20, 31]] = True                    # 10% of the prefix, including group 0's heavy tiles
    lists = v27_fa4.block_sparse_tensors(kept[None])
    page = 64
    pages = nk // page
    table = torch.randperm(pages, device='cuda').to(torch.int32)
    cache = torch.zeros(pages, HK, page, 2 * D, device='cuda', dtype=torch.bfloat16)
    kc, vc = cache.transpose(1, 2).split(D, dim=-1)
    kc[table.long()] = k[0].transpose(0, 1).reshape(pages, page, HK, D)
    vc[table.long()] = v[0].transpose(0, 1).reshape(pages, page, HK, D)
    scale = D ** -0.5
    out = {}
    for guard in (None, 0.5):
        a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='mage', drop_guard=guard)
        a._kept_prefix, a._prefix_total = None, 0
        a.paged = dict(k=kc, v=vc, table=table, nk=nk, prefix=prefix)
        a._cur_layer = 5
        out[guard] = a.sparse_lists(None, q, k, v, lists, scale)[0].float()                   # [n, H, D]
    assert a.calls['guard_units'] == H * 2 and a.calls['guard_flagged_units'] == 8 * 2      # all of group 1 only
    p = torch.softmax(_scores(q, k, scale), -1)
    dense = (p @ v[0].float().repeat_interleave(H // HK, 0)).transpose(0, 1)                  # [n, H, D]
    assert (out[0.5][:, 8:] - dense[:, 8:]).abs().max() < 5e-3                                # flagged: whole prefix
    assert (out[None][:, 8:] - dense[:, 8:]).abs().max() > 5e-2                               # ... which mattered
    assert torch.allclose(out[0.5][:, :8], out[None][:, :8], atol=1e-3)                       # unflagged unchanged
    a.sparse_lists(None, q, k, v, lists, scale)                                               # held map: cached
    assert a.calls['guard_units'] == H * 2
