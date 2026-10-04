"""v31 selection-granularity ladder on the MAGE port: kvhead / qhead / qblock / qblock_max units and the keep-fraction
budget (GPU only)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')


def _select(gran, q, k, v, prefix, n, frac=None, mage_k=64 * 4):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='mage', mage_k=mage_k,
                          mage_select='fa4', mage_granularity=gran, mage_keep_frac=frac)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    return a._mage_select_fa4(q, k, v, 0.5, prefix, n)[1], a


def _directions(g, D, m):
    d = torch.linalg.qr(torch.randn(D, m, device='cuda', generator=g))[0].T                  # m orthonormal rows
    return [d[i] for i in range(m)]


def test_units_follow_their_own_queries():
    g = torch.Generator(device='cuda').manual_seed(21)
    H, HK, D, n, pt = 16, 2, 512, 256, 60
    prefix = 64 * pt
    q = torch.randn(1, H, n, D, device='cuda', generator=g) * 0.05
    k = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.05
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g)
    a_, b_, c_, w_ = _directions(g, D, 4)
    q[0, 0] += a_ * 3                                    # head 0 (KV group 0) likes tile 5
    k[0, 0, 5 * 64:6 * 64] += a_ * 3
    q[0, 1, :128] += b_ * 3                              # head 1, block 0 likes tile 11; block 1 likes tile 23
    q[0, 1, 128:] += c_ * 3
    k[0, 0, 11 * 64:12 * 64] += b_ * 3
    k[0, 0, 23 * 64:24 * 64] += c_ * 3
    q[0, 2, 7] += w_ * 8                                 # one row of head 2 looks for a needle in tile 40
    k[0, 0, 40 * 64:40 * 64 + 2] += w_ * 10
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    kv, _ = _select('kvhead', q, k, v, prefix, n, mage_k=64 * 2)
    qh, _ = _select('qhead', q, k, v, prefix, n, mage_k=64 * 2)
    qbk, _ = _select('qblock', q, k, v, prefix, n, mage_k=64 * 1)
    qbm, a = _select('qblock_max', q, k, v, prefix, n, mage_k=64 * 1)
    # kvhead: one set shared by all 8 heads of a KV group; per-head units differ
    assert bool((kv[0, :8] == kv[0, :1]).all())
    assert bool(qh[0, 0, 0, 5]) and bool(qh[0, 1, 0, 11] | qh[0, 1, 0, 23])
    # qblock: head 1 keeps tile 11 for block 0 and tile 23 for block 1
    assert bool(qbk[0, 1, 0, 11]) and bool(qbk[0, 1, 1, 23]) and not bool(qbk[0, 1, 0, 23])
    # qblock_max: the single needle row of head 2 decides block 0; the block mean does not
    assert bool(qbm[0, 2, 0, 40]) and not bool(qbk[0, 2, 0, 40])
    for kept in (kv, qh, qbk, qbm):
        assert bool(kept[0, :, :, pt:].all())            # canvas always kept
    assert a.calls['mage_kept_prefix_tiles'] == H * 2 * 1 and a.calls['mage_prefix_tiles'] == H * 2 * pt


def test_keep_fraction_matches_topk_rule():
    g = torch.Generator(device='cuda').manual_seed(22)
    H, HK, D, n, pt = 16, 2, 512, 200, 77
    prefix = 64 * pt + 9                                 # partial boundary tile, 2 query blocks with padding
    q = (torch.randn(1, H, n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    k = (torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g).to(torch.bfloat16)
    want = pt - math.floor((1 - 0.12) * pt + 1e-9)
    for gran in ('kvhead', 'qhead', 'qblock', 'qblock_max'):
        kept, _ = _select(gran, q, k, v, prefix, n, frac=0.12)
        assert bool((kept[0, :, :, :pt].sum(-1) == want).all()), gran
        assert bool(kept[0, :, :, pt:].all()), gran
    try:
        _select('qblock', q, k, v, prefix, n, frac=0.0)
        raise AssertionError('keep fraction 0 accepted')
    except ValueError:
        pass
