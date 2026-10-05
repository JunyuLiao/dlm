"""v31 group-shared selection units (kvblock_max, kvhead_max) and the first-call carry (mage_carry_first) on the MAGE
port (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')

TYPES = ['sliding_attention'] * 5 + ['full_attention']


def _adapter(**kw):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    a = VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', **kw)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    return a


def _select(gran, q, k, v, prefix, n, mage_k=64 * 1, frac=None):
    a = _adapter(mage_k=mage_k, mage_granularity=gran, mage_keep_frac=frac)
    return a._mage_select_fa4(q, k, v, 0.5, prefix, n)[1], a


def test_group_units_share_one_set_per_kv_head():
    g = torch.Generator(device='cuda').manual_seed(31)
    H, HK, D, n, pt = 16, 2, 512, 256, 60
    G, prefix = H // HK, 64 * pt
    q = torch.randn(1, H, n, D, device='cuda', generator=g) * 0.05
    k = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.05
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g)
    d = torch.linalg.qr(torch.randn(D, 3, device='cuda', generator=g))[0].T
    q[0, 2, 7] += d[0] * 8                               # one row of head 2 (KV group 0, block 0): needle in tile 40
    k[0, 0, 40 * 64:40 * 64 + 2] += d[0] * 10
    q[0, 13, 200] += d[1] * 8                            # one row of head 13 (KV group 1, block 1): needle in tile 17
    k[0, 1, 17 * 64:17 * 64 + 2] += d[1] * 10
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    kvb, a = _select('kvblock_max', q, k, v, prefix, n)
    kvh, _ = _select('kvhead_max', q, k, v, prefix, n)
    qbm, b = _select('qblock_max', q, k, v, prefix, n)
    for kept in (kvb, kvh):
        for grp in range(HK):                             # identical sets within each KV group
            assert bool((kept[0, grp * G:(grp + 1) * G] == kept[0, grp * G:grp * G + 1]).all())
        assert bool(kept[0, :, :, pt:].all())             # canvas always kept
        assert bool((kept[0, :, :, :pt].sum(-1) == 1).all())   # the per-head budget is unchanged
    # the needle row decides its group's block (kvblock) or its group's whole canvas (kvhead), for every head
    assert bool(kvb[0, :G, 0, 40].all()) and bool(kvb[0, G:, 1, 17].all())
    assert bool(kvh[0, :G, :, 40].all()) and bool(kvh[0, G:, :, 17].all())
    assert bool(qbm[0, 2, 0, 40]) and bool(qbm[0, 13, 1, 17])  # per-head units find them for their own head only
    # same accounted work as the per-head unit at the same budget
    assert a.calls['mage_kept_prefix_tiles'] == b.calls['mage_kept_prefix_tiles'] == H * 2 * 1
    assert a.calls['mage_prefix_tiles'] == b.calls['mage_prefix_tiles'] == H * 2 * pt


def test_group_units_keep_fraction_budget():
    import math
    g = torch.Generator(device='cuda').manual_seed(32)
    H, HK, D, n, pt = 16, 2, 512, 200, 77
    prefix = 64 * pt + 9                                 # partial boundary tile, 2 query blocks with padding
    q = (torch.randn(1, H, n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    k = (torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g).to(torch.bfloat16)
    want = pt - math.floor((1 - 0.12) * pt + 1e-9)
    for gran in ('kvblock_max', 'kvhead_max'):
        kept, _ = _select(gran, q, k, v, prefix, n, frac=0.12)
        assert bool((kept[0, :, :, :pt].sum(-1) == want).all()), gran
        assert bool(kept[0, :, :, pt:].all()), gran


def test_carry_first_replaces_the_first_exact_call_only():
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(33)
    H, HK, D, n, prefix = 16, 2, 512, 256, 64 * 40
    k = (torch.randn(1, HK, prefix + 3 * n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    v = torch.randn(1, HK, prefix + 3 * n, D, device='cuda', generator=g).to(torch.bfloat16)
    qs = [(torch.randn(1, H, n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16) for _ in range(3)]
    cut = lambda m: dict(k=k[:, :, :m], v=v[:, :, :m])
    b1, b2 = cut(prefix + n), cut(prefix + 2 * n)
    a = _adapter(mage_k=64 * 4, mage_granularity='kvblock_max', mage_select_step=1, mage_carry_first=True)
    plain = _adapter(mage_k=64 * 4, mage_granularity='kvblock_max', mage_select_step=1)
    for x in (a, plain):
        x.canvas_id = 1
        out0 = x._mage(5, qs[0], b1, 0.05, prefix, n)                  # request's first canvas: exact call 0
        assert torch.equal(out0, v27_fa4.dense(qs[0], b1['k'], b1['v'], 0.05))
        x._mage(5, qs[1], b1, 0.05, prefix, n)                         # call 1: select
        x._mage(5, qs[2], b1, 0.05, prefix, n)                         # call 2: held
    assert a.calls.get('mage_carried_calls', 0) == 0 and a.calls['mage_warm_dense_calls'] == 1
    old = a.mage_state[5]['kept']
    assert torch.equal(a.mage_state[5]['lists'].full_block_idx, plain.mage_state[5]['lists'].full_block_idx)
    # next canvas of the same request: the prefix grew by exactly the previous canvas
    a.canvas_id = plain.canvas_id = 2
    out = a._mage(5, qs[0], b2, 0.05, prefix + n, n)
    assert a.calls['mage_carried_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 1
    want = torch.ones((1, H, 2, (prefix + 2 * n) // 64), dtype=torch.bool, device='cuda')
    want[..., :prefix // 64] = old[..., :prefix // 64]                # old prefix decisions; newer tiles kept
    assert torch.equal(out, v27_fa4.sparse_lists(qs[0], b2['k'], b2['v'], v27_fa4.block_sparse_tensors(want), 0.05))
    assert not torch.equal(out, v27_fa4.dense(qs[0], b2['k'], b2['v'], 0.05))
    ref = plain._mage(5, qs[0], b2, 0.05, prefix + n, n)               # without the carry: exact call 0
    assert torch.equal(ref, v27_fa4.dense(qs[0], b2['k'], b2['v'], 0.05)) and plain.calls['mage_warm_dense_calls'] == 2
    a._mage(5, qs[1], b2, 0.05, prefix + n, n)                         # call 1 still selects from its own queries
    plain._mage(5, qs[1], b2, 0.05, prefix + n, n)
    assert a.calls['mage_selections'] == plain.calls['mage_selections'] == 2
    assert torch.equal(a.mage_state[5]['lists'].full_block_idx, plain.mage_state[5]['lists'].full_block_idx)
    # not the direct continuation -> exact call 0: prefix not grown by the previous canvas; a different block layout;
    # a skipped canvas
    a.canvas_id = 3
    a._mage(5, qs[0], b2, 0.05, prefix + n, n)
    assert a.calls['mage_carried_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 2
    a._mage(5, qs[0][:, :, :128].contiguous(), cut(prefix + 2 * n + 128), 0.05, prefix + 2 * n, 128)
    assert a.calls['mage_carried_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 3
    a.canvas_id = 5
    a._mage(5, qs[0], cut(prefix + 3 * n), 0.05, prefix + 2 * n, n)
    assert a.calls['mage_carried_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 4


def test_carry_first_needs_a_select_step():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    for kw in (dict(), dict(mage_select_step=0)):
        try:
            VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', mage_carry_first=True, **kw)
            raise AssertionError(f'mage_carry_first accepted with {kw}')
        except ValueError:
            pass
