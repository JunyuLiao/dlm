"""v31 two-level selection: MAGE's shared top-k plus per-query-head critical tiles (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')


def _adapter(tau):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    a = VllmMethodAdapter(types, arm='mage', mage_k=64 * 4, mage_select='fa4', mage_critical=tau)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    return a


def test_critical_tiles_are_added_only_for_the_head_that_needs_them():
    g = torch.Generator(device='cuda').manual_seed(11)
    H, HK, D, n, prefix = 16, 2, 512, 256, 64 * 60
    q = torch.randn(1, H, n, D, device='cuda', generator=g) * 0.3
    k = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.3
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g)
    u = torch.randn(D, device='cuda', generator=g)
    u = u / u.norm()
    w = torch.randn(D, device='cuda', generator=g)
    w = w - (w @ u) * u
    w = w / w.norm()
    q[0, :8] += u * 4                                                    # every query of KV group 0 likes direction u
    for t in (3, 9, 21, 40):                                             # four tiles every head of the group attends
        k[0, 0, t * 64:(t + 1) * 64] += u * 4
    q[0, 2, :10] += w * 6                                                # rows 0-9 of query head 2 look for a needle
    k[0, 0, 50 * 64:50 * 64 + 4] += w * 12                               # ...that only they match (tile 50)
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    scale = 0.5
    plain = _adapter(None)._mage_select_fa4(q, k, v, scale, prefix, n)[1]
    same = _adapter(1.01)._mage_select_fa4(q, k, v, scale, prefix, n)[1]
    assert torch.equal(plain, same)                                      # tau > 1 adds nothing
    a = _adapter(0.05)
    kept = a._mage_select_fa4(q, k, v, scale, prefix, n)[1]
    assert not bool(plain[0, 2, 0, 50])                                  # MAGE alone misses the needle tile
    assert bool(kept[0, 2, 0, 50])                                       # the two-level selection keeps it
    assert bool((kept[0, :, :, :60] >= plain[0, :, :, :60]).all())       # a superset of MAGE
    assert int(kept[0, :, 0, 50].sum()) < H                              # per head, not for every head
    assert a.calls['mage_critical_added_tiles'] >= 1
