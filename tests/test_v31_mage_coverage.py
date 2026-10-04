"""v31 coverage-adaptive budget: diffuse KV heads keep more prefix tiles than concentrated ones (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')


def test_coverage_budget_grows_for_diffuse_heads_only():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    g = torch.Generator(device='cuda').manual_seed(13)
    H, HK, D, n, prefix = 16, 2, 512, 256, 64 * 60
    q = torch.randn(1, H, n, D, device='cuda', generator=g) * 0.05
    k = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.05
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g)
    u = torch.randn(D, device='cuda', generator=g)
    u = u / u.norm()
    q[0, :8] += u * 3                                                    # KV group 0: concentrated on 3 tiles
    for t in (5, 17, 33):
        k[0, 0, t * 64:(t + 1) * 64] += u * 3
    # KV group 1 stays near uniform (diffuse attention over the whole prefix)
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    a = VllmMethodAdapter(types, arm='mage', mage_k=64 * 2, mage_select='fa4', mage_coverage=0.9)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    kept = a._mage_select_fa4(q, k, v, 1.0, prefix, n)[1]
    per_group = kept[0, [0, 8], 0, :60].sum(-1)                           # one query head of each KV group
    assert int(per_group[0]) <= 6                                        # concentrated: about the 3 heavy tiles
    assert int(per_group[1]) >= 30                                       # diffuse: most of the prefix
    assert bool(kept[0, :, :, 60:].all())                                # canvas always kept
    plain = VllmMethodAdapter(types, arm='mage', mage_k=64 * 2, mage_select='fa4')
    plain.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    base = plain._mage_select_fa4(q, k, v, 1.0, prefix, n)[1]
    assert bool((kept >= base).all())                                    # the floor keeps MAGE's own top-k
