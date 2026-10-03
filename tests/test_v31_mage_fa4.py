"""v31 MAGE port: the FA4-observation selection equals the FP32 reference selection (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')


def test_mage_fa4_selection_matches_reference():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    a = VllmMethodAdapter(types, arm='mage', mage_k=64 * 6, mage_select='fa4')
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    g = torch.Generator(device='cuda').manual_seed(5)
    H, HK, D, n, prefix = 16, 2, 512, 256, 64 * 90 + 21
    q = torch.randn(1, H, n, D, device='cuda', generator=g).to(torch.bfloat16)
    k = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g).to(torch.bfloat16)
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g).to(torch.bfloat16)
    for t, w in ((4, 3.), (17, 2.5), (60, 2.), (61, 1.5)):                  # clearly separated heavy tiles
        k[0, 0, t * 64:(t + 1) * 64] += (q[0, :8].float().mean((0, 1)) * w).to(torch.bfloat16)
    scale = 0.05
    ref = a._mage_select(q, k, scale, prefix, n)
    out, new = a._mage_select_fa4(q, k, v, scale, prefix, n)
    assert torch.equal(ref, new)
    assert out.shape == (1, n, H, D)
