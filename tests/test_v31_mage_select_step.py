"""v31 selection-timing control: MAGE_STEP=1 runs the first GLOBAL call of a canvas dense and selects at the second
(GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')


def test_select_step_one_delays_the_selection_by_one_exact_call():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    g = torch.Generator(device='cuda').manual_seed(17)
    H, HK, D, n, prefix = 16, 2, 512, 256, 64 * 40
    k = (torch.randn(1, HK, prefix + n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16)
    v = torch.randn(1, HK, prefix + n, D, device='cuda', generator=g).to(torch.bfloat16)
    qs = [(torch.randn(1, H, n, D, device='cuda', generator=g) * 0.3).to(torch.bfloat16) for _ in range(3)]
    b = dict(k=k, v=v)
    a = VllmMethodAdapter(types, arm='mage', mage_k=64 * 4, mage_select='fa4', mage_select_step=1)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    ref = VllmMethodAdapter(types, arm='mage', mage_k=64 * 4, mage_select='fa4')
    ref.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    out0 = a._mage(5, qs[0], b, 0.05, prefix, n)
    assert a.calls['mage_selections'] == 0 and a.calls['mage_warm_dense_calls'] == 1
    assert torch.equal(out0, v27_fa4.dense(qs[0], k, v, 0.05))          # step 0: exact
    a._mage(5, qs[1], b, 0.05, prefix, n)                                # step 1: exact + select from step-1 queries
    assert a.calls['mage_selections'] == 1
    ref._mage(5, qs[1], b, 0.05, prefix, n)                              # plain MAGE selecting from the same queries
    assert torch.equal(a.mage_state[5]['lists'].full_block_idx, ref.mage_state[5]['lists'].full_block_idx)
    a._mage(5, qs[2], b, 0.05, prefix, n)                                # step 2: sparse, held map
    assert a.calls['mage_reused_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 1
    a.canvas_id += 1                                                     # a new canvas warms up again
    a._mage(5, qs[0], b, 0.05, prefix, n)
    assert a.calls['mage_warm_dense_calls'] == 2 and a.calls['mage_selections'] == 1
    try:
        VllmMethodAdapter(types, arm='mage', mage_select_step=-1)
        raise AssertionError('negative mage_select_step accepted')
    except ValueError:
        pass
