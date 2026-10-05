"""v27 q64: 64-row FA4 keep maps refined from the dense-prefix risk table (the same worst-row rule per 64-row half)."""
import math
import os

import pytest
import torch

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def _mainline(**kw):
    line = dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6)
    line.update(kw)
    return line


def _state(pt, h=16, qb=2, seed=0):
    from experiments.numerical_qk_reuse.v27_dense_prefix import DensePrefixState
    g = torch.Generator().manual_seed(seed)
    lognorm = torch.randn(1, h, qb, pt, 128, generator=g) * 2 - 4
    lognorm[..., 0, :] = float('inf')                      # first support: never dropped
    lognorm[0, 0, 0, 1, :5] = float('-inf')                # some inactive rows
    lognorm[..., 1:pt // 2, 64:] = -20.                    # tiles only the first 64-row half may need
    eligible = (torch.rand(1, h, qb, pt, generator=g) > .1).to(torch.int8)
    eligible[..., 0] = 1                                   # the first-support tile is always eligible
    z = torch.zeros(1, h, qb, 128)
    return DensePrefixState(lognorm=lognorm, eligible=eligible, bad=torch.zeros_like(eligible), previous=z,
                            projected=torch.zeros(1, h, qb, 128, 32), prefix_tiles=pt, identity=('t', seed))


def _brute(state, ref, sens, thr, rows):
    """Keep map of the worst-row rule over query groups of `rows` rows (prefix tiles only)."""
    b, h, qb, pt, _ = state.lognorm.shape
    kh = torch.arange(h) // (h // ref.shape[1])
    risk = state.lognorm - torch.log(ref)[:, kh][:, :, None, None, None] + torch.log(sens).view(1, 1, qb, 1, 128)
    risk = risk.permute(0, 1, 2, 4, 3).reshape(b, h, qb * 128, pt)
    worst = risk.view(b, h, qb * 128 // rows, rows, pt).amax(3)
    elig = state.eligible.bool().repeat_interleave(128 // rows, dim=2)
    return elig & (worst >= thr)


def test_refined_map_is_the_64_row_rule_and_a_subset_of_the_128_row_map():
    from experiments.numerical_qk_reuse.v27_dense_prefix import refine_q64
    pt, kt, thr = 40, 48, -3.87
    state = _state(pt)
    ref = torch.tensor([[1.3, 0.8]])
    sens = 1 + 3 * torch.rand(1, 256, generator=torch.Generator().manual_seed(1))
    kept128 = torch.ones(1, 16, 2, kt, dtype=torch.bool)
    kept128[..., :pt] = _brute(state, ref, sens, thr, 128)
    eligible = torch.ones_like(kept128)
    eligible[..., :pt] = state.eligible.bool()
    skipped = eligible & ~kept128
    kept64 = refine_q64(state, skipped, eligible, ref, sens, thr, 256)
    assert kept64.shape == (1, 16, 4, kt)
    assert not (kept64 & ~kept128.repeat_interleave(2, dim=2)).any()           # subset of the executed 128 map
    assert torch.equal(kept64[..., :pt], _brute(state, ref, sens, thr, 64))     # exact 64-row rule
    assert torch.equal(kept64[..., pt:], kept128.repeat_interleave(2, dim=2)[..., pt:])   # tail copied
    assert kept64[..., 0].all()                                                  # first support kept
    assert kept64.sum() < kept128.repeat_interleave(2, dim=2).sum()              # strictly finer here


def test_refine_without_sensitivity_and_fallbacks():
    from experiments.numerical_qk_reuse.v27_dense_prefix import refine_q64
    state = _state(12, seed=3)
    ref = torch.tensor([[1.0, 1.0]])
    eligible = torch.ones(1, 16, 2, 16, dtype=torch.bool)
    eligible[..., :12] = state.eligible.bool()                 # the router's prefix eligibility is the state's
    out = refine_q64(state, torch.zeros_like(eligible), eligible, ref, None, -3.87, 256)
    torch.testing.assert_close(out[..., :12], _brute(state, ref, torch.ones(1, 256), -3.87, 64))
    assert refine_q64(state, torch.zeros_like(eligible), eligible, ref, None, -3.87, 200) is None   # rows do not fill the blocks


def test_q64_config_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    cfg = v21.effective_config(dict(BASE), arm, scope, q_block=64, carry_first=True,
                               **_mainline(threshold_shift='minus_ln2'))
    v21.validate_effective(cfg, cfg['condition'])
    assert cfg['q_block'] == 64
    with pytest.raises(ValueError, match='q64'):
        v21.effective_config(dict(BASE), arm, scope, q_block=64, risk_topk='k30', **_mainline())
    with pytest.raises(ValueError, match='q64'):
        v21.effective_config(dict(BASE), arm, scope, q_block=32, **_mainline())
    with pytest.raises(ValueError, match='q64'):
        v21.effective_config(dict(BASE), arm, scope, q_block=64, **_mainline(fa4_consumer=False))


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('V27_FA4_OVERLAY'),
                    reason='requires CUDA and V27_FA4_OVERLAY')
@pytest.mark.parametrize('keys', [1100, 4133])
def test_fa4_64_row_block_lists_match_reference(keys):
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(keys)
    q = torch.randn(1, 256, 16, 512, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
    k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    kt = math.ceil(keys / 64)
    kept = torch.rand(1, 16, 4, kt, device='cuda', generator=g) > .6
    kept[..., 0] = True
    lists = v27_fa4.block_sparse_tensors(kept, q_block=64)
    got = v27_fa4.sparse_lists(q, k, v, lists, 512 ** -.5).float()
    h, hk = q.shape[1], k.shape[1]
    kr, vr = k.repeat_interleave(h // hk, 1).float(), v.repeat_interleave(h // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * 512 ** -.5
    m = kept.repeat_interleave(64, 2).repeat_interleave(64, 3)[:, :, :256, :keys]
    ref = torch.einsum('bhqk,bhkd->bhqd', s.masked_fill(~m, float('-inf')).softmax(-1), vr).transpose(1, 2)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
