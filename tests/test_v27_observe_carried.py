"""v27 observe_carried: observation-only fused kernel (no V / PV) and the named-variant guards."""
import math

import pytest
import torch

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def _mainline(**kw):
    line = dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6,
                threshold_shift='minus_ln2')
    line.update(kw)
    return line


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')
@pytest.mark.parametrize('keys,splits,mu', [(1100, 2, True), (2049, 2, True), (2049, 1, True), (1100, 2, False)])
def test_observation_only_kernel_writes_identical_summaries_and_tail(keys, splits, mu):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
    g = torch.Generator(device='cuda').manual_seed(keys + splits)
    nq, h, hk, d = 256, 16, 2, 512
    q = torch.randn(1, h, nq, d, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    z = torch.randn(1, hk, keys, 32, device='cuda', generator=g)
    pt = (keys - nq) // 64
    qb, kt = math.ceil(nq / 128), math.ceil(keys / 64)
    rank = 32 if mu else 0
    full = allocate_summary(1, h, qb, kt, pt, rank, 'cuda', ('a',))
    only = allocate_summary(1, h, qb, kt, pt, rank, 'cuda', ('b',))
    out, tail = fused_observe(q, k, v, z, d ** -.5, pt, full, splits=splits, mu=mu, mu_precision='bf16')
    none, tail2 = fused_observe(q, k, v, z, d ** -.5, pt, only, splits=splits, mu=mu, mu_precision='bf16',
                                output=False)
    assert none is None and out is not None
    assert torch.equal(tail, tail2) and torch.equal(full.active, only.active) and torch.equal(full.bad, only.bad)
    if mu:
        # the mu product consumes the same exponentials in both modes: bit-identical (the c01 configuration)
        assert torch.equal(full.z, only.z) and torch.equal(full.mu, only.mu)
    else:
        # without any product on p, Triton may lay out the row sums differently: equal up to summation order
        torch.testing.assert_close(full.z, only.z, rtol=1e-6, atol=1e-6)


def test_observe_carried_config_and_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    cfg = v21.effective_config(dict(BASE), arm, scope, carry_first=True, observe_carried=True, q_block=64,
                               q_carry64=True, **_mainline())
    v21.validate_effective(cfg, cfg['condition'])
    plain = v21.effective_config(dict(BASE), arm, scope, carry_first=True, q_block=64, q_carry64=True, **_mainline())
    assert cfg['observe_carried'] is True and 'observe_carried' not in plain
    assert cfg['fingerprint'] != plain['fingerprint']
    with pytest.raises(ValueError, match='observe_carried'):
        v21.effective_config(dict(BASE), arm, scope, observe_carried=True, **_mainline())   # no carry_first
    with pytest.raises(ValueError):
        v21.effective_config(dict(BASE), arm, scope, carry_first=True, observe_carried=True,
                             **_mainline(fa4_consumer=False))
