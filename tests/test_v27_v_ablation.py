"""v27 V-term ablation variants: score-only (mass) ranking for the target-sparsity selector and rank-r projected V."""
import math
import types

import pytest
import torch

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def _mainline(**kw):
    line = dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6)
    line.update(kw)
    return line


def test_log_mass_is_the_row_share_and_keeps_never_drop_tiles():
    from experiments.numerical_qk_reuse.v27_dense_prefix import log_mass
    z = torch.log(torch.tensor([1.0, 2.0, 5.0, 2.0])).view(1, 1, 1, 4, 1).expand(1, 1, 1, 4, 128).contiguous()
    summary = types.SimpleNamespace(z=z, active=torch.ones_like(z, dtype=torch.int8))
    lognorm = torch.zeros_like(z)
    lognorm[0, 0, 0, 0] = float('inf')                       # first support: never dropped
    out = log_mass(summary, lognorm)
    assert torch.isposinf(out[0, 0, 0, 0]).all()
    torch.testing.assert_close(out[0, 0, 0, 1:, 0].exp(), torch.tensor([0.2, 0.5, 0.2]))
    inactive = types.SimpleNamespace(z=z, active=torch.zeros_like(z, dtype=torch.int8))
    assert torch.isneginf(log_mass(inactive, torch.zeros_like(z))).all()


def test_mass_topk_drops_the_lowest_mass_tiles():
    from experiments.numerical_qk_reuse.v27_dense_prefix import log_mass, topk_skip
    mass = torch.tensor([4.0, 1.0, 3.0, 0.5, 2.0, 6.0])
    z = torch.log(mass).view(1, 1, 1, 6, 1).expand(1, 1, 1, 6, 128).contiguous()
    summary = types.SimpleNamespace(z=z, active=torch.ones_like(z, dtype=torch.int8))
    ranked = log_mass(summary, torch.zeros_like(z))
    drop = topk_skip(ranked, torch.ones(1, 1, 1, 6, dtype=torch.int8), torch.ones(1, 128), torch.ones(1, 1), 128, 0.5)
    assert drop[0, 0, 0].tolist() == [False, True, False, True, True, False]   # the three smallest: 1, 0.5, 2


def test_rank_masked_projection_is_a_nested_rank_r_gaussian():
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from experiments.numerical_qk_reuse.integration import RankMaskedProjections
    base = Projections()
    full = base.get(5, 2, 512, 'gaussian', 32, 1729, 'cpu')
    masked = RankMaskedProjections(base, 8).get(5, 2, 512, 'gaussian', 32, 1729, 'cpu')
    assert masked.shape == full.shape and not masked[..., 8:].any()
    torch.testing.assert_close(masked[..., :8], full[..., :8] * math.sqrt(32 / 8))
    # rank-r JL scale: E||x R||^2 = ||x||^2 for both
    x = torch.randn(4096, 512)
    for m in (full[0], masked[0]):
        assert abs((x @ m).square().sum(-1).mean() / x.square().sum(-1).mean() - 1) < 0.1
    with pytest.raises(ValueError):
        RankMaskedProjections(base, 32)


def test_v_ablation_config_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    cfg = v21.effective_config(dict(BASE), arm, scope, proj_rank=8, **_mainline(threshold_shift='minus_ln2'))
    v21.validate_effective(cfg, cfg['condition'])
    cfg = v21.effective_config(dict(BASE), arm, scope, risk_topk='k30', risk_value='mass', **_mainline())
    v21.validate_effective(cfg, cfg['condition'])
    with pytest.raises(ValueError, match='risk value'):
        v21.effective_config(dict(BASE), arm, scope, risk_value='mass', **_mainline())
    with pytest.raises(ValueError, match='projected-V rank'):
        v21.effective_config(dict(BASE), arm, scope, proj_rank=8, mu_mode='pooled_compact', **_mainline())
    with pytest.raises(ValueError, match='projected-V rank'):
        v21.effective_config(dict(BASE), arm, scope, proj_rank=12, **_mainline())
