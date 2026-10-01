"""v27 cross-canvas carry: map extension semantics and config guards."""
import pytest
import torch

from experiments.numerical_qk_reuse.integration import carried_map


def test_carried_map_keeps_old_prefix_decisions_and_keeps_every_newer_tile():
    torch.manual_seed(0)
    prefix, nq = 200, 256                      # old prefix 200 keys -> 3 whole prefix tiles (+ boundary tile 3)
    old_tiles = -(-(prefix + nq) // 64)        # 8 tiles in the old call
    skipped = torch.rand(1, 2, 2, old_tiles) < 0.5
    eligible = torch.rand(1, 2, 2, old_tiles) < 0.9
    keys = prefix + 2 * nq + nq                # two canvases later: prefix grew by 2 x 256, plus the canvas
    new_skipped, new_eligible = carried_map(skipped, eligible, prefix, keys)
    assert new_skipped.shape == (1, 2, 2, -(-keys // 64))
    assert torch.equal(new_skipped[..., :3], skipped[..., :3]) and torch.equal(new_eligible[..., :3], eligible[..., :3])
    assert not new_skipped[..., 3:].any() and new_eligible[..., 3:].all()


def test_carried_map_refuses_a_shrinking_extent():
    with pytest.raises(ValueError):
        carried_map(torch.zeros(1, 1, 2, 10, dtype=torch.bool), torch.ones(1, 1, 2, 10, dtype=torch.bool), 400, 300)


def _mainline():
    return dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64, fused_observe=True,
                async_route=True, risk_state='dense_prefix', decision_interval=6, threshold_shift='minus_ln2')


def test_carry_config_guards():
    from experiments.numerical_qk_reuse import v21
    base = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}
    config = v21.effective_config(dict(base), 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL',
                                  carry_canvases=2, **_mainline())
    assert config['carry_canvases'] == 2
    v21.validate_effective(config, config['condition'])
    with pytest.raises(ValueError, match='cross-canvas carry'):
        v21.effective_config(dict(base), 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL',
                             carry_canvases=5, **_mainline())
    no_fused = dict(_mainline(), fused_observe=False, async_route=False)
    with pytest.raises(ValueError):
        v21.effective_config(dict(base), 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL',
                             carry_canvases=2, **no_fused)
