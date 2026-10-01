"""v27 output protection (generated-token tiles never skipped) and observation step (observe at call 2)."""
import pytest
import torch

from experiments.numerical_qk_reuse.cache import ScoreCache
from experiments.numerical_qk_reuse.integration import protect_generated


def test_protect_generated_keeps_every_tile_from_the_prompt_boundary_tile_on():
    skipped = torch.ones(1, 2, 2, 10, dtype=torch.bool)
    out = protect_generated(skipped, 200)          # prompt keys 0..199: tile 3 (192..255) mixes prompt and output
    assert out is skipped                          # in place: the consumer's cached block lists see the same tensor
    assert skipped[..., :3].all() and not skipped[..., 3:].any()


def test_protect_generated_aligned_boundary_and_no_generated_tiles():
    skipped = torch.ones(1, 1, 1, 6, dtype=torch.bool)
    protect_generated(skipped, 256)                # tile 4 starts exactly at the first generated key
    assert skipped[..., :4].all() and not skipped[..., 4:].any()
    whole = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    protect_generated(whole, 64 * 5)               # extent ends inside the prompt: nothing to protect
    assert whole.all()
    with pytest.raises(ValueError):
        protect_generated(whole, None)


def test_score_clock_starts_at_the_observation_call():
    cache = ScoreCache(score_period=64, decision_interval=6)
    cache.origin = 2
    plan = lambda step: cache.plan(type('I', (), {'layer': 0})(), step)
    assert plan(2).score_refresh                   # no entry yet: initial observation
    assert (2 - cache.origin) % cache.score_period == 0 and (8 - cache.origin) % cache.score_period != 0


def _mainline():
    return dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6,
                threshold_shift='minus_ln2')


BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def test_observe_step_and_protect_output_config_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    config = v21.effective_config(dict(BASE), arm, scope, observe_step=2, protect_output=True, **_mainline())
    assert config['observe_step'] == 2 and config['protect_output'] is True
    v21.validate_effective(config, config['condition'])
    default = v21.effective_config(dict(BASE), arm, scope, **_mainline())
    assert 'observe_step' not in default and 'protect_output' not in default   # old fingerprints unchanged
    with pytest.raises(ValueError, match='observation step'):
        v21.effective_config(dict(BASE), arm, scope, observe_step=4, **_mainline())
    with pytest.raises(ValueError, match='observation step'):
        v21.effective_config(dict(BASE), arm, scope, observe_step=2, carry_canvases=2, **_mainline())
    no_fused = dict(_mainline(), fused_observe=False, async_route=False)
    with pytest.raises(ValueError):
        v21.effective_config(dict(BASE), arm, scope, protect_output=True, **no_fused)


def test_m3_parent_without_interval_override_is_r3():
    from experiments.numerical_qk_reuse import v21
    line = {k: v for k, v in _mainline().items() if k != 'decision_interval'}
    config = v21.effective_config(dict(BASE), 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL', **line)
    assert 'decision_interval' not in config
    assert v21.validate_effective(config, config['condition'])[1] == 3
