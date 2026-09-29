"""v27: each config guard is hit on its own, after a passing control."""
import copy

import pytest

from experiments.numerical_qk_reuse import v21

SCOPE = 'GLOBAL_ONLY_NATIVE_LOCAL'
M1, M3, B = 'M1_R1_A8_current_output', 'M3_R3_A8_current_output', 'B_A8_matched'
BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}
MAIN = dict(output_score_precision='fp32_scores_bf16_pv', output_layout='model_major',
            bootstrap_policy='native_bootstrap2_observe1', observation_producer='grouped_q',
            route_storage='aligned16_odd')


def build(arm=M3, **extra):
    return v21.effective_config(copy.deepcopy(BASE), arm, SCOPE, **MAIN, **extra)


def test_controls_build_and_validate():
    for arm, extra in ((M1, {}), (M1, dict(mu_mode='pooled')), (M3, dict(score_period=16)),
                       (M3, dict(score_period=64)), (M3, dict(decision_interval=6)),
                       (M3, dict(score_period=16, decision_interval=6, threshold_shift='plus_ln2')),
                       (B, dict(hold_only=True)), (B, dict(hold_only=True, score_period=16)),
                       (B, dict(hold_only=True, score_period=64, threshold_shift='minus_ln2')),
                       (M3, dict(score_period=64, min_route_keys=8192)),
                       (B, dict(hold_only=True, score_period=64, min_route_keys=8192)),
                       (M3, dict(score_period=64, route_layers='mid3')),
                       (M3, dict(score_period=64, share_layers='pairs')),
                       (M3, dict(score_period=64, route_layers='mid3', share_layers='mid3_one')),
                       (M1, dict(share_layers='one')),
                       (M3, dict(score_period=64, consumer64=2)),
                       (M3, dict(score_period=64, consumer64=2, fused_observe=True)),
                       (M3, dict(score_period=64, decision_interval=12))):
        config = build(arm, **extra)
        v21.validate_effective(config, config['condition'])


def test_old_configs_keep_their_fingerprint_without_new_keys():
    config = build(M3, score_period=16)
    assert not {'decision_interval', 'hold_only', 'threshold_shift'} & set(config)


@pytest.mark.parametrize('arm,extra', [
    (M1, dict(mu_mode='pooled', bootstrap_policy=None)),
    (M3, dict(score_period=12)),
    (M3, dict(score_period=32)),
    (M3, dict(decision_interval=4)),
    (M3, dict(decision_interval=24)),
    (M1, dict(decision_interval=6)),
    (B, dict(decision_interval=6)),
    (M3, dict(hold_only=True)),
    (B, dict(hold_only='yes')),
    (M3, dict(threshold_shift='plus_ln3')),
    (M3, dict(score_period=16, bootstrap_policy=None)),
    (M3, dict(min_route_keys=4096)),
    (M3, dict(min_route_keys=8192, bootstrap_policy=None)),
    (M3, dict(route_layers='first2')),
    (M3, dict(share_layers='triples')),
    (M3, dict(share_layers='mid3_one')),
    (M3, dict(consumer64=3)),
    (M3, dict(consumer64=2, fused_observe=True)),
    (M3, dict(score_period=64, fused_observe=True)),
])
def test_single_bad_field_is_rejected(arm, extra):
    kwargs = dict(MAIN)
    kwargs.update(extra)
    with pytest.raises(ValueError):
        v21.effective_config(copy.deepcopy(BASE), arm, SCOPE, **kwargs)


@pytest.mark.parametrize('key,value', [('decision_interval', 5), ('hold_only', False),
                                       ('threshold_shift', 'p'), ('score_period', 24)])
def test_validate_rejects_tampered_identity(key, value):
    config = build(B if key == 'hold_only' else M3,
                   **({'hold_only': True} if key == 'hold_only' else
                      {'decision_interval': 6} if key == 'decision_interval' else
                      {'threshold_shift': 'plus_ln2'} if key == 'threshold_shift' else
                      {'score_period': 16}))
    config[key] = value
    with pytest.raises(ValueError):
        v21.validate_effective(config, config['condition'])


def test_new_keys_change_the_fingerprint():
    prints = {build(M3)['fingerprint'], build(M3, decision_interval=6)['fingerprint'],
              build(M3, threshold_shift='plus_ln2')['fingerprint'],
              build(M3, threshold_shift='minus_ln2')['fingerprint'], build(M3, score_period=64)['fingerprint']}
    assert len(prints) == 5


def test_fresh_fused_is_an_m1_parent_execution_variant():
    config = build(M1, consumer64=2, fresh_fused=True)
    v21.validate_effective(config, config['condition'])
    assert config['fresh_fused'] is True
    shifted = build(M1, consumer64=2, fresh_fused=True, threshold_shift='plus_ln2')
    v21.validate_effective(shifted, shifted['condition'])
    assert shifted['fingerprint'] != config['fingerprint'] != build(M1, consumer64=2)['fingerprint']


@pytest.mark.parametrize('arm,extra', [
    (M1, dict(fresh_fused=True)),
    (M3, dict(consumer64=2, fresh_fused=True)),
    (B, dict(consumer64=2, fresh_fused=True, hold_only=True)),
    (M1, dict(consumer64=2, fresh_fused=True, share_layers='one')),
    (M1, dict(consumer64=2, fresh_fused=True, score_period=64, fused_observe=True)),
    (M1, dict(consumer64=2, fresh_fused=True, mu_mode='pooled_compact')),
    (M1, dict(consumer64=2, fresh_fused='yes'))])
def test_fresh_fused_guards(arm, extra):
    with pytest.raises(ValueError):
        build(arm, **extra)
