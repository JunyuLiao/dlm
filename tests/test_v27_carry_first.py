"""v27 first-call carry (carry_first): canvas call 0 reuses the previous canvas's decision; config guards."""
import pytest

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def _mainline():
    return dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6,
                threshold_shift='minus_ln2')


def test_carry_first_config_and_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    config = v21.effective_config(dict(BASE), arm, scope, carry_first=True, **_mainline())
    assert config['carry_first'] is True
    v21.validate_effective(config, config['condition'])
    default = v21.effective_config(dict(BASE), arm, scope, **_mainline())
    assert 'carry_first' not in default                       # old fingerprints unchanged
    assert config['fingerprint'] != default['fingerprint']
    with pytest.raises(ValueError, match='first-call carry'):
        v21.effective_config(dict(BASE), arm, scope, carry_first=True, carry_canvases=2, **_mainline())
    no_fa4 = dict(_mainline(), fa4_consumer=False)
    with pytest.raises(ValueError):
        v21.effective_config(dict(BASE), arm, scope, carry_first=True, **no_fa4)


def test_carry_first_with_hold_only_b():
    from experiments.numerical_qk_reuse import v21
    line = {k: v for k, v in _mainline().items() if k not in ('decision_interval', 'risk_state', 'threshold_shift')}
    config = v21.effective_config(dict(BASE), 'B_A8_matched', 'GLOBAL_ONLY_NATIVE_LOCAL', hold_only=True,
                                  route_pipeline=True, carry_first=True, **line)
    v21.validate_effective(config, config['condition'])
    assert config['carry_first'] is True and config['hold_only'] is True
