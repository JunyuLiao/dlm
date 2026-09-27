"""CPU v21 wrapper and dispatch contract; no GPU qualification implied."""
from contextlib import contextmanager
from unittest.mock import patch
import types

import pytest

from experiments.numerical_qk_reuse import v20, v21
from experiments.numerical_qk_reuse.cached_executor import _check_output_modes


def base(**changes):
    result = dict(diagnostic=False, policy={'local': {'log_threshold': -2.},
                                            'global': {'log_threshold': -3.}},
                  consumer='triton', m_ref=1., beta=3., gamma=.5, source_hashes={})
    result.update(changes)
    return result


@pytest.mark.parametrize('precision,layout', [
    ('legacy_bf16_scores', 'head_major'),
    ('legacy_bf16_scores', 'model_major'),
    ('fp32_scores_bf16_pv', 'head_major'),
    ('fp32_scores_bf16_pv', 'model_major'),
])
def test_effective_parent_is_intact_and_fingerprinted(precision, layout):
    config = v21.effective_config(base(), 'M3_R3_A8_current_output', v20.ALL_NATIVE_LEGAL,
                                  output_score_precision=precision, output_layout=layout)
    parent = config['parent_config']
    assert v20.validate_effective(parent, config['condition']) == (v20.ALL_NATIVE_LEGAL, 3)
    assert v21.validate_effective(config, config['condition']) == (v20.ALL_NATIVE_LEGAL, 3)
    assert parent['plugin'] == v20.PLUGIN
    assert parent['output_mode'] == v20.PREQK_MODE
    if precision != 'legacy_bf16_scores' or layout != 'head_major':
        assert parent['kernel_variant'] == 'generic'
    changed = {**config, 'unfingerprinted_change': True}
    with pytest.raises(ValueError, match='fingerprint'):
        v21.validate_effective(changed, config['condition'])


def test_reject_unknown_and_unsupported_modes():
    for kw in ({'output_score_precision': 'fp64'}, {'output_layout': 'other'}):
        with pytest.raises(ValueError):
            v21.effective_config(base(), 'M1_R1_A8_current_output', v20.ALL_NATIVE_LEGAL, **kw)
    for conflict in (dict(consumer='hopper', support_build='frozen'), dict(kernel_variant='static')):
        with pytest.raises(ValueError, match='generic Triton'):
            v21.effective_config(base(**conflict), 'M1_R1_A8_current_output', v20.ALL_NATIVE_LEGAL,
                                  output_score_precision='fp32_scores_bf16_pv')
    with pytest.raises(ValueError, match='generic Triton'):
        _check_output_modes('static', 'legacy_bf16_scores', 'model_major')
    with pytest.raises(ValueError, match='generic Triton'):
        _check_output_modes('static', 'fp32_scores_bf16_pv', 'head_major')


def test_wrapper_sets_only_output_owner_and_exposes_duplicate_qk():
    config = v21.effective_config(base(), 'M3_R2_A8_current_output', v20.ALL_NATIVE_LEGAL,
                                  output_score_precision='fp32_scores_bf16_pv',
                                  output_layout='model_major')
    owner = types.SimpleNamespace(output_mode=v20.PREQK_MODE)
    seen = []

    @contextmanager
    def fake_parent(adapter, parent, condition):
        seen.append((adapter, parent, condition))
        yield dict(router=owner, state=object(), counters=lambda: {'parent': 1})

    adapter = object()
    with patch.object(v20, 'install', fake_parent):
        with v21.install(adapter, config, config['condition']) as runtime:
            owner.output_precision_extra_qk_elements_upper_bound = 2048
            counters = runtime['counters']()
            assert counters['output_score_precision'] == 'fp32_scores_bf16_pv'
            assert counters['output_layout'] == 'model_major'
            assert counters['output_precision_extra_qk_elements_upper_bound'] == 2048
            assert counters['parent'] == 1
    assert seen == [(adapter, config['parent_config'], config['condition'])]


@pytest.mark.parametrize('scope', v20.SCOPES)
def test_d_matched_control_parent_and_owner_binding(scope):
    config = v21.effective_control_config(base(), scope,
                                          output_score_precision='fp32_scores_bf16_pv',
                                          output_layout='model_major')
    assert config['condition'] == v21.CONTROL_CONDITION
    assert config['parent_kind'] == 'v20_control'
    assert v21.validate_effective(config, config['condition']) == (scope, None)
    owner = types.SimpleNamespace(output_mode=v20.PREQK_MODE)
    router = types.SimpleNamespace(owner=owner)
    seen = []

    @contextmanager
    def fake_control(adapter, parent, condition):
        seen.append((adapter, parent, condition))
        yield dict(router=router, counters=lambda: {'control': 'D_matched'})

    import experiments.numerical_qk_reuse.v20_controls as controls
    adapter = object()
    with patch.object(controls, 'install', fake_control):
        with v21.install(adapter, config, config['condition']) as runtime:
            c = runtime['counters']()
            assert owner.output_score_precision == 'fp32_scores_bf16_pv'
            assert owner.output_layout == 'model_major'
            assert c['control'] == 'D_matched'
            assert c['output_precision_extra_qk_elements_upper_bound'] == 0
    assert seen == [(adapter, config['parent_config'], config['condition'])]


def test_controls_exclude_fresh_and_g75():
    config = v21.effective_control_config(base(), v20.ALL_NATIVE_LEGAL)
    for condition in ('v20_fresh_T', 'v20_G75L30_nativeQ128'):
        with pytest.raises(ValueError):
            v21.validate_effective(config, condition)
    parent = dict(config['parent_config'], fingerprint='0' * 64)
    with pytest.raises(ValueError, match='parent control fingerprint'):
        v21.validate_effective({**config, 'parent_config': parent}, config['condition'])
