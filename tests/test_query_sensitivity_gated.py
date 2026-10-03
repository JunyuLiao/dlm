import math

import pytest
import torch

from experiments.value_direction_hopper.query_adaptive import GATED_METHODS, State, weight
from experiments.value_direction_hopper.query_sensitivity_uniform import UniformThresholdState, uniform_policy


@pytest.mark.parametrize('method', GATED_METHODS + ('C_gate_soft',))
def test_gate_formula_and_missing_stability_protects(method):
    z = torch.zeros(1, 3)
    q = torch.full_like(z, .25)
    runs = torch.tensor([[0., 2., 100.]])
    margin, confidence = torch.full_like(z, 3.), torch.full_like(z, .75)
    u = {'M_gate': .25, 'C_gate': .5, 'T_gate': 0., 'M_gate_norm': 0.,
         'C_gate_soft': .5}[method]
    result = weight(method, margin, confidence, z, m_ref=1., trajectory=q,
                    stable_run=runs, gate_tau=2., margin_scale=.5)
    a = (1-torch.exp(-runs/2)) * (1-q)
    if method == 'C_gate_soft':
        a = .2 + .8*a
    expected = 1 + 3 * (1 - a * (1-u))
    assert torch.allclose(result, expected)
    if method == 'C_gate_soft':
        assert result[0, 0] < 4.
    else:
        assert result[0, 0] == 4.
    assert result[0, 1] > result[0, 2]


@pytest.mark.parametrize('method', GATED_METHODS + ('C_gate_soft',))
def test_gate_causal_reset_and_uniform_thresholds(method):
    class Router:
        query_sensitivity = None
        policy_selector = None
    router = Router()
    policy = uniform_policy(.1, -.2)
    state = UniformThresholdState(method, router, thresholds=policy,
        m_ref=1., gamma=.5, trajectory_gamma=.5, gate_tau=2.5,
        margin_scale=.4, diagnostics=False)
    canvas = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.tensor([[[5., 0.]] * 3])
    accept = torch.ones(1, 3, dtype=torch.bool)
    state.begin(48, canvas)
    assert torch.equal(router.query_sensitivity, torch.full((1, 3), 4.))
    for step in (48, 47, 46):
        state.observe_logits(logits, accept, step)
        state.begin(step-1, canvas)
    previous = router.query_sensitivity.clone()
    assert torch.all(previous < 4.)
    changed = logits.clone()
    changed[:, 1] = changed[:, 1].flip(-1)
    state.observe_logits(changed, torch.tensor([[False, True, True]]), 45)
    # Only the next routing call sees rejection and flip.
    assert torch.equal(router.query_sensitivity, previous)
    state.begin(44, canvas)
    if method == 'C_gate_soft':
        assert torch.all(router.query_sensitivity[0, :2] < 4.)
    else:
        assert torch.all(router.query_sensitivity[0, :2] == 4.)
    assert router.query_sensitivity[0, 2] < 4.
    assert state.margin_scale == .4
    for call in (1, 2, 3, 48):
        for kind in ('local', 'global'):
            assert router.policy_selector(call, kind) == policy[kind]
    state.begin(48, canvas)
    assert torch.all(router.query_sensitivity == 4.)


def test_normalized_margin_is_frozen_log_clip():
    margins = torch.tensor([[0., 1., 3., 10.]])
    z = torch.zeros_like(margins)
    result = weight('M_gate_norm', margins, z, z, m_ref=3.,
                    trajectory=z, stable_run=torch.full_like(z, 1000.),
                    margin_scale=2.)
    expected = 1 + 3 * ((math.log(4) - torch.log1p(margins))/2).clamp(0, 1)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_gate_parameter_rejected(value):
    with pytest.raises(ValueError):
        State('T_gate', None, m_ref=1., gate_tau=value)
    with pytest.raises(ValueError):
        State('M_gate_norm', None, m_ref=1., margin_scale=value)
    if value != 0.:
        with pytest.raises(ValueError):
            State('C_gate_soft', None, m_ref=1., soft_gate_lambda=value)
