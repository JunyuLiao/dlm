import torch
import pytest

from experiments.value_direction_hopper.query_adaptive import State, weight, percentile_rank
from experiments.value_direction_hopper.query_sensitivity_uniform import (
    UniformThresholdState, uniform_policy,
)


def test_t_prior_is_bounded_and_protects_missing_history():
    temporal = torch.tensor([[0., .5, 1.]])
    trajectory = torch.tensor([[1., .5, 0.]])
    values = weight('T_prior', None, None, temporal, beta=3.,
                    trajectory=trajectory)
    assert values.tolist() == [[4., 3.25, 4.]]
    assert bool(((values >= 1.) & (values <= 4.)).all())


def test_hybrid_and_run_length_variants_are_bounded():
    temporal = torch.tensor([[0., 0.]])
    trajectory = torch.tensor([[0., .25]])
    drift = torch.tensor([[1., 0.]])
    hybrid = weight('T_hybrid', None, None, temporal, beta=3.,
                    trajectory=trajectory, drift=drift)
    run = weight('T_run', None, None, temporal, beta=3.,
                 run_uncertainty=torch.tensor([[.25, .5]]))
    assert hybrid.tolist() == [[4., 1.75]]
    assert torch.all((run >= 1.) & (run <= 4.))


def test_confidence_tail_keeps_absolute_uncertainty_and_lifts_only_tail():
    confidence = torch.tensor([[0., .25, .75, 1.]])
    temporal = torch.zeros_like(confidence)
    trajectory = torch.full_like(confidence, .5)
    result = weight('C_tail', None, confidence, temporal, beta=3.,
                    trajectory=trajectory, tail_tau=.5, tail_lambda=.5)
    # u_C is [1, sqrt(.75), .5, 0].  With ascending percentile ranks
    # [1, 2/3, 1/3, 0], only the first two positions receive a lift.
    base = torch.tensor([[1., .75**.5, .5, 0.]])
    tail = torch.tensor([[1., 1/3, 0., 0.]])
    expected = 1 + 3 * (base + .5*.5*tail*(1-base))
    assert torch.allclose(result, expected, atol=1e-6)
    assert result[0, 0] > result[0, 1] > result[0, 2] > result[0, 3]


def test_confidence_tail_q_zero_is_base_and_first_call_is_protected():
    confidence = torch.tensor([[.25, .75, .95]])
    temporal = torch.zeros_like(confidence)
    base = (1-confidence).sqrt()
    result = weight('C_tail', None, confidence, temporal, beta=3.,
                    trajectory=torch.zeros_like(confidence), tail_lambda=.5)
    assert torch.allclose(result, 1 + 3*base)
    class Router:
        query_sensitivity = None
        policy_selector = None
    router = Router()
    state = State('C_tail', router, m_ref=1., beta=3., gamma=.5,
                  diagnostics=False, seed=7)
    canvas = torch.zeros((1, 3), dtype=torch.long)
    state.begin(48, canvas)
    assert torch.equal(router.query_sensitivity, torch.full((1, 3), 4.))


@pytest.mark.parametrize('method', ['C_tail'])
def test_confidence_tail_parameters_are_validated(method):
    temporal = torch.zeros((1, 2))
    confidence = torch.full_like(temporal, .5)
    with pytest.raises(ValueError):
        weight(method, None, confidence, temporal, tail_tau=1.)
    with pytest.raises(ValueError):
        weight(method, None, confidence, temporal, tail_lambda=1.1)


def test_confidence_tail_ties_batch_and_permutation():
    values = torch.tensor([[.1, .4, .4, .8, .8], [.9, .9, .9, .9, .9]])
    expected = torch.tensor([[0., .375, .375, .875, .875], [.5]*5])
    assert torch.equal(percentile_rank(values), expected)
    confidence = 1-values.square()
    q = torch.ones_like(values)
    z = torch.zeros_like(values)
    s = weight('C_tail', None, confidence, z, trajectory=q)
    perm = torch.tensor([4, 0, 3, 2, 1])
    changed = weight('C_tail', None, confidence[:, perm], z, trajectory=q)
    assert torch.equal(changed, s[:, perm])
    assert s[0, 3] == s[0, 4]
    assert torch.equal(percentile_rank(values[:, :1]), torch.zeros(2, 1))


def test_confidence_tail_completed_history_uniform_thresholds_and_reset():
    class Router:
        query_sensitivity = None
        policy_selector = None
    router = Router()
    policy = uniform_policy(.1, -.2)
    state = UniformThresholdState('C_tail', router, m_ref=1.,
        trajectory_gamma=.65, thresholds=policy, diagnostics=False)
    canvas = torch.zeros((2, 5), dtype=torch.long)
    logits = torch.zeros((2, 5, 2))
    logits[..., 0] = torch.arange(5.)
    accepted = torch.ones_like(canvas, dtype=torch.bool)
    state.begin(48, canvas)
    first = router.query_sensitivity.clone()
    state.observe_logits(logits, accepted, 48)
    assert torch.equal(router.query_sensitivity, first)
    assert torch.allclose(state.trajectory, torch.full_like(state.trajectory, .65))
    state.begin(47, canvas)
    second = router.query_sensitivity.clone()
    assert torch.all(second < 4.)
    state.observe_logits(logits.flip(-1), ~accepted, 47)
    assert torch.equal(router.query_sensitivity, second)
    state.begin(46, canvas)
    # Rejections and flips cannot saturate every position.
    assert torch.all(router.query_sensitivity < 4.)
    for call in (1, 2, 3, 48):
        for kind in ('local', 'global'):
            assert router.policy_selector(call, kind) == policy[kind]
    state.begin(48, canvas)
    assert torch.equal(router.query_sensitivity, first)


def test_margin_confidence_and_smooth_temporal_priors_match_causal_formulas():
    margin = torch.tensor([[0., 1.]])
    confidence = torch.tensor([[0., .75]])
    temporal = torch.zeros((1, 2))
    trajectory = torch.full((1, 2), .5)
    margin_prior = weight('M_prior', margin, confidence, temporal, beta=3.,
                          m_ref=1., trajectory=trajectory)
    confidence_prior = weight('C_prior', margin, confidence, temporal, beta=3.,
                              trajectory=trajectory)
    smooth_temporal = weight('T_smooth', margin, confidence, temporal, beta=3.,
                             trajectory=trajectory)
    # q=.5; u_M=[1,.5], u_C=[1,.5], z=0.
    expected = torch.tensor([[4., 3.25]])
    assert torch.allclose(margin_prior, expected)
    assert torch.allclose(confidence_prior, expected)
    assert torch.allclose(smooth_temporal, torch.full((1, 2), 2.5))


def test_causal_variants_require_their_previous_step_signal():
    temporal = torch.zeros((1, 2))
    with pytest.raises(ValueError, match='margin history'):
        weight('M_prior', None, None, temporal)
    with pytest.raises(ValueError, match='confidence history'):
        weight('C_prior', torch.ones((1, 2)), None, temporal)
    with pytest.raises(ValueError, match='temporal history'):
        weight('T_smooth', torch.ones((1, 2)), torch.ones((1, 2)), None)


def test_t_prior_relaxes_stable_accepted_queries_but_keeps_flips_high():
    stable = weight('T_prior', None, None, torch.zeros((1, 2)), beta=3.,
                    trajectory=torch.tensor([[.125, .125]]))
    flipped = weight('T_prior', None, None, torch.tensor([[1., 0.]]), beta=3.,
                    trajectory=torch.tensor([[.125, .125]]))
    assert stable[0, 0] == stable[0, 1] == 1.375
    assert flipped[0, 0] == 4.
    assert flipped[0, 1] == stable[0, 1]


def test_t_prior_state_starts_protected_and_uses_only_previous_acceptance():
    class Router:
        query_sensitivity = None
        policy_selector = None

    router = Router()
    state = State('T_prior', router, m_ref=1., beta=3., gamma=.5,
                  diagnostics=False, seed=7)
    canvas = torch.zeros((1, 4), dtype=torch.long)
    state.begin(48, canvas)
    assert torch.equal(router.query_sensitivity, torch.full((1, 4), 4.))

    logits = torch.zeros((1, 4, 3))
    logits[..., 0] = 3.
    accepted = torch.tensor([[True, False, True, False]])
    state.observe_logits(logits, accepted, 48)
    state.begin(47, canvas)
    # q = .5 * 1 + .5 * {0,1,0,1}; coefficient = 1 + 3 q.
    assert torch.allclose(router.query_sensitivity,
                          torch.tensor([[2.5, 4., 2.5, 4.]]))

    state.observe_logits(logits, torch.ones_like(accepted), 47)
    state.begin(46, canvas)
    # One accepted step halves q, while no flip history keeps the rows stable.
    assert torch.allclose(router.query_sensitivity,
                          torch.tensor([[1.75, 2.5, 1.75, 2.5]]))


def test_t_prior_canvas_reset_restores_protected_prior():
    class Router:
        query_sensitivity = None
        policy_selector = None

    router = Router()
    state = State('T_prior', router, m_ref=1., beta=3., gamma=.5,
                  diagnostics=False, seed=7)
    canvas = torch.zeros((1, 2), dtype=torch.long)
    state.begin(48, canvas)
    state.observe_logits(torch.tensor([[[3., 0.], [3., 0.]]]),
                         torch.ones((1, 2), dtype=torch.bool), 48)
    state.begin(47, canvas)
    assert torch.all(router.query_sensitivity < 4.)
    state.begin(48, canvas)
    assert torch.equal(router.query_sensitivity, torch.full((1, 2), 4.))


def test_all_causal_variants_start_at_maximum_sensitivity():
    class Router:
        query_sensitivity = None
        policy_selector = None

    canvas = torch.zeros((1, 3), dtype=torch.long)
    for method in ('M_prior', 'C_prior', 'T_prior', 'T_smooth', 'T_hybrid', 'T_run'):
        router = Router()
        state = State(method, router, m_ref=1., beta=3., gamma=.5,
                      diagnostics=False, seed=7)
        state.begin(48, canvas)
        assert torch.equal(router.query_sensitivity, torch.full((1, 3), 4.)), method


def test_smooth_prior_uses_a_slower_trajectory_ema():
    class Router:
        query_sensitivity = None
        policy_selector = None

    canvas = torch.zeros((1, 1), dtype=torch.long)
    logits = torch.tensor([[[3., 0.]]])
    accepted = torch.ones((1, 1), dtype=torch.bool)
    fast_router, slow_router = Router(), Router()
    fast = State('T_prior', fast_router, m_ref=1., beta=3., gamma=.5,
                 trajectory_gamma=.5, diagnostics=False, seed=7)
    slow = State('T_smooth', slow_router, m_ref=1., beta=3., gamma=.5,
                 trajectory_gamma=.8, diagnostics=False, seed=7)
    for state in (fast, slow):
        state.begin(48, canvas)
        state.observe_logits(logits, accepted, 48)
        state.begin(47, canvas)
    assert slow_router.query_sensitivity.item() > fast_router.query_sensitivity.item()


def test_hybrid_state_uses_previous_confidence_drift_only():
    class Router:
        query_sensitivity = None
        policy_selector = None

    router = Router()
    state = State('T_hybrid', router, m_ref=1., beta=3., gamma=.5,
                  diagnostics=False, seed=7)
    canvas = torch.zeros((1, 2), dtype=torch.long)
    state.begin(48, canvas)
    logits_a = torch.tensor([[[3., 0.], [3., 0.]]])
    state.observe_logits(logits_a, torch.ones((1, 2), dtype=torch.bool), 48)
    state.begin(47, canvas)
    first = router.query_sensitivity.clone()
    # A changed confidence distribution is observed after call two and only
    # affects the coefficient on call three.
    logits_b = torch.tensor([[[0.5, 0.], [3., 0.]]])
    state.observe_logits(logits_b, torch.ones((1, 2), dtype=torch.bool), 47)
    state.begin(46, canvas)
    assert router.query_sensitivity[0, 0] > 1.75
    assert router.query_sensitivity[0, 1] == 1.75


def test_uniform_state_reuses_one_threshold_pair_on_every_call():
    class Router:
        query_sensitivity = None
        policy_selector = None

    router = Router()
    policy = uniform_policy(-1.25, -0.75)
    state = UniformThresholdState('T_prior', router, m_ref=1., beta=3.,
                                  gamma=.5, diagnostics=False, seed=7,
                                  thresholds=policy)
    canvas = torch.zeros((1, 2), dtype=torch.long)
    state.begin(48, canvas)
    assert router.policy_selector(1, 'local') == policy['local']
    assert router.policy_selector(1, 'global') == policy['global']
    state.observe_logits(torch.tensor([[[3., 0.], [3., 0.]]]),
                         torch.ones((1, 2), dtype=torch.bool), 48)
    state.begin(47, canvas)
    assert router.policy_selector(2, 'local') == policy['local']
    assert router.policy_selector(2, 'global') == policy['global']
