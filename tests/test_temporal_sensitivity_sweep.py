"""CPU-only invariants for the frozen temporal-sensitivity sweep."""
import pytest
import torch

from experiments.value_direction_hopper import temporal_sensitivity_sweep as sweep
from experiments.value_direction_hopper import temporal_sensitivity_sweep_report as summary_report
from experiments.value_direction_hopper.query_adaptive import weight


def test_grid_is_unique_and_beta_zero_has_one_gamma():
    assert len(sweep.GRID) == 10
    assert len(set(sweep.GRID)) == len(sweep.GRID)
    assert [gamma for beta, gamma in sweep.GRID if beta == 0] == [0]
    assert len(sweep.combinations()) == 20
    assert sweep.label(3., .5, 70) == 'b3_g0p5_s70'


@pytest.mark.parametrize('target', [50, 70])
def test_candidates_keep_identical_early_pair(target):
    original = sweep._reference_policy(target)['policy']
    policies = sweep.candidate_policies(target)
    assert len(policies) == len(sweep.SHIFTS)
    assert all(policy['early'] == original['early'] for policy in policies)
    assert any(policy['late'] == original['late'] for policy in policies)


def test_beta_zero_is_unweighted_and_more_beta_protects_fixed_rows():
    margin = torch.ones((1, 3))
    confidence = torch.full((1, 3), .8)
    temporal = torch.tensor([[0., .25, 1.]])
    zero = weight('T', margin, confidence, temporal, beta=0.)
    small = weight('T', margin, confidence, temporal, beta=1.)
    large = weight('T', margin, confidence, temporal, beta=6.)
    assert torch.equal(zero, torch.ones_like(zero))
    assert torch.all(large >= small)
    assert large[0, 0] == small[0, 0] == 1


def test_rank_prefers_feasibility_then_max_stratum_error():
    goal = {'whole': .7, 'global': .7, 'local': .7}
    def point(rates, steps, violations):
        return dict(metrics=dict(overall=dict(zip(goal, rates)),
            mean_steps=steps, executed_tiles=100), violations=violations)
    close = point((.7, .71, .69), 7., [])
    faster_but_far = point((.7, .72, .68), 4., [])
    infeasible = point((.7, .7, .7), 3., ['mean_steps'])
    assert sweep.rank_point(close, goal) < sweep.rank_point(faster_but_far, goal)
    assert sweep.rank_point(close, goal) < sweep.rank_point(infeasible, goal)


def test_condition_fingerprint_changes_with_parameters():
    base = {'fingerprint': 'frozen'}
    spec = {'fingerprint': 'sweep'}
    a = sweep.config_for(base, spec, 1., .5, 70)
    b = sweep.config_for(base, spec, 3., .5, 70)
    c = sweep.config_for(base, spec, 1., .9, 70)
    assert len({a['fingerprint'], b['fingerprint'], c['fingerprint']}) == 3
    with pytest.raises(ValueError):
        sweep.config_for(base, spec, 0., .9, 70)


def test_report_plots_show_measured_sparsity_and_gate_miss(tmp_path):
    rows = [dict(beta=0., gamma=0., target=50, mean_calls=5., accuracy=.8,
                 actual_overall=.49, calibration_status='unattainable_under_guardrails'),
            dict(beta=3., gamma=.5, target=70, mean_calls=7., accuracy=.9,
                 actual_overall=.69, calibration_status='attained')]
    summary_report._plot(tmp_path, rows, dict(mean_calls=4.))
    assert (tmp_path/'plots/calls_beta_gamma_s50.png').is_file()
    assert (tmp_path/'plots/accuracy_beta_gamma_s70.png').is_file()
    assert (tmp_path/'plots/calls_vs_beta_s70.png').is_file()
