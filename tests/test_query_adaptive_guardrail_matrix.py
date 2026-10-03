from experiments.value_direction_hopper.query_adaptive_guardrail import pair, phase_policy
from experiments.value_direction_hopper.query_adaptive_guardrail_matrix import (
    candidate_policies, refine_policies,
)


def test_method_search_keeps_frozen_early_pair():
    base=phase_policy(pair(-.4,-1.4),pair(.2,-.8))
    for method in ('unweighted','C','M','T_uniform','T_tile_mean','T_shuffle'):
        points=candidate_policies(method,base)
        assert len(points)==6
        assert all(point['early']==base['early'] for point in points)
        assert len({point['late']['local']['log_threshold'] for point in points})==6


def test_refinement_changes_late_pair_only():
    base=phase_policy(pair(-.4,-1.4),pair(.2,-.8))
    point=dict(policy=base)
    refined=refine_policies(point,base['early'])
    assert len(refined)==6
    assert all(policy['early']==base['early'] for policy in refined)
    assert any(policy['late']['global']==base['late']['global']
               and policy['late']['local']!=base['late']['local']
               for policy in refined)
