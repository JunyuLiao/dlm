from experiments.value_direction_hopper.query_adaptive_guardrail_refine import candidates


def test_refinement_grid_preserves_shared_archived_early_phase():
    points=candidates(70)
    assert len(points)>=9
    early=points[0]['early']
    assert all(p['early']==early for p in points)
    assert len({p['late']['global']['log_threshold'] for p in points})>=4
    assert any(p['late']['global']['log_threshold']>early['global']['log_threshold']
               and p['late']['local']['log_threshold']>early['local']['log_threshold']
               for p in points)
