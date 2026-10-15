import torch

from experiments.numerical_qk_reuse.v31_cached_refresh import (
    CachedRefreshState, advance_ages, bounded_swaps, drift_metric,
    filled_intersection, full_information_omission_error, hybrid_log_mass,
    hybrid_qblock_score, log_decay, map_iou, strict_intersection, topk_fixed,
)


def test_hybrid_replaces_only_retained_and_decays_outside():
    dense = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    fresh = torch.tensor([[[9.0, 9.0], [3.0, 3.0], [4.0, 4.0]]])
    current = torch.tensor([[False, True, False]])
    age = torch.tensor([[4, 0, 1]], dtype=torch.int16)
    score = hybrid_qblock_score(dense, fresh, current, current, age,
                                decay_lambda=0.5, decay_floor=0.25)
    # The retained tile takes fresh data; only the two outside tiles are
    # penalized, while the retained tile is unaffected by decay.
    assert score.shape == (1, 3)
    assert score[0, 1] > score[0, 2]
    assert score[0, 0] < score[0, 2]


def test_global_topk_and_bounded_swaps_have_fixed_budget():
    score = torch.tensor([[0.1, 0.9, 0.8, 0.7, 0.6]])
    current = torch.tensor([[True, True, False, False, False]])
    selected = topk_fixed(score, 2)
    assert selected.sum().item() == 2
    swapped = bounded_swaps(score, current, 2, frontier_m=4, swap_budget=1)
    assert swapped.sum().item() == 2
    assert ((current & ~swapped).sum().item()) <= 1
    assert map_iou(current, swapped).item() >= 0.0


def test_decay_age_resets_selected_and_increments_unselected():
    old = torch.tensor([[0, 2, 32767]], dtype=torch.int16)
    selected = torch.tensor([[True, False, False]])
    age = advance_ages(old, selected)
    assert age.tolist() == [[0, 3, 32767]]
    assert torch.allclose(log_decay(age, 0.0), torch.zeros_like(age, dtype=torch.float32))


def test_hybrid_replacement_normalizes_complete_support():
    cached = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    fresh = torch.tensor([[[10.0, 10.0], [3.0, 3.0], [4.0, 4.0]]])
    retained = torch.tensor([[False, True, False]])
    hybrid = hybrid_log_mass(cached, fresh, retained)
    assert torch.equal(hybrid, torch.tensor([[[0.0, 0.0], [3.0, 3.0], [2.0, 2.0]]]))
    score = hybrid_qblock_score(cached, fresh, retained)
    expected = (hybrid - torch.logsumexp(hybrid, dim=1, keepdim=True)).amax(-1)
    assert torch.allclose(score, expected)


def test_total_frontier_and_stable_physical_ties():
    score = torch.tensor([[1.0, 1.0, 1.0, 0.9, 0.8]])
    current = torch.tensor([[True, False, True, False, False]])
    selected = bounded_swaps(score, current, k=2, frontier_m=4, swap_budget=8)
    # M is the total frontier.  Equal-mass candidates do not churn a held map.
    assert selected.tolist() == [[True, False, True, False, False]]
    assert int((current & ~selected).sum()) <= 8
    assert topk_fixed(score, 2).tolist() == [[True, True, False, False, False]]


def test_bounded_swap_cap_and_mass_first_value_tie_break():
    score = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
    current = torch.tensor([[True, True, False, False, False]])
    value = torch.tensor([[0.0, 0.0, 9.0, 8.0, 0.0]])
    selected = bounded_swaps(score, current, 2, 5, 1, value, mass_tolerance=0.2)
    assert int((current & ~selected).sum()) == 1
    assert selected[0, 2]


def test_state_age_stale_selection_and_noop():
    raw = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    selected = torch.tensor([[False, True, True, False]])
    state = CachedRefreshState.from_dense(raw, selected)
    result = state.refresh(raw.clone(), policy="bounded_swaps", frontier_m=4, swap_budget=0)
    assert result.no_op
    assert state.age.tolist() == [[1, 0, 0, 1]]
    assert state.receipt()["no_op_count"] == 1


def test_nonfinite_or_drift_forces_dense_without_map_change():
    raw = torch.zeros((1, 3, 2))
    selected = torch.tensor([[True, True, False]])
    state = CachedRefreshState.from_dense(raw, selected)
    fresh = raw.clone()
    fresh[0, 0, 0] = float("nan")
    result = state.refresh(fresh, policy="global_topk", drift_threshold=0.1)
    assert result.fallback and result.nonfinite
    assert torch.equal(state.selected, selected)
    assert state.receipt()["forced_dense_fallback_count"] == 1
    finite_drift, bad = drift_metric(raw + 1.0, raw, selected)
    assert not bad and torch.isfinite(finite_drift)
    structural = torch.full((1, 2, 2), float('-inf'))
    structural_drift, structural_bad = drift_metric(structural, structural, torch.ones((1, 2), dtype=torch.bool))
    assert structural_bad is False and structural_drift.item() == 0.0


def test_strict_and_filled_intersection_are_distinct():
    control = torch.tensor([[True, True, False, False]])
    other = torch.tensor([[False, True, True, False]])
    strict = strict_intersection(control, other)
    assert strict.tolist() == [[False, True, False, False]]
    filled = filled_intersection(control, other, torch.tensor([[0.1, 0.2, 0.9, 0.8]]), 2)
    assert filled.sum().item() == 2
    assert filled[0, 1] and filled[0, 2]


def test_full_information_omission_error_matches_simple_oracle():
    raw = torch.log(torch.tensor([[[0.5], [0.5]]]))
    values = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    dense = torch.tensor([[[0.5, 0.5]]])
    err = full_information_omission_error(raw, values, dense)
    assert err.shape == (1, 2, 1)
    assert torch.allclose(err[0, :, 0], torch.tensor([0.70710677, 0.70710677]), atol=1e-5)
