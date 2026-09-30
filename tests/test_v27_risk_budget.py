"""v27 risk-budget selector on M1-DP: drop the lowest-risk prefix tiles while their summed risk stays in budget."""
import math

import pytest
import torch

pytest.importorskip('triton')
from experiments.numerical_qk_reuse.v27_dense_prefix import budget_skip  # noqa: E402


def _state(worst_by_tile, nq=100, width=128):
    """One batch, two heads, one query block; row 0 carries each tile's worst risk, other rows lower."""
    pt = len(worst_by_tile)
    lognorm = torch.full((1, 2, 1, pt, width), -5.0)
    for j, w in enumerate(worst_by_tile):
        lognorm[0, :, 0, j, 0] = w
    lognorm[..., nq:] = float('nan')          # rows past the canvas are never written by the build kernel
    eligible = torch.ones((1, 2, 1, pt), dtype=torch.int8)
    return lognorm, eligible, torch.ones((1, nq)), torch.ones((1, 1))


def test_budget_drops_the_ascending_prefix_that_fits():
    lognorm, eligible, t, ref = _state([math.log(0.4), math.log(0.1), math.log(0.3), math.log(0.2)])
    drop = budget_skip(lognorm, eligible, t, ref, 100, math.log(0.65))
    # ascending: 0.1, 0.2, 0.3 (sum 0.6 < 0.65) then 0.4 would exceed the budget
    assert drop[0, 0, 0].tolist() == [False, True, True, True]
    assert torch.equal(drop[0, 0], drop[0, 1])


def test_infinite_risk_is_never_dropped_and_inactive_tiles_always_are():
    lognorm, eligible, t, ref = _state([float('inf'), float('-inf'), math.log(0.2)])
    lognorm[0, :, 0, 1] = float('-inf')        # a tile with no active row: every row -inf
    drop = budget_skip(lognorm, eligible, t, ref, 100, 50.0)
    assert drop[0, 0, 0].tolist() == [False, True, True]


def test_ineligible_tiles_and_rows_past_the_canvas_are_ignored():
    lognorm, eligible, t, ref = _state([math.log(0.1), math.log(0.1)], nq=10)
    lognorm[0, :, 0, 0, 50] = 100.0            # a huge value in a row past nq must not count
    eligible[0, :, 0, 1] = 0
    drop = budget_skip(lognorm, eligible, t, ref, 10, math.log(0.5))
    assert drop[0, 0, 0].tolist() == [True, False]


def test_sensitivity_and_reference_scale_the_risk():
    lognorm, eligible, t, ref = _state([math.log(0.3), math.log(0.3)])
    assert budget_skip(lognorm, eligible, t, ref, 100, math.log(0.7))[0, 0, 0].tolist() == [True, True]
    t = torch.full((1, 100), 2.0)             # doubled sensitivity: 0.6 + 0.6 > 0.7
    assert budget_skip(lognorm, eligible, t, ref, 100, math.log(0.7))[0, 0, 0].sum().item() == 1
    ref = torch.full((1, 1), 4.0)             # and a 4x reference halves it again: 0.15 + 0.15
    assert budget_skip(lognorm, eligible, t, ref, 100, math.log(0.7))[0, 0, 0].tolist() == [True, True]
