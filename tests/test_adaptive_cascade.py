import math

import pytest
import torch

from experiments.diffusion_gemma_jl_output_aware import adaptive
from experiments.diffusion_gemma_jl_output_aware import reference
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections


def _states(seed=4, width=8, rank=8):
    torch.manual_seed(seed)
    scores = torch.randn(1, 1, 128, 192)
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[..., 0, :64] = False
    values = torch.randn(1, 1, 192, width)
    # Make R_8 exactly identity, while still exercising nested scaling in the
    # lower ranks through nontrivial prefixes.
    bank = torch.eye(width) * math.sqrt(rank)
    projected = values @ (bank / math.sqrt(rank))
    return scores, valid, values, projected


def test_nested_bank_is_reproducible_and_actually_nested():
    p = Projections()
    a = p.get_nested_bank(2, 3, 8, 64, 1729, 'cpu')
    b = Projections().get_nested_bank(2, 3, 8, 64, 1729, 'cpu')
    assert torch.equal(a, b)
    assert torch.equal(a[..., :8], p.get_nested_bank(2, 3, 8, 64, 1729, 'cpu')[..., :8])
    assert not torch.equal(a, Projections().get_nested_bank(2, 3, 8, 64, 2718, 'cpu'))
    # The scaling relation is explicit, not a second random draw.
    assert torch.equal(a[..., :2] / math.sqrt(2),
                       (a[..., :8] / math.sqrt(8))[..., :2] * math.sqrt(8/2))


def test_chi_square_budget_and_monotone_factors():
    factors = adaptive.interval_factors((2, 8, 32, 64), 128, 1e-3)
    assert set(factors) == {2, 8, 32, 64}
    assert all(0 < lo < 1 < hi for lo, hi in factors.values())
    # More dimensions should tighten the upper factor under the same budget.
    assert factors[64][1] < factors[2][1]


def test_adaptive_max_rank_matches_exact_when_projection_is_identity():
    scores, valid, values, projected = _states()
    p = reference.block_statistics(scores, valid, projected)
    exact = reference.block_statistics(scores, valid, values)
    # Rank-8 projection is exactly the original value space here.
    cfg = adaptive.CascadeConfig(stages=(8,), interval_delta=.01,
                                 exact_fallback=False)
    kv = valid.any(-2)
    ref = ((values.square().sum(-1) * kv).sum(-1) /
           kv.sum(-1).clamp_min(1)).sqrt()
    out = adaptive.route(p, ref, threshold=.2,
                         config=cfg, empirical_bands={8: (1., 1.)})
    expected = reference.route(exact, ref, Config(family='identity', log_threshold=math.log(.2)))
    assert torch.equal(out.skip, expected)
    assert out.diagnostics['exact_fallback_tiles'] == 0


def test_adaptive_unresolved_tiles_keep_without_oracle_and_charge_work():
    scores, valid, values, projected = _states(seed=9)
    state = reference.block_statistics(scores, valid, projected[..., :2])
    ref = torch.ones(1, 1)
    cfg = adaptive.CascadeConfig(stages=(2,), interval_delta=1e-6,
                                 exact_fallback=False)
    out = adaptive.route(state, ref, threshold=1e-9, config=cfg)
    assert not out.skip.any()  # first support plus uncertainty are retained
    assert out.diagnostics['projected_coordinate_evals'] > 0


def test_empirical_bands_and_decision_metrics():
    bands = adaptive.fit_empirical_bands({2: torch.linspace(.7, 1.3, 32),
                                          8: torch.linspace(.9, 1.1, 32)})
    assert bands[2][0] < bands[2][1]
    eligible = torch.ones(1, 2, 3, dtype=torch.bool)
    a = torch.tensor([[[True, False, False], [False, True, False]]])
    b = torch.tensor([[[False, False, False], [False, True, True]]])
    m = adaptive.decision_metrics(a, b, eligible)
    assert m['false_skips'] == 1 and m['false_keeps'] == 1
    assert m['disagreement'] == 2


def test_deterministic_bound_never_screens_first_support():
    alpha = torch.full((1, 1, 4), .1)
    block = torch.full_like(alpha, .01)
    retained = torch.full_like(alpha, .01)
    ref = torch.ones(1, 1)
    active = torch.ones_like(alpha, dtype=torch.bool)
    supported = torch.zeros_like(active)
    out = adaptive.deterministic_upper_bound(alpha, block, retained, ref, .5,
                                              active, supported)
    assert not out.any()
