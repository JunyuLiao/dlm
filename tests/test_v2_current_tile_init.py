import math

import torch

from experiments.numerical_qk_reuse.v2_current_tile_init import current_tile_reference
from experiments.numerical_qk_reuse.v31_value_selectors import Stats, online_reference


def fixture(means, mass=None):
    j = len(means)
    lm = torch.tensor(mass or [0.] * j).view(1, j, 1)
    mu = torch.tensor(means).view(1, j, 1, -1)
    s = Stats(lm, mu, torch.ones(1, 1), torch.zeros_like(lm, dtype=torch.bool), j, 1, 1)
    mandatory = torch.zeros(1, j, dtype=torch.bool)
    mandatory[:, -1] = True
    return s, mandatory


def test_equal_values_do_not_force_first_tile():
    s, m = fixture([[1., 0.]] * 4)
    keep, state, lm, _ = current_tile_reference(s, .1, m)
    assert keep.tolist() == [[False, False, False, True]]
    assert torch.equal(state, torch.tensor([[[1., 0.]]]))
    assert torch.allclose(lm, torch.tensor([[math.log(4.)]]))
    assert online_reference(s, .1, True, m)[0].tolist() == [[True, False, False, True]]


def test_current_is_processed_once_and_remaining_scan_starts_at_zero():
    s, m = fixture([[3.], [1.], [0.]])
    keep, state, lm, _ = current_tile_reference(s, .4, m)
    assert keep.tolist() == [[True, False, True]]
    assert torch.allclose(state, torch.tensor([[[1.5]]]))
    assert torch.allclose(lm, torch.tensor([[math.log(3.)]]))


def test_zero_threshold_is_dense_and_matches_exact_weighted_output():
    s, m = fixture([[1., 2.], [4., -2.], [-1., 5.]], [1., -1., .5])
    keep, state, lm, _ = current_tile_reference(s, 0., m)
    assert keep.all()
    expected = (s.log_mass.softmax(1)[..., None] * s.mean).sum(1)
    assert torch.allclose(state, expected)
    assert torch.allclose(lm, torch.logsumexp(s.log_mass, 1))


def test_one_tile_and_invalid_fail_closed():
    s, m = fixture([[2., -1.]])
    assert current_tile_reference(s, 100., m)[0].all()
    s, m = fixture([[1.], [2.], [3.]])
    s.invalid[:, 1] = True
    assert current_tile_reference(s, 100., m)[0].all()
