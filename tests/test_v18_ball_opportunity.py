"""CPU properties for the explicitly ideal-arithmetic ball diagnostic."""
import numpy as np
import pytest
import torch

from scripts.v18_ball_opportunity import ideal_ball_bound, packed_valid
from scripts.v18_scope_capture import check_capture_config


def test_ball_bound_dominates_convex_block_mean_and_handles_cancellation():
    z = np.array([[10., 0.], [-10., 0.], [0., 2.]], dtype=np.float64)
    old = np.array([[1., 2.], [-2., 1.]], dtype=np.float64)
    prev_lse = np.array([0., -1.])
    block_lse = np.array([.5, 0.])
    sensitivity = np.array([1., 2.])
    reference = 3.
    bound = ideal_ball_bound(z, old, prev_lse, block_lse, sensitivity, reference)
    for weights in (np.array([.5, .5, 0.]), np.array([1., 0., 0.]), np.array([0., .2, .8])):
        mu = weights @ z
        alpha = np.exp(block_lse - np.logaddexp(prev_lse, block_lse))
        actual = sensitivity * alpha * np.linalg.norm(mu - old, axis=1) / reference
        assert np.all(actual <= bound + 1e-12)
    assert bound[0] > 0  # mean-only would hide the +/-10 outlier pair


def test_inactive_row_has_zero_bound():
    z = np.array([[1., 0.], [2., 0.]])
    value = ideal_ball_bound(z, np.zeros((2, 2)), np.array([0., 0.]),
                             np.array([-np.inf, 0.]), np.ones(2), 1.)
    assert value[0] == 0. and value[1] > 0.


def test_packed_valid_preserves_high_bit_and_partial_edge():
    words = torch.tensor([[[[1 << 63, 0b101]]]], dtype=torch.uint64)
    valid = packed_valid(words, 67)
    assert valid.shape == (1, 1, 1, 67)
    assert valid[0, 0, 0, 63] and valid[0, 0, 0, 64] and valid[0, 0, 0, 66]
    assert not valid[0, 0, 0, 65]


def test_capture_arm_must_match_frozen_t_target():
    config = dict(condition='native_legal_all_layers', frontier_arm='T60', method='T', target=60)
    check_capture_config(config, 'T60')
    with pytest.raises(ValueError, match='explicitly selected'):
        check_capture_config(config, 'T50')
    with pytest.raises(ValueError, match='explicitly selected'):
        check_capture_config(dict(config, target=50), 'T60')
