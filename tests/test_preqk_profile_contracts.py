"""Pin the v7 profiling-harness defects that v8 identified, so they cannot
silently return.

These are CPU contract tests on the harness helpers, not GPU measurements.
"""
import numpy as np
import pytest
import torch

from scripts.preqk_full_forward_profile import (clone_kwargs, describe_sensitivity,
                                                tensor_digest)


def test_uniform_t_is_reported_as_uniform_even_when_present():
    """Bug B: 'a sensitivity tensor exists' is not 'the T path was profiled'."""
    assert describe_sensitivity(None)['uniform'] is True
    assert 'NOT the production T path' in describe_sensitivity(None)['note']
    ones = torch.ones(1, 8)
    described = describe_sensitivity(ones)
    assert described['present'] and described['uniform']
    assert 'uniform despite being present' in described['note']
    varied = torch.linspace(1., 4., 8).reshape(1, 8)
    described = describe_sensitivity(varied)
    assert described['present'] and not described['uniform']
    assert described['min'] == pytest.approx(1.) and described['max'] == pytest.approx(4.)
    assert described['spread'] == pytest.approx(3.)


def test_tensor_digest_is_order_sensitive():
    """Bug D: a digest that ignores order cannot prove replay inputs match."""
    left = torch.tensor([1., 2., 3.])
    right = torch.tensor([3., 2., 1.])
    assert tensor_digest(left) != tensor_digest(right)
    assert tensor_digest(left) == tensor_digest(left.clone())


def test_clone_kwargs_does_not_protect_non_tensor_state():
    """Bug D: the harness must not pretend cloning isolates sampler state."""
    class Sampler:
        def __init__(self):
            self.calls = 0

    sampler = Sampler()
    kwargs = dict(current_canvas=torch.zeros(4), sampler=sampler, cur_step=7)
    cloned = clone_kwargs(kwargs)
    cloned['current_canvas'] += 1
    assert torch.equal(kwargs['current_canvas'], torch.zeros(4))   # tensor isolated
    assert cloned['sampler'] is sampler                            # object shared
    cloned['sampler'].calls += 1
    assert sampler.calls == 1                                      # mutation leaks


def test_reference_prefix_summary_fixture_still_holds():
    """The v8 CPU illustration must keep passing in-repo (5 cases)."""
    import unittest

    from experiments.numerical_qk_reuse import prefix_summary_reference as reference
    suite = unittest.TestLoader().loadTestsFromModule(reference)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.testsRun == 5 and result.wasSuccessful()


def test_reference_reuse_saves_exactly_the_full_prefix_blocks():
    """The fixture's accounting: reuse must skip precisely the cached blocks."""
    from experiments.numerical_qk_reuse.prefix_summary_reference import (build_prefix_cache,
                                                                         route)
    rng = np.random.default_rng(5)
    prefix, queries, rank = 128, 7, 8
    keys = prefix + 70
    scores = rng.normal(size=(queries, keys))
    scores[rng.random((queries, keys)) > .9] = -np.inf
    projected = rng.normal(size=(keys, rank))
    epoch = ('req', 1, 3, 'proj')
    cache = build_prefix_cache(scores, projected, prefix, 64, epoch)
    assert len(cache.values) == prefix // 64
    sensitivity = 1. + rng.random(queries) * 3.
    base = route(scores, projected, sensitivity, .9, -.5, prefix=prefix, epoch=epoch)
    optimized = route(scores, projected, sensitivity, .9, -.5, prefix=prefix,
                      epoch=epoch, cache=cache)
    np.testing.assert_array_equal(base[0], optimized[0])
    assert base[3] - optimized[3] == len(cache.values)
