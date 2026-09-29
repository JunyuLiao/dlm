"""v27: effective A/R clock, R6 and hold-only B over a whole 0..47 canvas."""
import pytest

from experiments.numerical_qk_reuse.cache import Identity, ScoreCache


def _identity(canvas=0, epoch=0, keys=1000):
    return Identity(0, canvas, epoch, 5, 1, 16, 8, 256, keys, 256, 0, 0, keys,
                    1.0, 'torch.bfloat16', 'cuda:0', (True, None, None), 1)


def phases(score_period, decision_interval, hold_only=False, calls=48, stop=None):
    """Bootstrap origin-1 clock: call 0 native, call 1 observation, then the cache."""
    cache = ScoreCache(score_period, decision_interval, origin=1)
    cache.hold_only = hold_only
    ident, out = _identity(), ['B0', 'BO']
    cache.publish_scores(ident, 1, object())
    cache.publish_decision(ident, 1, object())
    for step in range(2, calls if stop is None else stop):
        plan = cache.plan(ident, step)
        if plan.score_refresh:
            cache.publish_scores(ident, step, object())
            out.append('A')
        elif plan.decision_refresh:
            out.append('D')
        else:
            out.append('H')
        if plan.decision_refresh:
            cache.publish_decision(ident, step, object())
    return out


def idx(seq, phase):
    return [i for i, p in enumerate(seq) if p == phase]


@pytest.mark.parametrize('a', [8, 16, 64])
def test_anchor_positions(a):
    seq = phases(a, 3)
    assert idx(seq, 'A') == [i for i in range(2, 48) if (i - 1) % a == 0]


def test_r1_is_m1_every_call_redecides():
    seq = phases(8, 1)
    assert 'H' not in seq and idx(seq, 'A') == [9, 17, 25, 33, 41]


@pytest.mark.parametrize('a,r', [(8, 3), (8, 6), (16, 3), (16, 6), (64, 3), (64, 6)])
def test_redecision_age_resets_at_anchor(a, r):
    seq = phases(a, r)
    last = 1
    for i in range(2, 48):
        if seq[i] == 'A':
            last = i
        elif seq[i] == 'D':
            assert i - last == r, (a, r, i, last)
            last = i
        else:
            assert i - last < r


def test_r6_a8_positions():
    seq = phases(8, 6)
    assert idx(seq, 'D')[:4] == [7, 15, 23, 31]


def test_b16_hold_only_has_no_intermediate_decision():
    seq = phases(16, 8, hold_only=True)
    assert idx(seq, 'D') == [] and idx(seq, 'A') == [17, 33]
    # The unfixed parent clock (R8 kept with A16) would redecide at call 9.
    assert idx(phases(16, 8), 'D')[0] == 9


def test_b8_hold_only_matches_old_b8_clock():
    assert phases(8, 8, hold_only=True) == phases(8, 8)


def test_a64_bootstrap_has_no_anchor_inside_max48():
    assert idx(phases(64, 3), 'A') == [] and idx(phases(64, 3, hold_only=True), 'D') == []


def test_early_stop_prefix_consistent():
    full = phases(16, 6)
    for stop in (3, 10, 20):
        assert phases(16, 6, stop=stop) == full[:stop]


def test_identity_change_forces_refresh_even_hold_only():
    cache = ScoreCache(16, 8, origin=1)
    cache.hold_only = True
    ident = _identity()
    cache.publish_scores(ident, 1, object())
    cache.publish_decision(ident, 1, object())
    assert not cache.plan(ident, 5).decision_refresh
    for changed in (_identity(canvas=1), _identity(epoch=1), _identity(keys=1064)):
        plan = cache.plan(changed, 5)
        assert plan.score_refresh and plan.decision_refresh
    assert cache.plan(ident, 5, force_refresh=True).decision_refresh
