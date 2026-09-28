"""CPU checks for the v23 origin-1 score clock (no GPU, no model)."""
import pytest

from experiments.numerical_qk_reuse.cache import Identity, ScoreCache


def _identity():
    return Identity(0, 0, 0, 5, 1, 16, 2, 256, 1024, 512, 1000, 0, 1024, .1,
                    'torch.bfloat16', 'cuda:0', (False, None, None), 1)


def _schedule(interval, origin, first, last=18):
    cache, ident, phases = ScoreCache(8, interval, origin=origin), _identity(), {}
    for step in range(first, last):
        plan = cache.plan(ident, step)
        phases[step] = 'A' if plan.score_refresh else 'D' if plan.decision_refresh else 'H'
        if plan.score_refresh:
            cache.publish_scores(ident, step, object())
        if plan.decision_refresh:
            cache.publish_decision(ident, step, object())
    return ''.join(phases[s] for s in range(first, last))


def test_origin0_is_unchanged_incumbent_clock():
    assert _schedule(3, 0, 0) == 'AHHDHHDHAHHDHHDHAH'


def test_bootstrap_origin1_m3_r3_anchor_at_9_and_d_at_4_7_12():
    # Calls 0 and 1 are handled by the bootstrap path; call 1 publishes both.
    assert _schedule(3, 1, 1) == 'AHHDHHDHAHHDHHDHA'


def test_bootstrap_m1_and_matched_b():
    assert _schedule(1, 1, 1) == 'ADDDDDDDADDDDDDDA'
    assert _schedule(8, 1, 1) == 'AHHHHHHHAHHHHHHHA'


def test_origin_must_be_declared():
    with pytest.raises(ValueError):
        ScoreCache(8, 3, origin=2)
