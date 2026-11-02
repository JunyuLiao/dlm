import pytest

from scripts.multi_model_aime_rematch import next_threshold, pool


def test_pool_uses_physical_tiles_not_average_request_sparsity():
    rows = [{"id": str(i), "seed": 42, "counts": {"kept_tiles": k, "eligible_tiles": e}}
            for i, (k, e) in enumerate(((1, 10), (90, 100)))]
    assert pool(rows)["sparsity"] == pytest.approx(19 / 110)
    with pytest.raises(ValueError, match="duplicate"):
        pool(rows + rows[:1])


def test_nonmonotone_history_uses_a_valid_ordered_bracket():
    trials = [{"threshold": t, "sparsity": s} for t, s in
              ((.01, .3), (.02, .6), (.03, .4), (.04, .7))]
    assert .03 < next_threshold(trials, 1e-6, 1) < .04
