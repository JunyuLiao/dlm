import pytest

from experiments.value_direction_hopper.aime_query_sensitivity_uniform_report import (
    question_disagreements, summarize,
)


def _row(question, seed, score, iterations, eligible=100, skipped=50):
    return dict(id=f"aime26/{question}", seed=seed, score=score,
                steps=sum(iterations),
                canvases=[dict(iterations=n) for n in iterations],
                counts={k: dict(eligible=eligible, skipped=skipped)
                        for k in ("whole", "local", "global")},
                phase_counts={p: {k: dict(eligible=eligible, skipped=skipped)
                                 for k in ("whole", "local", "global")}
                              for p in ("call1", "call2", "late")})


def test_report_uses_canvas_quantiles_and_pooled_tile_counts():
    rows = [_row(1, 42, 1, [2, 4], eligible=100, skipped=25),
            _row(2, 42, 0, [6, 48], eligible=300, skipped=225)]
    result = summarize(rows)
    assert result["accuracy"] == .5
    assert result["overall_sparsity"] == .625
    assert result["call1_whole_sparsity"] == .625
    assert result["mean_canvas_steps"] == 15.
    assert result["median_canvas_steps"] == 5.
    assert result["p90_canvas_steps"] == pytest.approx(35.4)
    assert result["cap_canvases"] == 1


def test_report_disagreement_requires_exactly_one_row_per_question_seed():
    rows = [_row(1, seed, int(seed != 43), [2]) for seed in (42, 43, 44)]
    result = question_disagreements("T_prior", rows)
    assert result[0]["seed_disagreement"] == 1
    assert result[0]["correct_seed_count"] == 2
    with pytest.raises(ValueError, match="incomplete question"):
        question_disagreements("T_prior", rows[:2])
    with pytest.raises(ValueError, match="duplicate question/seed"):
        question_disagreements("T_prior", rows + rows[:1])
