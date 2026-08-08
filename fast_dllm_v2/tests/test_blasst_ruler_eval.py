from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.eval_blasst_ruler_legacy import (  # noqa: E402
    PAPER_TASKS,
    RULER_COMMIT,
    SHORT_512_TASKS,
    _balanced_counts,
    _length_distribution,
    _length_tolerance,
    _official_postprocess,
    _reference_hit_pattern,
    _score_one,
    _score_subset,
)


def test_ruler_task_mixtures_and_balance_are_deterministic() -> None:
    assert len(RULER_COMMIT) == 40
    assert PAPER_TASKS == (
        "niah_multikey_1",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "fwe",
    )
    assert SHORT_512_TASKS == ("niah_multikey_2", "fwe")
    assert _balanced_counts(PAPER_TASKS, 32) == {
        "niah_multikey_1": 7,
        "niah_multivalue": 7,
        "niah_multiquery": 6,
        "vt": 6,
        "fwe": 6,
    }
    assert set(_balanced_counts(PAPER_TASKS, 100).values()) == {20}


def test_ruler_length_tolerance_has_absolute_floor() -> None:
    args = SimpleNamespace(
        length_tolerance_min_tokens=32,
        length_tolerance_fraction=0.06,
    )
    assert _length_tolerance(512, args) == 32
    assert _length_tolerance(8192, args) == 492


def test_official_style_scoring_and_answer_agreement() -> None:
    def official_string_match_all(
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        score = sum(
            sum(
                1.0 if reference.lower() in prediction.lower() else 0.0
                for reference in expected
            )
            / len(expected)
            for prediction, expected in zip(predictions, references)
        )
        return round(score / len(predictions) * 100, 2)

    references = ["needle-a", "needle-b"]
    assert _official_postprocess(" \tneedle-a\x00noise ") == "needle-a\nnoise"
    assert _score_one(
        official_string_match_all,
        "needle-a only",
        references,
    ) == pytest.approx(0.5)
    assert _reference_hit_pattern("needle-a only", references) == (
        True,
        False,
    )
    assert _reference_hit_pattern("Needle-B then needle-a", references) == (
        True,
        True,
    )


def test_accuracy_summary_invokes_official_scorer_over_full_subset() -> None:
    def official_string_match_all(
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        score = sum(
            sum(
                1.0 if reference.lower() in prediction.lower() else 0.0
                for reference in expected
            )
            / len(expected)
            for prediction, expected in zip(predictions, references)
        )
        return round(score / len(predictions) * 100, 2)

    rows = [
        {
            "task_base": "niah",
            "dense_prediction": "a",
            "outputs": ["a", "b", "c", "d"],
        },
        {
            "task_base": "niah",
            "dense_prediction": "e",
            "outputs": ["e", "f", "g"],
        },
    ]
    score = _score_subset(
        {"niah": official_string_match_all},
        rows,
        "dense_prediction",
    )
    assert score == pytest.approx(0.2917)
    per_example_mean = sum(
        _score_one(
            official_string_match_all,
            str(row["dense_prediction"]),
            list(row["outputs"]),
        )
        for row in rows
    ) / len(rows)
    assert per_example_mean == pytest.approx(0.29165)
    assert score != per_example_mean


def test_length_distribution_counts_each_record_once() -> None:
    assert _length_distribution(
        [
            {"target_length": 512, "actual_prompt_length": 500},
            {"target_length": 512, "actual_prompt_length": 508},
            {"target_length": 1024, "actual_prompt_length": 1000},
        ]
    ) == [
        {
            "target": 512,
            "samples": 2,
            "minimum": 500,
            "mean": 504.0,
            "maximum": 508,
        },
        {
            "target": 1024,
            "samples": 1,
            "minimum": 1000,
            "mean": 1000.0,
            "maximum": 1000,
        },
    ]
