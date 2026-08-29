from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks/diffusion_gemma_math500/run_blockmax_quantiles.py"
)
SPEC = importlib.util.spec_from_file_location("math500_blockmax_quantiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gym_verify_symbolic_equivalence_and_extraction() -> None:
    verifier = MODULE.build_math_verifier()
    score, answer = MODULE.gym_verify(
        verifier,
        r"\frac{1}{2}",
        r"The value is therefore \boxed{0.5}.",
    )
    assert score == 1.0
    assert answer is not None


def test_gym_metrics_matches_majority_self_consistency_contract() -> None:
    rows = []
    # Problem 0: answer 4 is the mode and correct. Problem 1: answer 8 is the
    # mode and wrong, although one of its ten rollouts is correct.
    for problem_index, answers in enumerate(
        [
            [("4", 1.0)] * 6 + [("3", 0.0)] * 4,
            [("8", 0.0)] * 6 + [("7", 1.0)] + [(None, 0.0)] * 3,
        ]
    ):
        for sample_index, (answer, reward) in enumerate(answers):
            rows.append(
                {
                    "subset_index": problem_index,
                    "dataset_index": problem_index,
                    "sample_index": sample_index,
                    "unique_id": str(problem_index),
                    "subject": "Algebra",
                    "level": 1,
                    "extracted_answer": answer,
                    "library_reward": reward,
                }
            )
    metrics, per_problem = MODULE.gym_metrics(rows, 10)
    assert metrics["majority_at_10_symbolic_accuracy"] == 0.5
    assert metrics["pass_at_1_avg_of_10_symbolic_accuracy"] == pytest.approx(0.35)
    assert metrics["pass_at_10_symbolic_accuracy"] == 1.0
    assert per_problem[0]["majority_votes"] == 6
    assert per_problem[1]["majority_at_10"] == 0.0


def test_condition_names() -> None:
    assert MODULE.condition_name(0) == "dense_eager"
    assert MODULE.condition_name(95) == "block_max_k95"
