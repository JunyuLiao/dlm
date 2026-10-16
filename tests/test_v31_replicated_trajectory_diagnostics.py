import pytest

from scripts.v31_replicated_trajectory_diagnostics import outcome_group, summarize_cells


def test_outcome_classification_does_not_treat_fractional_score_as_binary():
    assert outcome_group([100, 100], [100, 100], 100) == "always_full_credit_both"
    assert outcome_group([0, 0], [0, 0], 100) == "always_zero_both"
    assert outcome_group([50, 50], [50, 50], 100) == "equal_mixed_score"
    assert outcome_group([0, 100], [100, 100], 100) == "higher_mean_score"
    assert outcome_group([100, 100], [0, 100], 100) == "lower_mean_score"


def test_product_decomposition_separates_canvases_and_calls_per_canvas():
    cells = [{"control": {"N": 100, "C": 10, "T": 2500},
              "candidate": {"N": 64, "C": 8, "T": 2000}}]
    result = summarize_cells(cells)
    assert result["delta"]["N"] == -36
    assert result["symmetric_N_decomposition"]["canvas_count_term"] == pytest.approx(-18)
    assert result["symmetric_N_decomposition"]["calls_per_canvas_term"] == pytest.approx(-18)
    assert result["N_changes"] == {"fewer": 1, "equal": 0, "more": 0}
