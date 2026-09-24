"""The v6 prose claimed the new M1 successor matched dense AND fresh T
"exactly (same missed question)". Records say otherwise for T. These tests
pin the actual vectors and make the wrong phrasing raise, so a later summary
cannot restate it.
"""
import json
from pathlib import Path

import pytest

from scripts.preqk_correctness_vectors import (IDS, agreement_label, build,
                                               new_arm_vector, old_arm_vectors,
                                               vector, verify_claim)

ROOT = Path(__file__).resolve().parents[1]
REQUEST_TABLE = ROOT / "results/numerical_qk_reuse_20260924/request_table.csv"
M1_QUALITY = ROOT / "results/numerical_qk_reuse_recovery_20260924/panel/quality.M1.json"
M3_QUALITY = ROOT / "results/numerical_qk_reuse_recovery_20260924/panel/quality.M3.json"

pytestmark = pytest.mark.skipif(not REQUEST_TABLE.is_file(), reason="frozen records absent")


def test_recorded_vectors_are_what_the_receipts_say():
    old = old_arm_vectors(REQUEST_TABLE)
    assert vector(old["native_dense"]) == (True, True, False, True)
    assert vector(old["fresh_junyu_T"]) == (True, True, True, False)
    assert vector(new_arm_vector(M1_QUALITY)) == (True, True, False, True)
    assert vector(new_arm_vector(M3_QUALITY)) == (True, True, False, False)


def test_t_answers_14_uncapped_and_misses_20():
    """The specific facts the v6 'same missed question' claim contradicted."""
    rows = old_arm_vectors(REQUEST_TABLE)["fresh_junyu_T"]
    assert rows["aime26/14"]["correct"] and not rows["aime26/14"]["capped"]
    assert rows["aime26/14"]["output_tokens"] == 7276
    assert not rows["aime26/20"]["correct"] and not rows["aime26/20"]["capped"]
    assert rows["aime26/20"]["output_tokens"] == 2777


def test_equal_counts_never_pass_as_vector_identity():
    dense = vector(old_arm_vectors(REQUEST_TABLE)["native_dense"])
    fresh_t = vector(old_arm_vectors(REQUEST_TABLE)["fresh_junyu_T"])
    m1 = vector(new_arm_vector(M1_QUALITY))
    assert sum(fresh_t) == sum(m1) == 3
    # The v6 sentence, expressed as a checkable claim, must fail.
    with pytest.raises(AssertionError):
        verify_claim(m1, fresh_t, claim_same_vector=True)
    # What the records actually support.
    verify_claim(m1, dense, claim_same_vector=True)
    verify_claim(m1, fresh_t, claim_same_vector=False)
    assert "DIFFERENT per-ID vector" in agreement_label(m1, fresh_t)
    assert agreement_label(m1, dense) == "identical per-ID correctness vector"


def test_build_reports_pairwise_labels_without_prose():
    payload = build(REQUEST_TABLE, {"M1_routing_only": M1_QUALITY})
    assert payload["pairwise"]["M1_routing_only vs native_dense"] == (
        "identical per-ID correctness vector")
    assert "DIFFERENT per-ID vector" in payload["pairwise"][
        "M1_routing_only vs fresh_junyu_T"]
    assert list(payload["ids"]) == list(IDS)
