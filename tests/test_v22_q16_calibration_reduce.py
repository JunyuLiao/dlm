"""Tests for scripts/v22_q16_calibration_reduce.py.

Pytest-style, but also runnable directly:
    PYTHONPATH=. python tests/test_v22_q16_calibration_reduce.py
(pytest is not installed in this environment).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.v22_q16_calibration_reduce import (  # noqa: E402
    EXPECTED_OFFSETS,
    aggregate_work,
    build_point_summary,
    interpolate_error_ratio_at_work_ratio_one,
    main,
    nonmonotonic_pairs,
    per_state_error_ratio,
    pooled_relative_l2,
    rule1_select,
    rule2_select,
    verify_receipts,
)


# ----------------------------------------------------------------------
# Synthetic receipt builder
# ----------------------------------------------------------------------

def make_candidate(offset, threshold, retained, abs_err, ref, relative_l2=None,
                    row_p99=0.5, row_max=0.9, removed_p99=0.4, removed_max=0.8,
                    bitmap_bytes=100, near_threshold=1):
    if relative_l2 is None:
        relative_l2 = abs_err / ref
    return {
        "offset": offset,
        "threshold": threshold,
        "retained_legal_pairs": retained,
        "output_vs_full_fp32": {
            "absolute_error_l2": abs_err,
            "reference_l2": ref,
            "relative_l2": relative_l2,
        },
        "output_relative_l2_per_row": {"p99": row_p99, "max": row_max},
        "removed_mass_per_row": {"p99": removed_p99, "max": removed_max},
        "bitmap_bytes": bitmap_bytes,
        "risk_groups_within_1e_3_of_threshold": near_threshold,
    }


def make_mq(inherited_threshold, denom, coarse, candidates_by_offset):
    return {
        "inherited_threshold": inherited_threshold,
        "denominator_legal_pairs": denom,
        "offsets": EXPECTED_OFFSETS,
        "coarse_baseline": coarse,
        "q16_candidates": [candidates_by_offset[o] for o in EXPECTED_OFFSETS],
    }


def make_layer(attention_kind="LOCAL", query_sensitivity="neutral", parity_status="match",
               geometries=None, extra_comparison=None):
    if geometries is None:
        geometries = {
            "geomA": {
                "kept_legal_pair_fraction": 0.5,
                "output_vs_full_fp32": {"relative_l2": 0.2},
                "output_relative_l2_per_row": {"p99": 0.3, "max": 0.6},
                "removed_mass_per_row": {"p99": 0.4, "max": 0.7},
            }
        }
    comparison = {
        "coarse_route_only_parity": {"status": parity_status},
        "independent_sequential": geometries,
    }
    if extra_comparison:
        comparison.update(extra_comparison)
    return {
        "attention_kind": attention_kind,
        "status": "qualified",
        "geometry": {
            "query_sensitivity": query_sensitivity,
            "comparison": comparison,
        },
    }


def make_group(dataset, group_id, call0_layers, call3_layer5_extra=None):
    calls = {
        "0": {"layers": call0_layers},
    }
    if call3_layer5_extra is not None:
        calls["3"] = {
            "layers": {
                "0": make_layer(),
                "5": make_layer(attention_kind="GLOBAL", extra_comparison={
                    "matched_q16_calibration": call3_layer5_extra,
                }),
            }
        }
    return {"dataset": dataset, "id": group_id, "calls": calls}


def make_receipt(source_commit, mq, label="host"):
    call0_layers = {"0": make_layer("LOCAL"), "5": make_layer("GLOBAL")}
    groups = {
        f"aime26|{label}": make_group("aime26", f"aime26/{label}", call0_layers),
        f"ruler4k|{label}": make_group("ruler4k", f"ruler4k/{label}", call0_layers),
        f"longbench_v2|{label}": make_group(
            "longbench_v2", f"longbench_v2/{label}", call0_layers, call3_layer5_extra=mq,
        ),
    }
    return {
        "source_commit": source_commit,
        "completeness": {"status": "complete"},
        "qualification_tests": {"status": "passed"},
        "groups": groups,
    }


def default_two_receipts():
    """Two synthetic receipts with a clean, well-behaved calibration curve."""
    inherited_threshold = -3.0
    candidates_mpk = {
        0.0: make_candidate(0.0, -3.0, retained=100, abs_err=30, ref=100),
        -0.25: make_candidate(-0.25, -3.25, retained=130, abs_err=25, ref=100),
        -0.5: make_candidate(-0.5, -3.5, retained=170, abs_err=20, ref=100),
        -1.0: make_candidate(-1.0, -4.0, retained=230, abs_err=12, ref=100),
        -2.0: make_candidate(-2.0, -5.0, retained=400, abs_err=5, ref=100),
    }
    coarse_mpk = make_candidate(None, -3.0, retained=200, abs_err=18, ref=100)
    mq_mpk = make_mq(inherited_threshold, denom=1000, coarse=coarse_mpk,
                      candidates_by_offset=candidates_mpk)

    candidates_dllm = {
        0.0: make_candidate(0.0, -3.0, retained=110, abs_err=28, ref=90),
        -0.25: make_candidate(-0.25, -3.25, retained=140, abs_err=23, ref=90),
        -0.5: make_candidate(-0.5, -3.5, retained=180, abs_err=19, ref=90),
        -1.0: make_candidate(-1.0, -4.0, retained=240, abs_err=11, ref=90),
        -2.0: make_candidate(-2.0, -5.0, retained=410, abs_err=4, ref=90),
    }
    coarse_dllm = make_candidate(None, -3.0, retained=210, abs_err=16, ref=90)
    mq_dllm = make_mq(inherited_threshold, denom=1200, coarse=coarse_dllm,
                       candidates_by_offset=candidates_dllm)

    receipt_mpk = make_receipt("commitABC", mq_mpk, label="mpk")
    receipt_dllm = make_receipt("commitABC", mq_dllm, label="dllm")
    return receipt_mpk, receipt_dllm


# ----------------------------------------------------------------------
# 1. pooled-vs-average difference
# ----------------------------------------------------------------------

def test_pooled_error_differs_from_naive_average():
    # host A: abs_err=3, ref=10 -> relative_l2=0.3
    # host B: abs_err=4, ref=1  -> relative_l2=4.0
    abs_errors = [3.0, 4.0]
    references = [10.0, 1.0]

    naive_average = sum(e / r for e, r in zip(abs_errors, references)) / 2
    pooled = pooled_relative_l2(abs_errors, references)

    expected_pooled = math.sqrt((3.0 ** 2 + 4.0 ** 2) / (10.0 ** 2 + 1.0 ** 2))
    assert math.isclose(pooled, expected_pooled, rel_tol=1e-12)
    assert not math.isclose(pooled, naive_average, rel_tol=1e-6), (
        "pooled error must not collapse to the naive average of per-state relative errors"
    )


# ----------------------------------------------------------------------
# 2. zero-reference raises
# ----------------------------------------------------------------------

def test_zero_reference_raises():
    try:
        pooled_relative_l2([1.0, 2.0], [0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero summed reference")


def test_per_state_zero_coarse_raises():
    try:
        per_state_error_ratio(0.5, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero coarse relative_l2")


# ----------------------------------------------------------------------
# 3. rule-1 tie toward more retention
# ----------------------------------------------------------------------

def test_rule1_tie_prefers_more_retention():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    work_coarse = 100
    # -0.5 -> 90 (distance 10), -1.0 -> 110 (distance 10): exact tie in distance,
    # rule1 must pick -1.0 because it retains more (110 > 90).
    work_by_offset = {0.0: 40, -0.25: 70, -0.5: 90, -1.0: 110, -2.0: 200}
    result = rule1_select(offsets, work_by_offset, work_coarse)
    assert result["offset"] == -1.0, result
    assert result["work_mismatch"] == 10


# ----------------------------------------------------------------------
# 4. rule-2 per-state cap excludes a point that passes the pooled band
# ----------------------------------------------------------------------

def test_rule2_per_state_cap_excludes_pooled_pass():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    # offset -0.5 passes the pooled 1.05 band but host 'b' individually is 1.20 (>1.15)
    pooled_error_ratio_by_offset = {0.0: 1.5, -0.25: 1.3, -0.5: 1.02, -1.0: 0.9, -2.0: 0.5}
    per_state_ratio_by_offset = {
        0.0: {"a": 1.6, "b": 1.4},
        -0.25: {"a": 1.4, "b": 1.2},
        -0.5: {"a": 0.95, "b": 1.20},  # b exceeds 1.15 cap despite pooled passing
        -1.0: {"a": 0.9, "b": 0.9},
        -2.0: {"a": 0.5, "b": 0.5},
    }
    work_by_offset = {0.0: 40, -0.25: 70, -0.5: 90, -1.0: 500, -2.0: 900}
    result = rule2_select(offsets, pooled_error_ratio_by_offset, per_state_ratio_by_offset, work_by_offset)
    assert result is not None
    assert result["offset"] != -0.5, "offset -0.5 must be excluded by the per-state 1.15 cap"
    # only -1.0 and -2.0 qualify (both per-state ratios <=1.15); minimum W among them is -1.0
    assert result["offset"] == -1.0, result


# ----------------------------------------------------------------------
# 5. rule-2 returns null when nothing qualifies
# ----------------------------------------------------------------------

def test_rule2_returns_none_when_nothing_qualifies():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    pooled_error_ratio_by_offset = {0.0: 1.5, -0.25: 1.3, -0.5: 1.2, -1.0: 1.1, -2.0: 1.06}
    per_state_ratio_by_offset = {o: {"a": 1.5, "b": 1.5} for o in offsets}
    work_by_offset = {0.0: 40, -0.25: 70, -0.5: 90, -1.0: 500, -2.0: 900}
    result = rule2_select(offsets, pooled_error_ratio_by_offset, per_state_ratio_by_offset, work_by_offset)
    assert result is None


# ----------------------------------------------------------------------
# 6. nonmonotonic detection
# ----------------------------------------------------------------------

def test_nonmonotonic_detection():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    # W decreases from -0.25 (100) to -0.5 (95): flagged.
    work_by_offset = {0.0: 40, -0.25: 100, -0.5: 95, -1.0: 200, -2.0: 400}
    result = nonmonotonic_pairs(offsets, work_by_offset)
    assert len(result) == 1
    assert result[0]["from_offset"] == -0.25
    assert result[0]["to_offset"] == -0.5


def test_monotonic_has_no_flags():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    work_by_offset = {0.0: 40, -0.25: 70, -0.5: 90, -1.0: 200, -2.0: 400}
    assert nonmonotonic_pairs(offsets, work_by_offset) == []


# ----------------------------------------------------------------------
# 7. interpolation bracketing / null
# ----------------------------------------------------------------------

def test_interpolation_bracketed():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    work_ratio_by_offset = {0.0: 0.5, -0.25: 0.7, -0.5: 0.85, -1.0: 1.2, -2.0: 2.0}
    error_ratio_by_offset = {0.0: 1.5, -0.25: 1.3, -0.5: 1.1, -1.0: 0.9, -2.0: 0.5}
    result = interpolate_error_ratio_at_work_ratio_one(offsets, work_ratio_by_offset, error_ratio_by_offset)
    assert result is not None
    assert result["bracket_offsets"] == [-0.5, -1.0]

    xa, xb = math.log(0.85), math.log(1.2)
    t = (0.0 - xa) / (xb - xa)
    expected = 1.1 + t * (0.9 - 1.1)
    assert math.isclose(result["interpolated_error_ratio"], expected, rel_tol=1e-9)


def test_interpolation_null_when_not_bracketed():
    offsets = [0.0, -0.25, -0.5, -1.0, -2.0]
    # all work ratios < 1.0: never brackets 1.0
    work_ratio_by_offset = {0.0: 0.1, -0.25: 0.2, -0.5: 0.3, -1.0: 0.4, -2.0: 0.5}
    error_ratio_by_offset = {0.0: 1.5, -0.25: 1.3, -0.5: 1.1, -1.0: 0.9, -2.0: 0.5}
    result = interpolate_error_ratio_at_work_ratio_one(offsets, work_ratio_by_offset, error_ratio_by_offset)
    assert result is None


# ----------------------------------------------------------------------
# 8. refusal to overwrite outputs
# ----------------------------------------------------------------------

def test_refuses_to_overwrite_existing_output():
    receipt_mpk, receipt_dllm = default_two_receipts()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mpk_path = tmp / "mpk.json"
        dllm_path = tmp / "dllm.json"
        mpk_path.write_text(json.dumps(receipt_mpk))
        dllm_path.write_text(json.dumps(receipt_dllm))

        out_json = tmp / "out.json"
        out_csv = tmp / "out.csv"
        out_json.write_text("{}")  # pre-existing output

        rc = main([
            "--receipt", f"mpk={mpk_path}",
            "--receipt", f"dllm={dllm_path}",
            "--out-json", str(out_json),
            "--out-csv", str(out_csv),
        ])
        assert rc != 0
        assert not out_csv.exists(), "must not write out-csv when out-json already exists"


# ----------------------------------------------------------------------
# 9. mismatched source_commit raises
# ----------------------------------------------------------------------

def test_mismatched_source_commit_raises():
    receipt_mpk, receipt_dllm = default_two_receipts()
    receipt_dllm["source_commit"] = "commitDIFFERENT"
    try:
        verify_receipts({"mpk": receipt_mpk, "dllm": receipt_dllm})
    except ValueError as exc:
        assert "source_commit" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched source_commit")


def test_mismatched_inherited_threshold_raises():
    receipt_mpk, receipt_dllm = default_two_receipts()
    lb_key = [k for k in receipt_dllm["groups"] if receipt_dllm["groups"][k]["dataset"] == "longbench_v2"][0]
    mq = receipt_dllm["groups"][lb_key]["calls"]["3"]["layers"]["5"]["geometry"]["comparison"]["matched_q16_calibration"]
    mq["inherited_threshold"] = -999.0
    try:
        verify_receipts({"mpk": receipt_mpk, "dllm": receipt_dllm})
    except ValueError as exc:
        assert "inherited_threshold" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched inherited_threshold")


def test_bad_offsets_order_raises():
    receipt_mpk, receipt_dllm = default_two_receipts()
    lb_key = [k for k in receipt_mpk["groups"] if receipt_mpk["groups"][k]["dataset"] == "longbench_v2"][0]
    mq = receipt_mpk["groups"][lb_key]["calls"]["3"]["layers"]["5"]["geometry"]["comparison"]["matched_q16_calibration"]
    mq["offsets"] = [0.0, -0.5, -0.25, -1.0, -2.0]
    try:
        verify_receipts({"mpk": receipt_mpk, "dllm": receipt_dllm})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-order/incorrect offsets")


# ----------------------------------------------------------------------
# End-to-end sanity on the synthetic pair (build_point_summary + rules)
# ----------------------------------------------------------------------

def test_end_to_end_point_summary_on_synthetic_receipts():
    receipt_mpk, receipt_dllm = default_two_receipts()
    _source_commit, _lb_groups, mqs = verify_receipts({"mpk": receipt_mpk, "dllm": receipt_dllm})
    summary = build_point_summary(mqs)
    assert summary["points"]["coarse"]["aggregate_work"] == 200 + 210
    assert summary["rule1_work_matched"]["offset"] in EXPECTED_OFFSETS
    # rule2 may or may not find a qualifying point depending on synthetic numbers;
    # just assert the call succeeds and returns either None or a dict with 'offset'.
    r2 = summary["rule2_min_work_under_caps"]
    assert r2 is None or "offset" in r2


ALL_TESTS = [
    test_pooled_error_differs_from_naive_average,
    test_zero_reference_raises,
    test_per_state_zero_coarse_raises,
    test_rule1_tie_prefers_more_retention,
    test_rule2_per_state_cap_excludes_pooled_pass,
    test_rule2_returns_none_when_nothing_qualifies,
    test_nonmonotonic_detection,
    test_monotonic_has_no_flags,
    test_interpolation_bracketed,
    test_interpolation_null_when_not_bracketed,
    test_refuses_to_overwrite_existing_output,
    test_mismatched_source_commit_raises,
    test_mismatched_inherited_threshold_raises,
    test_bad_offsets_order_raises,
    test_end_to_end_point_summary_on_synthetic_receipts,
]


if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL: {t.__name__}: {exc}")
        else:
            print(f"OK: {t.__name__}")
    if failures:
        print(f"\n{failures} test(s) FAILED")
        raise SystemExit(1)
    print(f"\nAll {len(ALL_TESTS)} tests OK")
