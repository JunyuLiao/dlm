"""Plain-function tests for scripts/v25b_alignment_exposure.py.

No pytest available in this environment: every test is a plain function
named test_*, and this file is run directly:

    PYTHONPATH=. python tests/test_v25b_exposure.py

All test cases are synthetic / self-contained (no dependency on the
private pilot files). Several cases hardcode numbers that were hand-
verified against one real M3_boot_aligned16 ledger row
(longbench_v2/66ed2c87821e116aacb1f149, seed 101, prompt_token_count
19292, per_canvas_calls [12, 14, 17, 17, 2]) before the reducer script
was written, so they double as a regression check against that
real-world example.
"""
from __future__ import annotations

import sys

from scripts.v25b_alignment_exposure import (
    a_schedule_calls,
    canvas_key_count,
    check_alignment_conservation,
    check_phase_conservation,
    classify_arms,
    kdiv_of,
    pitch_for_k,
    predict_alignment_exposure,
    predict_canvas_phases,
    predict_request_phases,
    raw_phase_counts,
)


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


# ----------------------------------------------------------------------
# A-schedule
# ----------------------------------------------------------------------

def test_a_schedule_calls_basic():
    _assert_eq(a_schedule_calls(9), set(), "n=9 excludes call 9 itself (< n)")
    _assert_eq(a_schedule_calls(10), {9}, "n=10 includes call 9")
    _assert_eq(a_schedule_calls(17), {9}, "n=17 excludes call 17 itself (< n)")
    _assert_eq(a_schedule_calls(18), {9, 17}, "n=18 includes calls 9 and 17")
    _assert_eq(a_schedule_calls(42), {9, 17, 25, 33, 41}, "n=42 full schedule below 42")


# ----------------------------------------------------------------------
# Phase prediction: M3 (decision_interval=3)
# ----------------------------------------------------------------------

def test_phase_prediction_m3_single_call_canvas():
    # n=1: only B0 exists, no BO/A/D/H.
    ph = predict_canvas_phases(1, decision_interval=3)
    _assert_eq(ph, dict(B0=1, BO=0, A=0, D=0, H=0))
    _assert_eq(sum(ph.values()), 1)


def test_phase_prediction_m3_two_call_canvas():
    # n=2: B0 then BO, nothing else.
    ph = predict_canvas_phases(2, decision_interval=3)
    _assert_eq(ph, dict(B0=1, BO=1, A=0, D=0, H=0))
    _assert_eq(sum(ph.values()), 2)


def test_phase_prediction_m3_canvas_reaching_call17():
    # Hand-derived schedule for n=17, interval=3, matching the task's
    # worked example: decisions at 1,4,7, A at 9 (resets), 12, 15, then
    # call 16 is held (16-15=1, next D would be at 18 which is >= n).
    ph = predict_canvas_phases(17, decision_interval=3)
    _assert_eq(ph, dict(B0=1, BO=1, A=1, D=4, H=10))
    _assert_eq(sum(ph.values()), 17)


def test_phase_prediction_m3_canvas_reaching_call41_a_schedule_full():
    # n=42 exercises every A-schedule slot (9,17,25,33,41) at least once.
    ph = predict_canvas_phases(42, decision_interval=3)
    _assert_eq(ph["A"], 5)
    _assert_eq(sum(ph.values()), 42)


# ----------------------------------------------------------------------
# Phase prediction: M1 (decision_interval=1)
# ----------------------------------------------------------------------

def test_phase_prediction_m1_canvas_reaching_call17():
    # interval=1: every call >= 2 is D except the one A call at 9.
    ph = predict_canvas_phases(17, decision_interval=1)
    _assert_eq(ph, dict(B0=1, BO=1, A=1, D=14, H=0))
    _assert_eq(sum(ph.values()), 17)


# ----------------------------------------------------------------------
# Phase prediction: B (bootstrap-only, decision_interval=None -> no D ever)
# ----------------------------------------------------------------------

def test_phase_prediction_b_bootstrap_only_canvas_reaching_call17():
    ph = predict_canvas_phases(17, decision_interval=None)
    _assert_eq(ph["D"], 0, "bootstrap-only arm must never produce a D call")
    _assert_eq(ph, dict(B0=1, BO=1, A=1, D=0, H=14))
    _assert_eq(sum(ph.values()), 17)


def test_phase_prediction_b_bootstrap_only_single_call_canvas():
    ph = predict_canvas_phases(1, decision_interval=None)
    _assert_eq(ph, dict(B0=1, BO=0, A=0, D=0, H=0))


# ----------------------------------------------------------------------
# K / K%16 / KDIV / pitch
# ----------------------------------------------------------------------

def test_canvas_key_count_formula():
    # K_c = prompt_token_count + 256*(c+1), c 0-based.
    _assert_eq(canvas_key_count(19292, 0), 19548)
    _assert_eq(canvas_key_count(19292, 4), 20572)


def test_k_mod16_invariant_across_canvases():
    prompt = 19292
    for c in range(6):
        k = canvas_key_count(prompt, c)
        _assert_eq(k % 16, prompt % 16, f"K%16 must be invariant across canvases (c={c})")


def test_kdiv_of_various():
    _assert_eq(kdiv_of(19292), 4)     # 19292 % 4 == 0, % 8 != 0
    _assert_eq(kdiv_of(3315), 1)      # odd -> KDIV=1, exercises the odd-K path
    _assert_eq(kdiv_of(1600), 16)     # divisible by 16 -> fully aligned
    _assert_eq(kdiv_of(3949), 1)      # 3949 is odd (ruler4k single example)
    _assert_eq(kdiv_of(131), 1)       # aime26 example, 131 is odd


def test_pitch_for_k_rounds_up_to_16():
    _assert_eq(pitch_for_k(19548), 19552)
    _assert_eq(pitch_for_k(16), 16)   # already aligned: no rounding needed
    _assert_eq(pitch_for_k(17), 32)


# ----------------------------------------------------------------------
# Alignment exposure prediction
# ----------------------------------------------------------------------

def test_alignment_exposure_matches_real_pilot_example():
    # Real M3_boot_aligned16 example: prompt_token_count=19292 (%16=12,
    # unaligned), per_canvas_calls=[12,14,17,17,2]. Ledger counters
    # recorded aligned_score_copies=45, aligned_sketch_pads=110,
    # aligned_pad_bytes=14750842880 -- hand-verified before this reducer
    # was written.
    per_canvas_calls = [12, 14, 17, 17, 2]
    _, per_canvas_phases = predict_request_phases(per_canvas_calls, decision_interval=3)
    result = predict_alignment_exposure(per_canvas_calls, per_canvas_phases, 19292, is_aligned_arm=True)
    _assert_eq(result["k_mod16"], 12)
    _assert_eq(result["kdiv"], 4)
    _assert_eq(result["predicted_aligned_score_copies"], 45)
    _assert_eq(result["predicted_aligned_sketch_pads"], 110)
    _assert_eq(result["predicted_aligned_pad_bytes"], 14750842880)
    _assert_eq(result["affected_bo_plus_a_calls"], 9)   # sum(BO_c+A_c)
    _assert_eq(result["affected_d_calls"], 13)          # sum(D_c)


def test_alignment_exposure_already_aligned_k_is_zero():
    # prompt_token_count % 16 == 0 -> aligned16 has nothing to copy,
    # regardless of arm.
    per_canvas_calls = [5]
    _, per_canvas_phases = predict_request_phases(per_canvas_calls, decision_interval=3)
    result = predict_alignment_exposure(per_canvas_calls, per_canvas_phases, 1600, is_aligned_arm=True)
    _assert_eq(result["k_mod16"], 0)
    _assert_eq(result["predicted_aligned_score_copies"], 0)
    _assert_eq(result["predicted_aligned_pad_bytes"], 0)
    _assert_eq(result["predicted_aligned_sketch_pads"], 0)
    _assert_eq(result["affected_bo_plus_a_calls"], 0)
    _assert_eq(result["affected_d_calls"], 0)


def test_alignment_exposure_logical_arm_has_zero_copies_but_reports_exposure():
    # A logical-route-storage arm never actually copies, but the request
    # is still "exposed" (would be affected if it used aligned16) -- the
    # affected_* counters characterize the request, not the arm.
    per_canvas_calls = [12, 14, 17, 17, 2]
    _, per_canvas_phases = predict_request_phases(per_canvas_calls, decision_interval=3)
    result = predict_alignment_exposure(per_canvas_calls, per_canvas_phases, 19292, is_aligned_arm=False)
    _assert_eq(result["predicted_aligned_score_copies"], 0)
    _assert_eq(result["predicted_aligned_pad_bytes"], 0)
    _assert_eq(result["predicted_aligned_sketch_pads"], 0)
    _assert_eq(result["affected_bo_plus_a_calls"], 9)
    _assert_eq(result["affected_d_calls"], 13)


# ----------------------------------------------------------------------
# Conservation mismatch detection
# ----------------------------------------------------------------------

def test_raw_phase_counts_matches_real_pilot_example():
    counters = dict(
        bootstrap_dense_calls=25, bootstrap_observation_calls=25,
        score_refresh_calls=20, decision_refresh_calls=85, held_decision_calls=175,
    )
    raw, problems = raw_phase_counts(counters)
    _assert_eq(problems, [])
    _assert_eq(raw, dict(B0=5, BO=5, A=4, D=13, H=35))

    predicted_totals, _ = predict_request_phases([12, 14, 17, 17, 2], decision_interval=3)
    ok, detail = check_phase_conservation(predicted_totals, raw, [], decoder_calls=62)
    _assert_eq(ok, True, detail)


def test_phase_conservation_flags_non_multiple_of_5():
    counters = dict(
        bootstrap_dense_calls=26,  # not a multiple of 5 -> must be flagged, not silently rounded
        bootstrap_observation_calls=25,
        score_refresh_calls=20, decision_refresh_calls=85, held_decision_calls=175,
    )
    raw, problems = raw_phase_counts(counters)
    assert raw is None or problems, "a non-multiple-of-5 counter must produce a reported problem"
    assert any("bootstrap_dense_calls" in p for p in problems)


def test_phase_conservation_flags_value_mismatch():
    # Internally consistent counters (all multiples of 5, and they sum to
    # decoder_calls) but for the WRONG schedule -- must be flagged as a
    # mismatch against the predicted totals, not silently accepted.
    wrong_counters = dict(
        bootstrap_dense_calls=5, bootstrap_observation_calls=5,
        score_refresh_calls=5, decision_refresh_calls=10, held_decision_calls=200,
    )
    raw, problems = raw_phase_counts(wrong_counters)
    _assert_eq(problems, [])
    predicted_totals, _ = predict_request_phases([12, 14, 17, 17, 2], decision_interval=3)
    ok, detail = check_phase_conservation(predicted_totals, raw, [], decoder_calls=62)
    _assert_eq(ok, False)
    assert detail, "mismatch must produce a non-empty detail string"


def test_alignment_conservation_flags_nonzero_on_logical_arm():
    # A logical (non-aligned16) arm must report all-zero alignment
    # counters; if raw counters somehow show nonzero, that is a real
    # mismatch and must be flagged loudly, not swallowed.
    predicted = dict(
        predicted_aligned_score_copies=0, predicted_aligned_pad_bytes=0,
        predicted_aligned_sketch_pads=0,
    )
    bad_counters = dict(aligned_score_copies=3, aligned_pad_bytes=0, aligned_sketch_pads=0)
    ok, detail, raw_values = check_alignment_conservation(predicted, bad_counters, is_aligned_arm=False)
    _assert_eq(ok, False)
    assert "aligned_score_copies" in detail


def test_alignment_conservation_ok_when_matching():
    predicted = dict(
        predicted_aligned_score_copies=45, predicted_aligned_pad_bytes=14750842880,
        predicted_aligned_sketch_pads=110,
    )
    counters = dict(aligned_score_copies=45, aligned_pad_bytes=14750842880, aligned_sketch_pads=110)
    ok, detail, _ = check_alignment_conservation(predicted, counters, is_aligned_arm=True)
    _assert_eq(ok, True, detail)


# ----------------------------------------------------------------------
# Arm classification from a protocol-shaped dict
# ----------------------------------------------------------------------

def test_classify_arms_decision_intervals_and_route_storage():
    protocol = {
        "arm_contracts": {
            "B_boot_aligned16": {
                "kind": "v21_method", "route_storage": "aligned16",
                "parent_v20_arm": "B_A8_matched",
            },
            "M1_boot_aligned16": {
                "kind": "v21_method", "route_storage": "aligned16",
                "parent_v20_arm": "M1_R1_A8_current_output",
            },
            "M3_boot_aligned16": {
                "kind": "v21_method", "route_storage": "aligned16",
                "parent_v20_arm": "M3_R3_A8_current_output",
            },
            "M3_boot_logical": {
                "kind": "v21_method",
                "parent_v20_arm": "M3_R3_A8_current_output",
            },
            "D_native": {"kind": "native"},
            "T_scope": {"kind": "v20_legacy", "parent_v20_arm": "T_scope"},
        }
    }
    info = classify_arms(protocol)
    _assert_eq(info["B_boot_aligned16"]["decision_interval"], None)
    _assert_eq(info["M1_boot_aligned16"]["decision_interval"], 1)
    _assert_eq(info["M3_boot_aligned16"]["decision_interval"], 3)
    _assert_eq(info["M3_boot_aligned16"]["route_storage"], "aligned16")
    _assert_eq(info["M3_boot_logical"]["decision_interval"], 3)
    _assert_eq(info["M3_boot_logical"]["route_storage"], None)
    _assert_eq(info["D_native"]["kind"], "native")
    _assert_eq(info["D_native"]["decision_interval"], None)
    _assert_eq(info["T_scope"]["kind"], "v20_legacy")


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------

def _all_tests():
    mod = sys.modules[__name__]
    for name in dir(mod):
        if name.startswith("test_"):
            yield name, getattr(mod, name)


if __name__ == "__main__":
    failures = []
    count = 0
    for name, fn in _all_tests():
        count += 1
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")

    print(f"\n{count - len(failures)}/{count} tests passed")
    if failures:
        sys.exit(1)
    sys.exit(0)
