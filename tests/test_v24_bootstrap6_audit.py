"""Synthetic tests for scripts/v24_bootstrap6_audit.py.

No pytest dependency: every test_* function is a plain function that raises
AssertionError on failure. Run directly:

    PYTHONPATH=. python tests/test_v24_bootstrap6_audit.py
"""

import csv
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.v24_bootstrap6_audit import (  # noqa: E402
    Segment,
    run_audit,
    five_phase_from_counters,
    compute_estimand,
    sha256_file,
    geomean,
)

CSV_FIELDS = [
    "dataset", "id", "seed", "arm", "block", "cell_id", "host", "gpu_uuid",
    "first_status", "quality_eligible", "scored_first", "strict_correct",
    "warm_status", "accepted_warm_request_wall_s", "decoder_calls", "canvases",
]


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in CSV_FIELDS}
            w.writerow(full)


def _write_ledger(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def make_protocol_file(path, cells):
    schedule = []
    block_assignments = {}
    for c in cells:
        for role, idx, rep in (("attempt0", 0, 0), ("warm", 1, 1)):
            schedule.append({
                "arm": c["arm"], "block": c["block"], "cell_id": c["cell_id"],
                "dataset": c["dataset"], "gpu_uuid": c["gpu_uuid"], "host": c["host"],
                "id": c["id"], "index": idx, "repeat": rep, "role": role, "seed": c["seed"],
            })
        block_assignments[str(c["block"])] = {
            "dataset": c["dataset"], "gpu_uuid": c["gpu_uuid"], "host": c["host"],
            "id": c["id"], "seed": c["seed"],
        }
    protocol = {"protocol_id": "test_protocol", "schedule": schedule, "block_assignments": block_assignments}
    _write_json(path, protocol)


def make_binding_file(path, protocol_path, source_commit="deadbeef"):
    sha = sha256_file(protocol_path)
    _write_json(path, {"panel_protocol_sha256": sha, "source_commits": {"host-x": source_commit}})


def placeholder_row(c):
    return {
        "dataset": c["dataset"], "id": c["id"], "seed": str(c["seed"]), "arm": c["arm"],
        "block": c["block"], "cell_id": c["cell_id"], "host": c["host"], "gpu_uuid": c["gpu_uuid"],
        "first_status": "missing", "quality_eligible": "True", "scored_first": "False",
        "strict_correct": "", "warm_status": "missing", "accepted_warm_request_wall_s": "",
        "decoder_calls": "", "canvases": "",
    }


def executed_row(c, decoder_calls, strict_correct="True", warm_status="accepted", wall=10.0,
                  canvases=5, first_status="success", quality_eligible="True", scored_first="True",
                  host=None, gpu_uuid=None):
    return {
        "dataset": c["dataset"], "id": c["id"], "seed": str(c["seed"]), "arm": c["arm"],
        "block": c["block"], "cell_id": c["cell_id"],
        "host": host if host is not None else c["host"],
        "gpu_uuid": gpu_uuid if gpu_uuid is not None else c["gpu_uuid"],
        "first_status": first_status, "quality_eligible": quality_eligible, "scored_first": scored_first,
        "strict_correct": strict_correct, "warm_status": warm_status,
        "accepted_warm_request_wall_s": "" if wall is None else str(wall),
        "decoder_calls": "" if decoder_calls is None else str(decoder_calls),
        "canvases": "" if canvases is None else str(canvases),
    }


def run_event(c, role, decoder_calls, counters=None, host=None, gpu_uuid=None, index=0):
    return {
        "event": "run", "cell_id": c["cell_id"], "role": role,
        "host": host if host is not None else c["host"],
        "gpu_uuid": gpu_uuid if gpu_uuid is not None else c["gpu_uuid"],
        "decoder_calls": decoder_calls, "counters": counters, "arm": c["arm"],
        "dataset": c["dataset"], "id": c["id"], "seed": c["seed"], "block": c["block"],
        "execution_key": "%s:%s:%d" % (c["cell_id"], role, index), "ok": True, "error": None,
    }


def build_segment(tmpdir, seg_name, protocol_path, cells, scored_rows, ledger1_events, ledger2_events=()):
    scored_csv = os.path.join(tmpdir, "%s_scored.csv" % seg_name)
    ledger1 = os.path.join(tmpdir, "%s_ledger1.jsonl" % seg_name)
    ledger2 = os.path.join(tmpdir, "%s_ledger2.jsonl" % seg_name)
    binding_path = os.path.join(tmpdir, "%s_binding.json" % seg_name)
    _write_csv(scored_csv, scored_rows)
    _write_ledger(ledger1, ledger1_events)
    _write_ledger(ledger2, ledger2_events)
    make_binding_file(binding_path, protocol_path)
    return Segment(seg_name, scored_csv, ledger1, ledger2, protocol_path, binding_path)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_duplicate_executed_cells_fail():
    tmpdir = tempfile.mkdtemp()
    try:
        c = {"dataset": "ds", "id": "ds/1", "seed": 1, "arm": "D_native", "block": 0,
             "cell_id": "c1", "host": "h1", "gpu_uuid": "g1"}
        protocol_path = os.path.join(tmpdir, "protocol.json")
        make_protocol_file(protocol_path, [c])

        a0 = run_event(c, "attempt0", 100)
        wm = run_event(c, "warm", 100, index=1)

        segA = build_segment(tmpdir, "A", protocol_path, [c],
                              [executed_row(c, 100)], [a0, wm])
        segB = build_segment(tmpdir, "B", protocol_path, [c],
                              [executed_row(c, 100)], [a0, wm])  # identical duplicate execution

        result = run_audit([segA, segB])
        assert result["union_verdict"] == "FAIL", result
        assert result["n_invalid"] == 1
        cell = result["cells"][0]
        assert cell["class"] == "invalid"
        assert any("duplicate_execution_identical" in r for r in cell["reasons"]), cell["reasons"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_conflicting_duplicates_fail():
    tmpdir = tempfile.mkdtemp()
    try:
        c = {"dataset": "ds", "id": "ds/1", "seed": 1, "arm": "D_native", "block": 0,
             "cell_id": "c1", "host": "h1", "gpu_uuid": "g1"}
        protocol_path = os.path.join(tmpdir, "protocol.json")
        make_protocol_file(protocol_path, [c])

        segA = build_segment(tmpdir, "A", protocol_path, [c],
                              [executed_row(c, 100)],
                              [run_event(c, "attempt0", 100), run_event(c, "warm", 100, index=1)])
        segB = build_segment(tmpdir, "B", protocol_path, [c],
                              [executed_row(c, 150)],  # conflicting decoder_calls
                              [run_event(c, "attempt0", 150), run_event(c, "warm", 150, index=1)])

        result = run_audit([segA, segB])
        assert result["union_verdict"] == "FAIL", result
        cell = result["cells"][0]
        assert cell["class"] == "invalid"
        assert any("duplicate_execution_conflicting" in r for r in cell["reasons"]), cell["reasons"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_placeholder_plus_valid_resolves():
    tmpdir = tempfile.mkdtemp()
    try:
        c = {"dataset": "ds", "id": "ds/1", "seed": 1, "arm": "D_native", "block": 0,
             "cell_id": "c1", "host": "h1", "gpu_uuid": "g1"}
        protocol_path = os.path.join(tmpdir, "protocol.json")
        make_protocol_file(protocol_path, [c])

        segA = build_segment(tmpdir, "A", protocol_path, [c],
                              [executed_row(c, 100, strict_correct="True")],
                              [run_event(c, "attempt0", 100), run_event(c, "warm", 100, index=1)])
        segB = build_segment(tmpdir, "B", protocol_path, [c],
                              [placeholder_row(c)], [])

        result = run_audit([segA, segB])
        assert result["union_verdict"] == "PASS", result["union_reasons"]
        assert result["n_invalid"] == 0
        cell = result["cells"][0]
        assert cell["class"] == "scored_correct", cell
        assert cell["executed_segment"] == "A"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_host_mismatch_fails():
    tmpdir = tempfile.mkdtemp()
    try:
        c = {"dataset": "ds", "id": "ds/1", "seed": 1, "arm": "D_native", "block": 0,
             "cell_id": "c1", "host": "h1", "gpu_uuid": "g1"}
        protocol_path = os.path.join(tmpdir, "protocol.json")
        make_protocol_file(protocol_path, [c])  # registry says host h1

        # CSV row (and its ledger) claim host h2 -- mismatched against the protocol.
        bad_row = executed_row(c, 100, host="h2", gpu_uuid="g2")
        segA = build_segment(tmpdir, "A", protocol_path, [c],
                              [bad_row],
                              [run_event(c, "attempt0", 100, host="h2", gpu_uuid="g2"),
                               run_event(c, "warm", 100, host="h2", gpu_uuid="g2", index=1)])
        segB = build_segment(tmpdir, "B", protocol_path, [c], [placeholder_row(c)], [])

        result = run_audit([segA, segB])
        assert result["union_verdict"] == "FAIL", result
        cell = result["cells"][0]
        assert cell["class"] == "invalid"
        assert any("host_gpu_mismatch" in r for r in cell["reasons"]), cell["reasons"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_missing_and_unscored_not_counted_wrong():
    tmpdir = tempfile.mkdtemp()
    try:
        c_missing = {"dataset": "aime26", "id": "aime26/1", "seed": 1, "arm": "D_native", "block": 0,
                     "cell_id": "cA", "host": "h1", "gpu_uuid": "g1"}
        c_unscored = {"dataset": "aime26", "id": "aime26/2", "seed": 1, "arm": "D_native", "block": 1,
                      "cell_id": "cB", "host": "h1", "gpu_uuid": "g1"}
        protocol_path = os.path.join(tmpdir, "protocol.json")
        make_protocol_file(protocol_path, [c_missing, c_unscored])

        unscored_row = executed_row(c_unscored, 100, quality_eligible="False", strict_correct="")
        segA = build_segment(tmpdir, "A", protocol_path, [c_missing, c_unscored],
                              [placeholder_row(c_missing), unscored_row],
                              [run_event(c_unscored, "attempt0", 100), run_event(c_unscored, "warm", 100, index=1)])
        segB = build_segment(tmpdir, "B", protocol_path, [c_missing, c_unscored],
                              [placeholder_row(c_missing), placeholder_row(c_unscored)], [])

        result = run_audit([segA, segB])
        assert result["union_verdict"] == "PASS", result["union_reasons"]
        classes = {c["cell_id"]: c["class"] for c in result["cells"]}
        assert classes["cA"] == "missing", classes
        assert classes["cB"] == "unscored", classes

        summary = result["per_dataset_arm_summary"]["aime26"]["D_native"]
        assert summary["n_wrong"] == 0
        assert summary["n_correct"] == 0
        assert summary["n_missing"] == 1
        assert summary["n_unscored"] == 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_warm_rejection_excluded_from_timing():
    # Build cells_by_key / ledger_index by hand -- unit-tests compute_estimand
    # directly rather than the full audit_cell pipeline.
    cells_by_key = {}
    for seed, warm_status, wall, strict in (
        (101, "accepted", 10.0, "True"),
        (202, "rejected", None, "False"),  # scored, but warm rejected -> timing-excluded
    ):
        cells_by_key[("ds", "q1", str(seed), "M3_native_bootstrap2_observe1")] = {
            "class": "scored_correct" if strict == "True" else "scored_wrong",
            "host": "h1", "gpu_uuid": "g1", "warm_status": warm_status,
            "accepted_warm_request_wall_s": wall, "decoder_calls": 100, "cell_id": "cx_%s" % seed,
            "strict_correct": strict,
        }
        cells_by_key[("ds", "q1", str(seed), "D_native")] = {
            "class": "scored_correct", "host": "h1", "gpu_uuid": "g1", "warm_status": "accepted",
            "accepted_warm_request_wall_s": 20.0, "decoder_calls": 200, "cell_id": "cy_%s" % seed,
            "strict_correct": "True",
        }

    est = compute_estimand("ds", "M3_native_bootstrap2_observe1", "D_native", cells_by_key, ledger_index={})
    assert est["n_valid_pairs"] == 2, est
    assert est["n_timing_pairs"] == 1, est  # only seed 101's pair is timing-eligible
    assert est["geo_W_ratio"] is not None


def test_geo_identity_holds():
    cells_by_key = {}
    seeds_wn = [(1, 10.0, 100, 15.0, 90), (2, 22.0, 130, 9.0, 210), (3, 5.5, 40, 30.0, 400)]
    for seed, wx, nx, wy, ny in seeds_wn:
        cells_by_key[("ds", "q", str(seed), "ARM_X")] = {
            "class": "scored_correct", "host": "h1", "gpu_uuid": "g1", "warm_status": "accepted",
            "accepted_warm_request_wall_s": wx, "decoder_calls": nx, "cell_id": "x_%s" % seed,
            "strict_correct": "True",
        }
        cells_by_key[("ds", "q", str(seed), "ARM_Y")] = {
            "class": "scored_wrong", "host": "h1", "gpu_uuid": "g1", "warm_status": "accepted",
            "accepted_warm_request_wall_s": wy, "decoder_calls": ny, "cell_id": "y_%s" % seed,
            "strict_correct": "False",
        }

    est = compute_estimand("ds", "ARM_X", "ARM_Y", cells_by_key, ledger_index={})
    assert est["n_timing_pairs"] == 3
    assert est["geo_identity_ok"] is True, est
    assert est["geo_identity_abs_diff"] < 1e-9
    # sanity: also check the raw geomean identity directly
    w = [wx / wy for (_, wx, _, wy, _) in seeds_wn]
    n = [nx / ny for (_, wx, nx, wy, ny) in seeds_wn]
    wn = [(wx / nx) / (wy / ny) for (_, wx, nx, wy, ny) in seeds_wn]
    assert abs(geomean(w) - geomean(n) * geomean(wn)) < 1e-9


def test_conservation_check_detects_violation():
    counters_ok = {
        "bootstrap_dense_calls": 60, "bootstrap_observation_calls": 60,
        "score_refresh_calls": 105, "decision_refresh_calls": 370,
        "held_decision_calls": 660,
    }
    fp_ok = five_phase_from_counters(counters_ok, decoder_calls=230)
    assert fp_ok["conservation_ok"] is True, fp_ok
    assert fp_ok["B0"] == 60 and fp_ok["BO"] == 60 and fp_ok["A"] == 105 and fp_ok["D"] == 265 and fp_ok["H"] == 660
    assert fp_ok["model_calls"] == 230.0

    # violate conservation by using the wrong decoder_calls (as if a run's
    # counters and decoder_calls came from different executions)
    fp_bad = five_phase_from_counters(counters_ok, decoder_calls=231)
    assert fp_bad["conservation_ok"] is False, fp_bad

    # absent raw field -> never infer zero, mark fields null and skip the check
    counters_missing = dict(counters_ok)
    del counters_missing["held_decision_calls"]
    fp_missing = five_phase_from_counters(counters_missing, decoder_calls=230)
    assert fp_missing["H"] is None
    assert fp_missing["conservation_ok"] is None
    assert fp_missing["five_phase_fields_complete"] is False


TESTS = [
    test_duplicate_executed_cells_fail,
    test_conflicting_duplicates_fail,
    test_placeholder_plus_valid_resolves,
    test_host_mismatch_fails,
    test_missing_and_unscored_not_counted_wrong,
    test_warm_rejection_excluded_from_timing,
    test_geo_identity_holds,
    test_conservation_check_detects_violation,
]


if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t()
            print("PASS %s" % t.__name__)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print("FAIL %s: %r" % (t.__name__, e))
            import traceback
            traceback.print_exc()
    print("%d/%d tests passed" % (len(TESTS) - failures, len(TESTS)))
    sys.exit(1 if failures else 0)
