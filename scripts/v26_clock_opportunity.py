"""v26: clock-opportunity analysis over recorded canvas lengths.

Reconstructs the bootstrap-path reference clock (B0/BO/A/D/H, GLOBAL layer
schedule) from first principles, per the method's real per-canvas schedule,
and applies it (a) to the actual recorded canvas lengths of the bootstrap
method arms, where it is validated exactly against the raw counters, and
(b) as a stated hypothetical to the native D_native / T_scope canvas
lengths and to counterfactual (A, R) policies never actually run.

This produces no new generation and no new claim about quality or a new
E2E measurement; it only recomputes, from canvas lengths already on disk,
how many GLOBAL layer-calls of each kind (B0/BO/A/D/H) a given clock
policy would spend.

Reference clock (must match the real schedule exactly):
  - call 0            -> B0  (native, always, if n >= 1)
  - call 1            -> BO  (native output + observation), only if n >= 2
  - later anchors A    at calls  1 + k*A_period  for k >= 1,  call < n
      (this excludes k=0, i.e. call 1 itself, which is BO not A)
  - decision age resets to 0 at any anchor (BO at call 1, or a later A)
  - a non-anchor call c is a decision refresh D when
      (c - last_reset) >= R
    and then last_reset := c; otherwise it is held H.
  - "matched" B uses R == A_period, which makes D identically 0 (every
    call that would qualify for D is itself an anchor).

Cross-check (must hold for every canvas with n >= 2):
    A == max(0, floor((n - 2) / A_period))

Validated against the recorded raw counters as follows. The raw per-record
counters (GLOBAL layer-calls, 5 active layers per decoder call) are:
    bootstrap_dense_calls        == 5 * B0
    bootstrap_observation_calls  == 5 * BO
    score_refresh_calls          == 5 * A
    decision_refresh_calls       == 5 * (A + D)   (an anchor call also
                                                    triggers a decision
                                                    refresh; see
                                                    v24_bootstrap6_audit.py:
                                                    d = decision_refresh - a)
    held_decision_calls          == 5 * H
so D = decision_refresh_calls/5 - score_refresh_calls/5.

This matches the decomposition documented in
results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6/v24_errata_and_audit.md
("Conservation B0 + BO + A + D + H = 5 x decoder calls") and the CPU clock
unit tests in tests/test_v23_bootstrap_clock.py (ScoreCache with origin=1:
the bootstrap path's internal clock treats call 1 as its own anchor point,
which is exactly why the *later* anchor formula above starts at k=1, not
k=0).

Usage:
    python scripts/v26_clock_opportunity.py \
        --index results/m3_numeric_trajectory_bridge_20260927/generation_records_v21_v25b/index.csv \
        --records-dir results/m3_numeric_trajectory_bridge_20260927/generation_records_v21_v25b \
        --out-csv results/m3_output_numerics_.../aggregate.csv \
        --out-json results/m3_output_numerics_.../report.json

stdlib only. Refuses to overwrite existing output files.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

# ----------------------------------------------------------------------
# Reference clock
# ----------------------------------------------------------------------

RUN_PREFIXES = ("v23_", "v24_", "v25_", "v25b_")

# Arm-name substrings that mark a "method arm" record in scope for the
# per-record clock reconstruction (per task spec: "arms whose name
# contains M1/M3/B or 'boot'"). Checked as case-sensitive substrings.
METHOD_ARM_TAGS = ("M1", "M3", "B", "boot")

NATIVE_ARMS = ("D_native", "T_scope")

# Policies to evaluate on every qualifying record.
A_PERIODS = (8, 16)
R_VALUES = (1, 3, 6)
REFERENCE_POLICY = ("A8_R3", 8, 3, False)  # label, A_period, R, matched


def policy_grid():
    """Return the ordered list of (label, A_period, R, matched) policies."""
    grid = []
    for a in A_PERIODS:
        for r in R_VALUES:
            grid.append((f"A{a}_R{r}", a, r, False))
        grid.append((f"A{a}_matchedB", a, a, True))
    return grid


def classify_canvas(n, a_period, r):
    """Classify the n decoder calls (0-indexed 0..n-1) of one canvas under
    the reference clock with score-anchor period a_period and decision
    interval r (r == a_period for the matched-B policy, which forces D=0).

    Returns a dict with integer counts B0, BO, A, D, H that sum to n
    (for n >= 0), plus the sorted list of later-anchor call indices (for
    the floor cross-check) and the list of D call indices.
    """
    if n <= 0:
        return dict(B0=0, BO=0, A=0, D=0, H=0, anchor_calls=[], d_calls=[])

    b0 = 1
    if n < 2:
        return dict(B0=b0, BO=0, A=0, D=0, H=0, anchor_calls=[], d_calls=[])

    bo = 1
    anchor_calls = []
    k = 1
    while True:
        call = 1 + k * a_period
        if call >= n:
            break
        anchor_calls.append(call)
        k += 1
    anchor_set = set(anchor_calls)

    last_reset = 1  # call 1 (BO) resets the decision age
    d = 0
    h = 0
    d_calls = []
    for call in range(2, n):
        if call in anchor_set:
            last_reset = call
            continue
        age = call - last_reset
        if age >= r:
            d += 1
            d_calls.append(call)
            last_reset = call
        else:
            h += 1

    out = dict(B0=b0, BO=bo, A=len(anchor_calls), D=d, H=h,
               anchor_calls=anchor_calls, d_calls=d_calls)
    return out


def floor_crosscheck(n, a_period, predicted_a):
    """Assert-style cross-check: for n >= 2, the later-anchor count must
    equal max(0, floor((n-2)/a_period)). Returns (expected, ok)."""
    if n < 2:
        expected = 0
    else:
        expected = max(0, (n - 2) // a_period)
    return expected, expected == predicted_a


# ----------------------------------------------------------------------
# Index / record loading
# ----------------------------------------------------------------------

def load_index(index_path):
    with open(index_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def is_method_row(row):
    if not row["run"].startswith(RUN_PREFIXES):
        return False
    arm = row["arm"]
    return any(tag in arm for tag in METHOD_ARM_TAGS)


def is_native_trajectory_row(row):
    if not row["run"].startswith(RUN_PREFIXES):
        return False
    return row["arm"] in NATIVE_ARMS


def load_record(records_dir, rel_file):
    path = os.path.join(records_dir, rel_file)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canvas_lengths(record):
    pc = record.get("per_canvas") or []
    return [int(c["decoder_calls"]) for c in pc]


# ----------------------------------------------------------------------
# Validation against recorded counters
# ----------------------------------------------------------------------

def recorded_phase_counts(record):
    """Extract (B0, BO, A, D, H) in decoder-call units from a record's raw
    counters, or None if the record has no bootstrap counters (e.g. the
    plain incumbent arm, or D_native/T_scope, which do not run this
    clock in reality)."""
    counters = record.get("counters") or {}
    b0 = counters.get("bootstrap_dense_calls")
    bo = counters.get("bootstrap_observation_calls")
    a = counters.get("score_refresh_calls")
    dr = counters.get("decision_refresh_calls")
    h = counters.get("held_decision_calls")
    if b0 is None or bo is None or a is None or dr is None or h is None:
        return None
    if b0 == 0 and bo == 0:
        # No real bootstrap instrumentation on this record (e.g. the
        # plain M3_R3_A8_incumbent arm, whose call 0 is itself an anchor
        # under a different, origin-0 clock -- out of scope for this
        # reference clock's validation).
        return None
    if any(x % 5 != 0 for x in (b0, bo, a, dr, h)):
        return None
    d = (dr - a) // 5 if (dr - a) % 5 == 0 else None
    if d is None:
        return None
    return dict(B0=b0 // 5, BO=bo // 5, A=a // 5, D=d, H=h // 5)


def record_decision_interval_and_period(record):
    counters = record.get("counters") or {}
    r = counters.get("decision_interval")
    a_period = counters.get("score_period")
    return r, a_period


def validate_record(record, arm):
    """Run the reference clock at the record's own (A_period, R) and
    compare the summed per-canvas classification to the recorded
    counters. Returns None if the record has no bootstrap counters to
    validate against; otherwise a dict with predicted, recorded, match.
    """
    recorded = recorded_phase_counts(record)
    if recorded is None:
        return None
    r, a_period = record_decision_interval_and_period(record)
    if r is None or a_period is None:
        return None
    lengths = canvas_lengths(record)
    predicted = dict(B0=0, BO=0, A=0, D=0, H=0)
    floor_mismatches = []
    for n in lengths:
        c = classify_canvas(n, a_period, r)
        for k in ("B0", "BO", "A", "D", "H"):
            predicted[k] += c[k]
        expected_a, ok = floor_crosscheck(n, a_period, c["A"])
        if not ok:
            floor_mismatches.append(dict(n=n, predicted_a=c["A"], expected_a=expected_a))
    match = predicted == recorded
    return dict(
        arm=arm, a_period=a_period, r=r,
        predicted=predicted, recorded=recorded, match=match,
        canvases=lengths, floor_mismatches=floor_mismatches,
    )


# ----------------------------------------------------------------------
# Main aggregation
# ----------------------------------------------------------------------

def aggregate(index_rows, records_dir):
    policies = policy_grid()

    validations = []
    # aggregate[(dataset, source_arm, policy_label)] -> counters
    agg = OrderedDict()
    # bo-waste tracking: (dataset, arm) -> [n_bo, n_bo_wasted]
    bo_waste = defaultdict(lambda: [0, 0])

    method_rows = [r for r in index_rows if r["role"] == "first" and is_method_row(r)]
    native_rows = [r for r in index_rows if r["role"] == "first" and is_native_trajectory_row(r)]

    def bump(dataset, source_arm, policy_label, a_period, r, matched, lengths):
        key = (dataset, source_arm, policy_label)
        if key not in agg:
            agg[key] = dict(
                dataset=dataset, source_arm=source_arm, policy=policy_label,
                a_period=a_period, r=r, matched=matched,
                canvases=0, canvases_reach_call9=0, canvases_reach_call17=0,
                total_B0=0, total_BO=0, total_A=0, total_D=0, total_H=0,
            )
        row = agg[key]
        for n in lengths:
            c = classify_canvas(n, a_period, r)
            row["canvases"] += 1
            if n > 9:
                row["canvases_reach_call9"] += 1
            if n > 17:
                row["canvases_reach_call17"] += 1
            row["total_B0"] += c["B0"]
            row["total_BO"] += c["BO"]
            row["total_A"] += c["A"]
            row["total_D"] += c["D"]
            row["total_H"] += c["H"]

    # ---- method arms: validate + populate policy grid ----
    for row in method_rows:
        arm = row["arm"]
        dataset = row["dataset"]
        record = load_record(records_dir, row["file"])
        lengths = canvas_lengths(record)

        v = validate_record(record, arm)
        if v is not None:
            v["dataset"] = dataset
            v["cell_id"] = row["cell_id"]
            v["run"] = row["run"]
            v["file"] = row["file"]
            validations.append(v)

            # BO-waste bookkeeping on the *real* recorded canvas lengths
            key = (dataset, arm)
            for n in lengths:
                if n >= 2:
                    bo_waste[key][0] += 1
                    if n == 2:
                        bo_waste[key][1] += 1

        for label, a_period, r, matched in policies:
            bump(dataset, arm, label, a_period, r, matched, lengths)

    # ---- native trajectory (hypothetical) ----
    for row in native_rows:
        arm = row["arm"]
        dataset = row["dataset"]
        record = load_record(records_dir, row["file"])
        lengths = canvas_lengths(record)
        for label, a_period, r, matched in policies:
            bump(dataset, "native_trajectory:" + arm, label, a_period, r, matched, lengths)

    # ---- deltas vs the A8_R3 reference, within (dataset, source_arm) ----
    ref_label = REFERENCE_POLICY[0]
    ref_lookup = {}
    for (dataset, source_arm, policy_label), row in agg.items():
        if policy_label == ref_label:
            ref_lookup[(dataset, source_arm)] = row

    out_rows = []
    for (dataset, source_arm, policy_label), row in agg.items():
        ref = ref_lookup.get((dataset, source_arm))
        is_recorded = _is_recorded_policy(source_arm, policy_label)
        r2 = dict(row)
        r2["is_recorded_actual"] = is_recorded
        r2["label"] = ("recorded (matches actual instrumented schedule)" if is_recorded
                        else "estimate on recorded canvas lengths; trajectories would change")
        if ref is not None and ref is not row:
            r2["delta_A_vs_A8_R3"] = row["total_A"] - ref["total_A"]
            r2["delta_D_vs_A8_R3"] = row["total_D"] - ref["total_D"]
            r2["delta_H_vs_A8_R3"] = row["total_H"] - ref["total_H"]
            r2["delta_BO_vs_A8_R3"] = row["total_BO"] - ref["total_BO"]
        else:
            r2["delta_A_vs_A8_R3"] = 0
            r2["delta_D_vs_A8_R3"] = 0
            r2["delta_H_vs_A8_R3"] = 0
            r2["delta_BO_vs_A8_R3"] = 0
        out_rows.append(r2)

    bo_waste_rows = []
    for (dataset, arm), (n_bo, n_wasted) in sorted(bo_waste.items()):
        frac = (n_wasted / n_bo) if n_bo else None
        bo_waste_rows.append(dict(
            dataset=dataset, arm=arm,
            canvases_with_BO=n_bo,
            canvases_BO_wasted_n_eq_2=n_wasted,
            fraction_wasted=frac,
        ))

    return out_rows, validations, bo_waste_rows


def _is_recorded_policy(source_arm, policy_label):
    """A policy row is the 'recorded' (actually-run) schedule only for a
    real bootstrap method arm (not native_trajectory, not the plain
    M3_R3_A8_incumbent, which structurally lacks B0/BO), and only at
    A_period=8 with the arm's own real decision interval (R=3 for M3
    variants, R=1 for M1 variants, matched R=A_period for B variants).
    No A16 policy was ever actually run in this data.
    """
    if source_arm.startswith("native_trajectory:"):
        return False
    if source_arm == "M3_R3_A8_incumbent":
        return False
    if "M1" in source_arm:
        return policy_label == "A8_R1"
    if "M3" in source_arm:
        return policy_label == "A8_R3"
    if source_arm.startswith("B_"):
        return policy_label == "A8_matchedB"
    return False


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "source_arm", "policy", "a_period", "r", "matched",
    "canvases", "canvases_reach_call9", "canvases_reach_call17",
    "total_B0", "total_BO", "total_A", "total_D", "total_H",
    "delta_A_vs_A8_R3", "delta_D_vs_A8_R3", "delta_H_vs_A8_R3", "delta_BO_vs_A8_R3",
    "is_recorded_actual", "label",
]


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in CSV_FIELDS})


def write_json(path, out_rows, validations, bo_waste_rows):
    all_match = all(v["match"] for v in validations) if validations else None
    mismatches = [v for v in validations if not v["match"]]
    floor_mismatches = []
    for v in validations:
        if v["floor_mismatches"]:
            floor_mismatches.append(dict(
                arm=v["arm"], dataset=v["dataset"], cell_id=v["cell_id"],
                mismatches=v["floor_mismatches"],
            ))
    payload = dict(
        schema="v26_clock_opportunity/1",
        reference_clock=dict(
            description=(
                "call0=B0 (native); call1=BO (native output+observation, "
                "n>=2 only); later anchors A at calls 1+k*A_period, k>=1, "
                "call<n; decision age resets at any anchor; D at a "
                "non-anchor call when (call-last_reset)>=R, else H; "
                "matched-B uses R=A_period which forces D=0."
            ),
            floor_crosscheck="A == max(0, floor((n-2)/A_period)) for n>=2",
        ),
        validation=dict(
            records_checked=len(validations),
            all_match=all_match,
            mismatch_count=len(mismatches),
            mismatches=[
                dict(arm=v["arm"], dataset=v["dataset"], cell_id=v["cell_id"],
                     run=v["run"], file=v["file"],
                     a_period=v["a_period"], r=v["r"],
                     predicted=v["predicted"], recorded=v["recorded"])
                for v in mismatches
            ],
            floor_crosscheck_failures=floor_mismatches,
        ),
        aggregate=out_rows,
        bo_wasted=bo_waste_rows,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    for out_path in (args.out_csv, args.out_json):
        if os.path.exists(out_path):
            print(f"refusing to overwrite existing output: {out_path}", file=sys.stderr)
            return 2

    index_rows = load_index(args.index)
    out_rows, validations, bo_waste_rows = aggregate(index_rows, args.records_dir)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    write_csv(args.out_csv, out_rows)
    write_json(args.out_json, out_rows, validations, bo_waste_rows)

    all_match = all(v["match"] for v in validations) if validations else None
    print(f"records validated: {len(validations)}, all_match={all_match}")
    mismatches = [v for v in validations if not v["match"]]
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}", file=sys.stderr)
        for v in mismatches[:10]:
            print(f"  {v['dataset']}/{v['arm']}/{v['cell_id']}: predicted={v['predicted']} recorded={v['recorded']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
