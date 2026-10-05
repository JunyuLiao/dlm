"""v25b_alignment_exposure.py -- aligned16 route-storage exposure reducer.

Reconciles the v21 bootstrap/decision phase schedule (B0/BO/A/D/H) and the
aligned16 route-storage copy/pad overhead against raw per-request ledger
counters for the v25 "aligned6" pilot, cross-checked against a strong
correctness-gated baseline: the raw layer-call counters recorded by the
harness itself (not a re-derivation from a naive model).  Produces one row
per (dataset, id, seed, arm, role) plus a paired M3_boot_aligned16 vs
M3_boot_logical comparison keyed on completion-token identity.

Facts this module encodes (verified against real pilot data before writing
any prediction code -- see the accompanying report):

  - GLOBAL layers = 5 per decoder call (LAYERS_PER_DECODER_CALL).
  - Per-canvas call schedule (0-based call index i, n = calls issued for
    that canvas):
      i == 0            -> B0 (bootstrap dense)
      i == 1 (n >= 2)    -> BO (bootstrap observation)
      i in {9,17,25,...} -> A  (score refresh; fixed schedule, step 8,
                                 starting at 9, independent of arm)
      else if the arm has a decision interval and i - last_decision ==
        that interval -> D (decision refresh); an A call always resets
        the "last decision" clock (this is `score_clock_origin`, which
        starts at call index 1, the BO call).
      else               -> H (held / no refresh)
    M3 (R3) uses decision_interval=3, M1 (R1) uses decision_interval=1,
    B (bootstrap-only, parent arm has no "_R<n>_" token) never issues D.
    D_native / T_scope arms are not v21_method arms and have no phase
    schedule at all.

  - For a request, canvas c (0-based) has real key count
      K_c = prompt_token_count + 256*(c+1)
    because every non-final canvas commits exactly 256 tokens.  Since 256
    is a multiple of 16 (and of every smaller power of two up to 16),
    K_c mod 16 == prompt_token_count mod 16 for *every* canvas of a given
    request -- so a request is either aligned on every canvas or
    unaligned on every canvas; there is no per-canvas mixing.
  - pitch_c = ceil(K_c / 16) * 16.  aligned16 only pays a copy when
    K_c % 16 != 0.
  - One stored score buffer = 16 (H) * 256 (Q) * 4 (fp32 bytes) * pitch
    = 16384 * pitch bytes.

  - Predicted aligned16 overhead (attributed only when the request is
    unaligned, i.e. prompt_token_count % 16 != 0):
      aligned_score_copies = 5 * sum_c (BO_c + A_c)
      aligned_pad_bytes    = sum_c [ 5 * (BO_c + A_c) * 16384 * pitch_c ]
      aligned_sketch_pads  = 5 * sum_c (BO_c + A_c + D_c)
    For logical-route-storage arms (and for aligned requests) these must
    be exactly 0.

All three formulas above were checked by hand against a real
M3_boot_aligned16 ledger row (longbench_v2/66ed2c87821e116aacb1f149,
seed 101) and matched the recorded counters exactly (45 / 14750842880 /
110) before this module was written; see the CLI's conservation output
for the same check run over every row in the pilot.

Stdlib only. Refuses to overwrite existing --out-csv / --out-json paths.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Constants (see module docstring for provenance).
# ----------------------------------------------------------------------

LAYERS_PER_DECODER_CALL = 5
CANVAS_COMMIT_TOKENS = 256
PITCH_ALIGN = 16
SCORE_BUFFER_BYTES_PER_PITCH_UNIT = 16 * 256 * 4  # H * Q * fp32_bytes = 16384
A_SCHEDULE_START = 9
A_SCHEDULE_STEP = 8
DECISION_CLOCK_ORIGIN_CALL = 1  # call index of the BO event

MANIFEST_FILENAMES = {
    "longbench_v2": "longbench_v2_generation_manifest.json",
    "aime26": "aime26_generation_manifest.json",
    "ruler4k": "ruler4k_generation_manifest.json",
}

CSV_FIELDS = [
    "dataset", "id", "seed", "arm", "role", "kind", "route_storage",
    "host", "gpu_uuid",
    "decoder_calls", "canvases",
    "prompt_token_count", "k_mod16", "kdiv", "odd_k_path",
    "predicted_B0", "predicted_BO", "predicted_A", "predicted_D", "predicted_H",
    "raw_B0", "raw_BO", "raw_A", "raw_D", "raw_H",
    "phase_conservation_ok", "phase_mismatch_detail",
    "predicted_aligned_score_copies", "raw_aligned_score_copies",
    "predicted_aligned_pad_bytes", "raw_aligned_pad_bytes",
    "predicted_aligned_sketch_pads", "raw_aligned_sketch_pads",
    "alignment_conservation_ok", "alignment_mismatch_detail",
    "affected_bo_plus_a_calls", "affected_d_calls", "fraction_calls_affected",
    "completion_token_hash", "api_wall_s",
    "strict_correct", "warm_status", "first_cold_request_wall_s",
    "accepted_warm_request_wall_s",
]


# ----------------------------------------------------------------------
# Phase-schedule prediction.
# ----------------------------------------------------------------------

def a_schedule_calls(n):
    """Set of 0-based call indices < n that fall on the fixed A schedule."""
    calls = set()
    c = A_SCHEDULE_START
    while c < n:
        calls.add(c)
        c += A_SCHEDULE_STEP
    return calls


def predict_canvas_phases(n, decision_interval):
    """Classify every call index of one canvas into B0/BO/A/D/H.

    n: number of decoder calls issued for this canvas.
    decision_interval: None means the arm never issues D (bootstrap-only
        arm); an int means an M-style arm with that decision interval.
    Returns a dict of ints that always sums to n.
    """
    if n <= 0:
        return dict(B0=0, BO=0, A=0, D=0, H=0)
    B0 = 1
    BO = 1 if n >= 2 else 0
    A = D = H = 0
    if n >= 2:
        a_sched = a_schedule_calls(n)
        last_decision = DECISION_CLOCK_ORIGIN_CALL
        for i in range(2, n):
            if i in a_sched:
                A += 1
                last_decision = i
            elif decision_interval is not None and (i - last_decision) == decision_interval:
                D += 1
                last_decision = i
            else:
                H += 1
    return dict(B0=B0, BO=BO, A=A, D=D, H=H)


def predict_request_phases(per_canvas_calls, decision_interval):
    """Sum per-canvas phase predictions across a whole request.

    Returns (totals_dict, list_of_per_canvas_dicts).
    """
    totals = dict(B0=0, BO=0, A=0, D=0, H=0)
    per_canvas = []
    for n in per_canvas_calls:
        ph = predict_canvas_phases(n, decision_interval)
        per_canvas.append(ph)
        for k in totals:
            totals[k] += ph[k]
    return totals, per_canvas


# ----------------------------------------------------------------------
# K / pitch / alignment-exposure prediction.
# ----------------------------------------------------------------------

def canvas_key_count(prompt_token_count, canvas_index):
    """K_c = prompt_token_count + 256*(c+1), c 0-based."""
    return prompt_token_count + CANVAS_COMMIT_TOKENS * (canvas_index + 1)


def pitch_for_k(k):
    return ((k + PITCH_ALIGN - 1) // PITCH_ALIGN) * PITCH_ALIGN


def kdiv_of(prompt_token_count):
    """Largest power of two <=16 dividing K (== dividing prompt_token_count,
    since every canvas commits a multiple of 256 = 2**8 tokens, so K mod 2^j
    for j<=8 is invariant across canvases and equals prompt_token_count mod
    2^j)."""
    for p in (16, 8, 4, 2, 1):
        if prompt_token_count % p == 0:
            return p
    return 1  # unreachable, p=1 always divides


def predict_alignment_exposure(per_canvas_calls, per_canvas_phases, prompt_token_count, is_aligned_arm):
    """Predict aligned16 copy/pad exposure for one request.

    is_aligned_arm: whether this arm's route_storage is 'aligned16'
        (logical arms and native/legacy arms always predict all-zero).
    """
    k_mod16 = prompt_token_count % 16
    unaligned = (k_mod16 != 0)
    copies = 0
    pad_bytes = 0
    sketch_pads = 0
    affected_obs = 0
    affected_dec = 0
    for idx, n in enumerate(per_canvas_calls):
        ph = per_canvas_phases[idx]
        obs_c = ph["BO"] + ph["A"]
        dec_c = ph["D"]
        if unaligned:
            affected_obs += obs_c
            affected_dec += dec_c
            if is_aligned_arm:
                k_c = canvas_key_count(prompt_token_count, idx)
                pitch_c = pitch_for_k(k_c)
                copies += LAYERS_PER_DECODER_CALL * obs_c
                pad_bytes += LAYERS_PER_DECODER_CALL * obs_c * SCORE_BUFFER_BYTES_PER_PITCH_UNIT * pitch_c
                sketch_pads += LAYERS_PER_DECODER_CALL * (obs_c + dec_c)
    return dict(
        k_mod16=k_mod16,
        kdiv=kdiv_of(prompt_token_count),
        predicted_aligned_score_copies=copies,
        predicted_aligned_pad_bytes=pad_bytes,
        predicted_aligned_sketch_pads=sketch_pads,
        affected_bo_plus_a_calls=affected_obs,
        affected_d_calls=affected_dec,
    )


# ----------------------------------------------------------------------
# Raw-counter recovery + conservation checks.
# ----------------------------------------------------------------------

def _safe_div5(x):
    """Return (value, exact) where value = x/5 and exact says whether the
    division was integral. Never raises, never rounds silently."""
    if x % LAYERS_PER_DECODER_CALL == 0:
        return x // LAYERS_PER_DECODER_CALL, True
    return x / LAYERS_PER_DECODER_CALL, False


def raw_phase_counts(counters):
    """Recover raw B0/BO/A/D/H model-call counts from ledger counters.
    Returns (dict_or_None, list_of_mismatch_strings)."""
    problems = []
    required = [
        "bootstrap_dense_calls", "bootstrap_observation_calls",
        "score_refresh_calls", "decision_refresh_calls", "held_decision_calls",
    ]
    for key in required:
        if key not in counters:
            problems.append(f"missing counter field '{key}'")
    if problems:
        return None, problems

    b0, b0_exact = _safe_div5(counters["bootstrap_dense_calls"])
    bo, bo_exact = _safe_div5(counters["bootstrap_observation_calls"])
    a, a_exact = _safe_div5(counters["score_refresh_calls"])
    d_raw = counters["decision_refresh_calls"] - counters["score_refresh_calls"]
    d, d_exact = _safe_div5(d_raw)
    h, h_exact = _safe_div5(counters["held_decision_calls"])

    for name, exact, raw in (
        ("bootstrap_dense_calls", b0_exact, counters["bootstrap_dense_calls"]),
        ("bootstrap_observation_calls", bo_exact, counters["bootstrap_observation_calls"]),
        ("score_refresh_calls", a_exact, counters["score_refresh_calls"]),
        ("decision_refresh_calls-score_refresh_calls", d_exact, d_raw),
        ("held_decision_calls", h_exact, counters["held_decision_calls"]),
    ):
        if not exact:
            problems.append(f"counter '{name}'={raw} not divisible by {LAYERS_PER_DECODER_CALL}")

    return dict(B0=b0, BO=bo, A=a, D=d, H=h), problems


def check_phase_conservation(predicted_totals, raw_counts, raw_problems, decoder_calls):
    """Compare predicted vs raw phase totals and B0+BO+A+D+H==decoder_calls.
    Returns (ok: bool, detail: str)."""
    problems = list(raw_problems)
    if raw_counts is not None:
        for k in ("B0", "BO", "A", "D", "H"):
            if predicted_totals[k] != raw_counts[k]:
                problems.append(f"{k}: predicted={predicted_totals[k]} raw={raw_counts[k]}")
        raw_sum = sum(raw_counts[k] for k in ("B0", "BO", "A", "D", "H"))
        if raw_sum != decoder_calls:
            problems.append(f"raw B0+BO+A+D+H={raw_sum} != decoder_calls={decoder_calls}")
    pred_sum = sum(predicted_totals[k] for k in ("B0", "BO", "A", "D", "H"))
    if pred_sum != decoder_calls:
        problems.append(f"predicted B0+BO+A+D+H={pred_sum} != decoder_calls={decoder_calls}")
    return (len(problems) == 0), "; ".join(problems)


def check_alignment_conservation(predicted, counters, is_aligned_arm):
    """Compare predicted aligned16 copy/pad/sketch totals against raw
    counters. Returns (ok: bool, detail: str)."""
    problems = []
    raw_fields = {
        "predicted_aligned_score_copies": "aligned_score_copies",
        "predicted_aligned_pad_bytes": "aligned_pad_bytes",
        "predicted_aligned_sketch_pads": "aligned_sketch_pads",
    }
    raw_values = {}
    for pred_key, raw_key in raw_fields.items():
        raw_val = counters.get(raw_key)
        raw_values[raw_key] = raw_val
        if raw_val is None:
            # Logical / native / legacy arms are not required to carry these
            # fields at all; that is fine as long as we did not predict a
            # nonzero exposure for them.
            if predicted[pred_key] != 0:
                problems.append(f"{raw_key} missing from counters but predicted {predicted[pred_key]}")
            continue
        if raw_val != predicted[pred_key]:
            problems.append(f"{raw_key}: predicted={predicted[pred_key]} raw={raw_val}")
    if not is_aligned_arm:
        # Logical/native/legacy arms must show zero exposure both ways.
        for raw_key in raw_fields.values():
            v = counters.get(raw_key)
            if v not in (None, 0):
                problems.append(f"non-aligned arm has nonzero {raw_key}={v}")
    return (len(problems) == 0), "; ".join(problems), raw_values


# ----------------------------------------------------------------------
# Protocol / arm classification.
# ----------------------------------------------------------------------

def classify_arms(protocol):
    """Return {arm_name: {'kind', 'route_storage', 'decision_interval'}}."""
    contracts = protocol["arm_contracts"]
    out = {}
    for arm, c in contracts.items():
        kind = c.get("kind")
        decision_interval = None
        route_storage = None
        if kind == "v21_method":
            route_storage = c.get("route_storage")  # 'aligned16' or None->logical
            parent = c.get("parent_v20_arm", "") or ""
            m = re.search(r"(?:^|_)R(\d+)(?:_|$)", parent)
            if m:
                decision_interval = int(m.group(1))
            # else: bootstrap-only arm (e.g. B_A8_matched) -> stays None,
            # meaning "this arm never issues D".
        out[arm] = dict(kind=kind, route_storage=route_storage, decision_interval=decision_interval)
    return out


# ----------------------------------------------------------------------
# Input loading.
# ----------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_manifests(manifests_dir):
    """Returns (prompt_token_count_by_id, sha_by_dataset). Never retains
    'prompt' or 'prompt_tokens' fields -- only prompt_token_count."""
    prompt_token_count_by_id = {}
    sha_by_dataset = {}
    for dataset, filename in MANIFEST_FILENAMES.items():
        path = os.path.join(manifests_dir, filename)
        sha_by_dataset[dataset] = {"path": path, "sha256": sha256_file(path)}
        rows = load_json(path)
        for row in rows:
            rid = row["id"]
            ptc = row["prompt_token_count"]
            if rid in prompt_token_count_by_id and prompt_token_count_by_id[rid] != ptc:
                raise ValueError(
                    f"conflicting prompt_token_count for id={rid}: "
                    f"{prompt_token_count_by_id[rid]} vs {ptc}"
                )
            prompt_token_count_by_id[rid] = ptc
    return prompt_token_count_by_id, sha_by_dataset


def load_ledger_runs(paths):
    """Merge 'run' events from one or more ledger JSONL files. The pilot
    ledgers are sharded (each file covers a disjoint subset of request
    cells), so events are concatenated, not de-duplicated by content --
    but exact-duplicate (dataset,id,seed,arm,role) tuples across files are
    flagged, not silently dropped."""
    runs = []
    seen_keys = {}
    dup_keys = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("event") != "run":
                    continue
                key = (d.get("dataset"), d.get("id"), d.get("seed"), d.get("arm"), d.get("role"))
                if key in seen_keys:
                    dup_keys.append((key, seen_keys[key], (path, line_no)))
                else:
                    seen_keys[key] = (path, line_no)
                d["_source_path"] = path
                d["_source_line"] = line_no
                runs.append(d)
    return runs, dup_keys


def load_scored_csv(path):
    """Returns {(dataset,id,seed,arm): row_dict} with seed as int."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["id"], int(row["seed"]), row["arm"])
            out[key] = row
    return out


# ----------------------------------------------------------------------
# Row construction.
# ----------------------------------------------------------------------

def build_row(run_event, arm_info, prompt_token_count_by_id, scored_by_key):
    dataset = run_event.get("dataset")
    rid = run_event.get("id")
    seed = run_event.get("seed")
    arm = run_event.get("arm")
    role = run_event.get("role")
    decoder_calls = run_event.get("decoder_calls")
    canvases = run_event.get("canvases")
    per_canvas_calls = run_event.get("per_canvas_calls")
    counters = run_event.get("counters") or {}

    info = arm_info.get(arm, {})
    kind = info.get("kind")
    route_storage = info.get("route_storage")
    decision_interval = info.get("decision_interval")
    is_v21_method = (kind == "v21_method")
    is_aligned_arm = (route_storage == "aligned16")

    row = {
        "dataset": dataset, "id": rid, "seed": seed, "arm": arm, "role": role,
        "kind": kind, "route_storage": route_storage or ("logical" if is_v21_method else ""),
        "host": run_event.get("host"), "gpu_uuid": run_event.get("gpu_uuid"),
        "decoder_calls": decoder_calls, "canvases": canvases,
        "completion_token_hash": run_event.get("completion_token_hash"),
        "api_wall_s": run_event.get("api_wall_s"),
    }

    prompt_token_count = prompt_token_count_by_id.get(rid)
    if prompt_token_count is None:
        raise KeyError(f"no prompt_token_count found for id={rid!r} in manifests")
    row["prompt_token_count"] = prompt_token_count

    if is_v21_method and per_canvas_calls is not None:
        predicted_totals, per_canvas_phases = predict_request_phases(per_canvas_calls, decision_interval)
        raw_counts, raw_problems = raw_phase_counts(counters)
        phase_ok, phase_detail = check_phase_conservation(predicted_totals, raw_counts, raw_problems, decoder_calls)

        align_pred = predict_alignment_exposure(per_canvas_calls, per_canvas_phases, prompt_token_count, is_aligned_arm)
        align_ok, align_detail, raw_align = check_alignment_conservation(align_pred, counters, is_aligned_arm)

        row.update({
            "k_mod16": align_pred["k_mod16"],
            "kdiv": align_pred["kdiv"],
            "odd_k_path": (align_pred["kdiv"] == 1),
            "predicted_B0": predicted_totals["B0"], "predicted_BO": predicted_totals["BO"],
            "predicted_A": predicted_totals["A"], "predicted_D": predicted_totals["D"],
            "predicted_H": predicted_totals["H"],
            "raw_B0": raw_counts["B0"] if raw_counts else "",
            "raw_BO": raw_counts["BO"] if raw_counts else "",
            "raw_A": raw_counts["A"] if raw_counts else "",
            "raw_D": raw_counts["D"] if raw_counts else "",
            "raw_H": raw_counts["H"] if raw_counts else "",
            "phase_conservation_ok": phase_ok,
            "phase_mismatch_detail": phase_detail,
            "predicted_aligned_score_copies": align_pred["predicted_aligned_score_copies"],
            "raw_aligned_score_copies": raw_align.get("aligned_score_copies", ""),
            "predicted_aligned_pad_bytes": align_pred["predicted_aligned_pad_bytes"],
            "raw_aligned_pad_bytes": raw_align.get("aligned_pad_bytes", ""),
            "predicted_aligned_sketch_pads": align_pred["predicted_aligned_sketch_pads"],
            "raw_aligned_sketch_pads": raw_align.get("aligned_sketch_pads", ""),
            "alignment_conservation_ok": align_ok,
            "alignment_mismatch_detail": align_detail,
            "affected_bo_plus_a_calls": align_pred["affected_bo_plus_a_calls"],
            "affected_d_calls": align_pred["affected_d_calls"],
            "fraction_calls_affected": (
                (align_pred["affected_bo_plus_a_calls"] + align_pred["affected_d_calls"]) / decoder_calls
                if decoder_calls else 0.0
            ),
        })
    else:
        row["k_mod16"] = prompt_token_count % 16
        row["kdiv"] = kdiv_of(prompt_token_count)
        row["odd_k_path"] = (row["kdiv"] == 1)
        for k in ("predicted_B0", "predicted_BO", "predicted_A", "predicted_D", "predicted_H",
                   "raw_B0", "raw_BO", "raw_A", "raw_D", "raw_H",
                   "predicted_aligned_score_copies", "raw_aligned_score_copies",
                   "predicted_aligned_pad_bytes", "raw_aligned_pad_bytes",
                   "predicted_aligned_sketch_pads", "raw_aligned_sketch_pads",
                   "affected_bo_plus_a_calls", "affected_d_calls", "fraction_calls_affected"):
            row[k] = ""
        row["phase_conservation_ok"] = "n/a"
        row["phase_mismatch_detail"] = "not a v21_method arm or missing per_canvas_calls"
        row["alignment_conservation_ok"] = "n/a"
        row["alignment_mismatch_detail"] = "not a v21_method arm or missing per_canvas_calls"

    scored_row = scored_by_key.get((dataset, rid, seed, arm))
    if scored_row:
        row["strict_correct"] = scored_row.get("strict_correct")
        row["warm_status"] = scored_row.get("warm_status")
        row["first_cold_request_wall_s"] = scored_row.get("first_cold_request_wall_s")
        row["accepted_warm_request_wall_s"] = scored_row.get("accepted_warm_request_wall_s")
    else:
        row["strict_correct"] = ""
        row["warm_status"] = ""
        row["first_cold_request_wall_s"] = ""
        row["accepted_warm_request_wall_s"] = ""

    return row


def build_paired_table(rows, scored_by_key):
    """Pair M3_boot_aligned16 vs M3_boot_logical attempt0 rows per
    (dataset,id,seed): token identity, call-count identity, warm wall
    ratio, amortized wall/decoder_calls ratio."""
    by_key = defaultdict(dict)
    for row in rows:
        if row["arm"] in ("M3_boot_aligned16", "M3_boot_logical") and row["role"] == "attempt0":
            by_key[(row["dataset"], row["id"], row["seed"])][row["arm"]] = row

    pairs = []
    for (dataset, rid, seed), arms in sorted(by_key.items()):
        a = arms.get("M3_boot_aligned16")
        l = arms.get("M3_boot_logical")
        entry = {"dataset": dataset, "id": rid, "seed": seed}
        if a is None or l is None:
            entry["note"] = "missing one of the two arms for this cell"
            pairs.append(entry)
            continue
        entry["aligned_completion_token_hash"] = a["completion_token_hash"]
        entry["logical_completion_token_hash"] = l["completion_token_hash"]
        entry["token_identity_match"] = (a["completion_token_hash"] == l["completion_token_hash"])
        entry["aligned_decoder_calls"] = a["decoder_calls"]
        entry["logical_decoder_calls"] = l["decoder_calls"]
        entry["calls_match"] = (a["decoder_calls"] == l["decoder_calls"])
        entry["aligned_host"] = a["host"]
        entry["logical_host"] = l["host"]

        scored_a = scored_by_key.get((dataset, rid, seed, "M3_boot_aligned16"))
        scored_l = scored_by_key.get((dataset, rid, seed, "M3_boot_logical"))
        warm = {}
        for label, scored_row, decoder_calls in (
            ("aligned", scored_a, a["decoder_calls"]),
            ("logical", scored_l, l["decoder_calls"]),
        ):
            if not scored_row or scored_row.get("warm_status") != "accepted" or not scored_row.get("accepted_warm_request_wall_s"):
                warm[label] = {"accepted": False, "note": "no accepted warm run for this cell"}
                continue
            wall_s = float(scored_row["accepted_warm_request_wall_s"])
            warm[label] = {
                "accepted": True,
                "wall_s": wall_s,
                "amortized_s_per_call": (wall_s / decoder_calls) if decoder_calls else None,
            }
        entry["warm"] = warm
        if warm.get("aligned", {}).get("accepted") and warm.get("logical", {}).get("accepted"):
            entry["warm_wall_ratio_aligned_over_logical"] = warm["aligned"]["wall_s"] / warm["logical"]["wall_s"]
            entry["warm_amortized_ratio_aligned_over_logical"] = (
                warm["aligned"]["amortized_s_per_call"] / warm["logical"]["amortized_s_per_call"]
            )
        else:
            entry["warm_wall_ratio_aligned_over_logical"] = None
            entry["warm_amortized_ratio_aligned_over_logical"] = None
            entry["warm_note"] = "at least one arm has no accepted warm run for this cell; ratio unavailable"
        pairs.append(entry)
    return pairs


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--scored-csv", required=True)
    p.add_argument("--ledger", action="append", required=True, dest="ledgers",
                    help="path to a ledger JSONL file; may be given multiple times")
    p.add_argument("--manifests-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for out_path in (args.out_csv, args.out_json):
        if os.path.exists(out_path):
            print(f"refusing to overwrite existing output: {out_path}", file=sys.stderr)
            return 2

    protocol = load_json(args.protocol)
    arm_info = classify_arms(protocol)
    prompt_token_count_by_id, manifest_sha = load_manifests(args.manifests_dir)
    scored_by_key = load_scored_csv(args.scored_csv)
    run_events, dup_keys = load_ledger_runs(args.ledgers)

    if dup_keys:
        for key, first_loc, dup_loc in dup_keys:
            print(f"WARNING: duplicate ledger run event for {key}: first at {first_loc}, "
                  f"also at {dup_loc}", file=sys.stderr)

    rows = []
    build_errors = []
    for run_event in run_events:
        try:
            rows.append(build_row(run_event, arm_info, prompt_token_count_by_id, scored_by_key))
        except Exception as exc:  # noqa: BLE001 -- surfaced loudly below, not swallowed
            build_errors.append({
                "dataset": run_event.get("dataset"), "id": run_event.get("id"),
                "seed": run_event.get("seed"), "arm": run_event.get("arm"),
                "role": run_event.get("role"), "error": str(exc),
            })

    if build_errors:
        print(f"ERROR: {len(build_errors)} ledger run event(s) failed to reduce:", file=sys.stderr)
        for e in build_errors:
            print(f"  {e}", file=sys.stderr)

    rows.sort(key=lambda r: (r["dataset"] or "", r["id"] or "", r["seed"] if r["seed"] is not None else -1, r["arm"] or "", r["role"] or ""))

    phase_mismatches = [r for r in rows if r["phase_conservation_ok"] is False]
    alignment_mismatches = [r for r in rows if r["alignment_conservation_ok"] is False]
    if phase_mismatches:
        print(f"CONSERVATION MISMATCH: {len(phase_mismatches)} row(s) failed phase conservation:", file=sys.stderr)
        for r in phase_mismatches:
            print(f"  {r['dataset']}/{r['id']} seed={r['seed']} arm={r['arm']} role={r['role']}: {r['phase_mismatch_detail']}", file=sys.stderr)
    if alignment_mismatches:
        print(f"CONSERVATION MISMATCH: {len(alignment_mismatches)} row(s) failed alignment conservation:", file=sys.stderr)
        for r in alignment_mismatches:
            print(f"  {r['dataset']}/{r['id']} seed={r['seed']} arm={r['arm']} role={r['role']}: {r['alignment_mismatch_detail']}", file=sys.stderr)

    paired = build_paired_table(rows, scored_by_key)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    inputs_sha = {
        "protocol": {"path": args.protocol, "sha256": sha256_file(args.protocol)},
        "scored_csv": {"path": args.scored_csv, "sha256": sha256_file(args.scored_csv)},
        "ledgers": [{"path": p, "sha256": sha256_file(p)} for p in args.ledgers],
        "manifests": manifest_sha,
    }

    out_json = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs_sha256": inputs_sha,
        "row_count": len(rows),
        "build_errors": build_errors,
        "duplicate_ledger_keys": [
            {"key": list(k), "first": list(first), "duplicate": list(dup)}
            for k, first, dup in dup_keys
        ],
        "conservation_summary": {
            "phase_mismatch_count": len(phase_mismatches),
            "alignment_mismatch_count": len(alignment_mismatches),
            "all_ok": (len(phase_mismatches) == 0 and len(alignment_mismatches) == 0 and not build_errors),
        },
        "rows": rows,
        "paired_m3_aligned_vs_logical": paired,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, sort_keys=False)

    print(f"wrote {len(rows)} rows to {args.out_csv} and {args.out_json}")
    print(f"phase conservation: {len(rows) - len(phase_mismatches)}/{len(rows)} ok; "
          f"alignment conservation: {len(rows) - len(alignment_mismatches)}/{len(rows)} ok")
    return 0 if (not phase_mismatches and not alignment_mismatches and not build_errors) else 1


if __name__ == "__main__":
    sys.exit(main())
