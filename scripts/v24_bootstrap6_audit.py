"""v24 bootstrap6 cross-segment output-numerics audit. Stdlib-only.

Two (or more) execution "segments" jointly cover one frozen v21 bootstrap6
panel (240 executions = 120 cells, each cell = one attempt0 run + one warm
repeat). Each segment contributes a scorer CSV that has a placeholder row
for every one of the 120 cells (executed cells have a non-empty
``decoder_calls``; cells the segment did not run are blank placeholders),
plus the two per-host ledgers (jsonl) that actually ran that segment's
share of the panel, plus the protocol.json/binding.json pair the segment
was launched under.

This script:
  A. Verifies the segments describe the SAME frozen protocol, that every
     cell was executed in exactly one segment, and cross-checks each
     executed row against the ledger 'run' events (host/gpu/decoder_calls).
  B. Restricts paired contrasts to same-host/same-gpu blocks (fails loudly
     on a cross-host pair).
  C. Reports per-dataset geo-mean W/N/(W/N) ratio estimands (+ identity
     check), the different sum-calls-ratio estimand, a host-stratified W/N
     decomposition, and discordant correctness counts.
  D. Reconstructs the five-phase (B0/BO/A/D/H) layer-call accounting for
     every executed FIRST (attempt0) run of the three bootstrap arms and
     the incumbent, straight from raw ledger counters, with an explicit
     conservation check (B0+BO+A+D+H == 5*decoder_calls).
  E. Writes JSON (everything), CSV (one row per cell) and MD (compact
     English report) outputs, refusing to overwrite existing files.

Usage:
    python -m scripts.v24_bootstrap6_audit \\
        --segment preview_aime=SCORED_CSV,LEDGER1,LEDGER2,PROTOCOL,BINDING \\
        --segment lb_rest=SCORED_CSV,LEDGER1,LEDGER2,PROTOCOL,BINDING \\
        --out-json OUT.json --out-md OUT.md --out-csv OUT.csv

No third-party imports; everything here is Python stdlib.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone

ARMS = [
    "D_native",
    "T_scope",
    "M3_R3_A8_incumbent",
    "M1_native_bootstrap2_observe1",
    "M3_native_bootstrap2_observe1",
    "B_native_bootstrap2_observe1",
]

BOOTSTRAP_ARMS = {
    "M1_native_bootstrap2_observe1",
    "M3_native_bootstrap2_observe1",
    "B_native_bootstrap2_observe1",
}

# Arms audited for the five-phase (B0/BO/A/D/H) layer-call accounting:
# the three native-bootstrap arms plus the incumbent they are compared to.
FIVE_PHASE_ARMS = BOOTSTRAP_ARMS | {"M3_R3_A8_incumbent"}

DATASETS = ["aime26", "longbench_v2"]

# (armX, armY) contrasts requested for the estimand table.
CONTRASTS = [
    ("M3_native_bootstrap2_observe1", "D_native"),
    ("M3_native_bootstrap2_observe1", "M3_R3_A8_incumbent"),
    ("M3_native_bootstrap2_observe1", "B_native_bootstrap2_observe1"),
    ("M3_native_bootstrap2_observe1", "T_scope"),
    ("B_native_bootstrap2_observe1", "D_native"),
    ("M1_native_bootstrap2_observe1", "D_native"),
    ("T_scope", "D_native"),
]

FIVE_PHASE_COUNTER_KEYS = [
    "bootstrap_dense_calls",
    "bootstrap_observation_calls",
    "score_refresh_calls",
    "decision_refresh_calls",
    "held_decision_calls",
]

# Cell classes. "invalid" covers every structural audit violation
# (duplicate execution, host/gpu mismatch, ledger cross-check failure,
# malformed executed row). Never count failed/unscored/invalid as wrong.
VALID_SCORED_CLASSES = ("scored_correct", "scored_wrong")
EXECUTED_STRUCTURALLY_VALID_CLASSES = (
    "scored_correct",
    "scored_wrong",
    "unscored",
    "failed",
)

HEADER_NOTE = (
    "First-generation ('first') outputs only were used for quality scoring; "
    "the warm repeat is a timing measurement only, never rescored. In this "
    "panel host is confounded with seed by design: seed 101 always ran on "
    "149.165.151.254, seed 202 always ran on 149.165.159.64. All contrasts "
    "below are exploratory question-cluster comparisons on a frozen, "
    "correctness-gated but small development panel -- not a noninferiority "
    "test."
)

IDENTITY_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# small stdlib helpers
# --------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_ledger_run_events(path):
    """Return only the 'run' events of a ledger jsonl file."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("event") == "run":
                out.append(d)
    return out


def parse_bool_field(s):
    """CSV booleans are the literal strings 'True'/'False'."""
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def parse_float_or_none(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int_or_none(s):
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def geomean(values):
    """Geometric mean of a list of strictly positive floats, or None if empty."""
    if not values:
        return None
    total_log = 0.0
    for v in values:
        if v is None or v <= 0 or not math.isfinite(v):
            raise ValueError("geomean requires strictly positive finite values, got %r" % (v,))
        total_log += math.log(v)
    return math.exp(total_log / len(values))


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# segment loading
# --------------------------------------------------------------------------

class Segment(object):
    def __init__(self, name, scored_csv, ledger1, ledger2, protocol_path, binding_path):
        self.name = name
        self.scored_csv = scored_csv
        self.ledger_paths = [ledger1, ledger2]
        self.protocol_path = protocol_path
        self.binding_path = binding_path

        self.scored_rows = read_csv_rows(scored_csv)
        self.ledger_runs = []
        for lp in self.ledger_paths:
            for run in read_ledger_run_events(lp):
                run = dict(run)
                run["_ledger_path"] = lp
                run["_segment"] = name
                self.ledger_runs.append(run)
        self.protocol = read_json(protocol_path)
        self.binding = read_json(binding_path)
        self.protocol_sha256 = sha256_file(protocol_path)

        self.rows_by_key = {}
        for row in self.scored_rows:
            key = (row["dataset"], row["id"], row["seed"], row["arm"])
            self.rows_by_key[key] = row

    def input_paths(self):
        return [self.scored_csv] + self.ledger_paths + [self.protocol_path, self.binding_path]


def parse_segment_arg(arg):
    """NAME=SCORED_CSV,LEDGER1,LEDGER2,PROTOCOL,BINDING"""
    if "=" not in arg:
        raise ValueError("--segment must be NAME=SCORED_CSV,LEDGER1,LEDGER2,PROTOCOL,BINDING, got: %r" % (arg,))
    name, rest = arg.split("=", 1)
    parts = rest.split(",")
    if len(parts) != 5:
        raise ValueError(
            "--segment %r: expected 5 comma-separated paths "
            "(scored_csv,ledger1,ledger2,protocol,binding), got %d" % (name, len(parts))
        )
    return (name.strip(), [p.strip() for p in parts])


# --------------------------------------------------------------------------
# protocol registry
# --------------------------------------------------------------------------

def build_registry(protocol):
    """(dataset,id,seed,arm) -> {cell_id, block, host, gpu_uuid}; also
    returns block_assignments and any internal-consistency reasons found
    between a cell's attempt0 and warm schedule entries."""
    registry = {}
    reasons = []
    by_cell = defaultdict(list)
    for entry in protocol["schedule"]:
        by_cell[entry["cell_id"]].append(entry)

    for cell_id, entries in by_cell.items():
        roles = sorted(e["role"] for e in entries)
        if roles != ["attempt0", "warm"]:
            reasons.append("cell_id=%s does not have exactly one attempt0 and one warm schedule entry (roles=%r)" % (cell_id, roles))
            continue
        a0 = next(e for e in entries if e["role"] == "attempt0")
        wr = next(e for e in entries if e["role"] == "warm")
        for field in ("dataset", "id", "seed", "arm", "block", "host", "gpu_uuid"):
            if a0[field] != wr[field]:
                reasons.append(
                    "cell_id=%s attempt0/warm schedule entries disagree on %s (%r vs %r)"
                    % (cell_id, field, a0[field], wr[field])
                )
        key = (a0["dataset"], a0["id"], str(a0["seed"]), a0["arm"])
        if key in registry:
            reasons.append("duplicate (dataset,id,seed,arm) key %r in protocol schedule (cell_ids %s and %s)" % (key, registry[key]["cell_id"], cell_id))
            continue
        registry[key] = {
            "cell_id": cell_id,
            "block": a0["block"],
            "host": a0["host"],
            "gpu_uuid": a0["gpu_uuid"],
        }

    block_assignments = protocol.get("block_assignments", {})
    return registry, block_assignments, reasons


# --------------------------------------------------------------------------
# five-phase accounting
# --------------------------------------------------------------------------

def five_phase_from_counters(counters, decoder_calls):
    """Extract the B0/BO/A/D/H global layer-call phase accounting from a raw
    ledger 'counters' dict. Never infer a zero for an absent field -- if any
    of the five raw counters is missing, the corresponding derived field(s)
    are None. Returns a dict with keys:
    B0, BO, A, D, H, model_calls, conservation_ok, BO_lt_B0,
    observation_frequency, five_phase_fields_complete.
    """
    out = dict.fromkeys(
        ["B0", "BO", "A", "D", "H", "model_calls", "conservation_ok", "BO_lt_B0", "observation_frequency"],
        None,
    )
    out["five_phase_fields_complete"] = False
    if counters is None:
        return out

    have_all = all(k in counters for k in FIVE_PHASE_COUNTER_KEYS)
    out["five_phase_fields_complete"] = have_all
    if not have_all:
        # still surface whichever raw fields ARE present, un-derived.
        if "bootstrap_dense_calls" in counters:
            out["B0"] = counters["bootstrap_dense_calls"]
        if "bootstrap_observation_calls" in counters:
            out["BO"] = counters["bootstrap_observation_calls"]
        if "score_refresh_calls" in counters:
            out["A"] = counters["score_refresh_calls"]
        if out["B0"] is not None and out["BO"] is not None:
            out["BO_lt_B0"] = out["BO"] < out["B0"]
        return out

    b0 = counters["bootstrap_dense_calls"]
    bo = counters["bootstrap_observation_calls"]
    a = counters["score_refresh_calls"]
    decision_refresh = counters["decision_refresh_calls"]
    h = counters["held_decision_calls"]
    d = decision_refresh - a

    out["B0"] = b0
    out["BO"] = bo
    out["A"] = a
    out["D"] = d
    out["H"] = h
    out["BO_lt_B0"] = bo < b0
    total = b0 + bo + a + d + h
    out["model_calls"] = total / 5.0
    if decoder_calls is not None:
        out["conservation_ok"] = (total == 5 * decoder_calls)
        out["observation_frequency"] = (bo + a) / decoder_calls if decoder_calls else None
    return out


# --------------------------------------------------------------------------
# per-cell audit
# --------------------------------------------------------------------------

def audit_cell(key, registry, block_assignments, seg_rows_by_key, ledger_index):
    """Audit one (dataset,id,seed,arm) cell across every segment.

    seg_rows_by_key: {segment_name: {key: row_or_None}}
    ledger_index: {(cell_id, role): [run_dict, ...]}  (combined over ALL
    segments' ledger files, so a misfiled duplicate is still caught)
    """
    dataset, id_, seed, arm = key
    reasons = []
    reg = registry.get(key)
    if reg is None:
        return {
            "dataset": dataset, "id": id_, "seed": seed, "arm": arm,
            "cell_id": None, "block": None, "host": None, "gpu_uuid": None,
            "class": "invalid", "reasons": ["key_not_in_protocol_schedule"],
            "executed_segment": None, "first_status": None,
            "quality_eligible": None, "scored_first": None,
            "strict_correct": None, "warm_status": None,
            "accepted_warm_request_wall_s": None,
            "decoder_calls": None, "canvases": None,
            "ledger_attempt0_decoder_calls": None, "ledger_warm_decoder_calls": None,
            "five_phase": five_phase_from_counters(None, None),
        }

    cell_id, block, reg_host, reg_gpu = reg["cell_id"], reg["block"], reg["host"], reg["gpu_uuid"]

    ba = block_assignments.get(str(block))
    if ba is None:
        reasons.append("block_assignment_missing_for_block_%s" % block)
    else:
        if ba.get("host") != reg_host or ba.get("gpu_uuid") != reg_gpu \
                or ba.get("dataset") != dataset or ba.get("id") != id_ \
                or str(ba.get("seed")) != str(seed):
            reasons.append("block_assignment_inconsistent_with_schedule_for_block_%s" % block)

    executed = []
    for seg_name, rows in seg_rows_by_key.items():
        row = rows.get(key)
        if row is not None and row.get("decoder_calls", "") != "":
            executed.append((seg_name, row))

    cls = None
    row = None
    seg_name = None
    a0 = None

    if len(executed) == 0:
        cls = "missing"
    elif len(executed) > 1:
        cls = "invalid"
        seg_names = sorted(s for s, _ in executed)
        calls_vals = set(r.get("decoder_calls") for _, r in executed)
        if len(calls_vals) == 1 and all(executed[0][1] == r for _, r in executed):
            reasons.append("duplicate_execution_identical segments=%s" % ",".join(seg_names))
        else:
            reasons.append("duplicate_execution_conflicting segments=%s" % ",".join(seg_names))
        # keep the first row only for descriptive fields in the record
        seg_name, row = executed[0]
    else:
        seg_name, row = executed[0]

        if row.get("cell_id") != cell_id:
            reasons.append("cell_id_mismatch csv=%r protocol=%r" % (row.get("cell_id"), cell_id))
        if row.get("host") != reg_host or row.get("gpu_uuid") != reg_gpu:
            reasons.append("host_gpu_mismatch csv=(%r,%r) protocol=(%r,%r)" % (row.get("host"), row.get("gpu_uuid"), reg_host, reg_gpu))

        a0_runs = ledger_index.get((cell_id, "attempt0"), [])
        w_runs = ledger_index.get((cell_id, "warm"), [])
        if len(a0_runs) != 1:
            reasons.append("ledger_attempt0_count_%d_expected_1" % len(a0_runs))
        if len(w_runs) != 1:
            reasons.append("ledger_warm_count_%d_expected_1" % len(w_runs))
        a0 = a0_runs[0] if len(a0_runs) == 1 else None
        wr = w_runs[0] if len(w_runs) == 1 else None

        if a0 is not None:
            if a0.get("host") != row.get("host") or a0.get("gpu_uuid") != row.get("gpu_uuid"):
                reasons.append("ledger_attempt0_host_gpu_mismatch")
            if parse_int_or_none(str(a0.get("decoder_calls"))) != parse_int_or_none(row.get("decoder_calls")):
                reasons.append("ledger_attempt0_decoder_calls_mismatch ledger=%r csv=%r" % (a0.get("decoder_calls"), row.get("decoder_calls")))
        if wr is not None:
            if wr.get("host") != row.get("host") or wr.get("gpu_uuid") != row.get("gpu_uuid"):
                reasons.append("ledger_warm_host_gpu_mismatch")

        if reasons:
            cls = "invalid"
        else:
            status = row.get("first_status")
            if status != "success":
                cls = "failed"
            else:
                qual_ok = parse_bool_field(row.get("quality_eligible")) is True
                scored_ok = parse_bool_field(row.get("scored_first")) is True
                if not (qual_ok and scored_ok):
                    cls = "unscored"
                else:
                    warm_status = row.get("warm_status", "")
                    if warm_status == "":
                        reasons.append("executed_row_missing_warm_status")
                        cls = "invalid"
                    else:
                        if warm_status == "accepted":
                            wall = parse_float_or_none(row.get("accepted_warm_request_wall_s"))
                            if wall is None or wall <= 0 or not math.isfinite(wall):
                                reasons.append("warm_accepted_without_finite_positive_wall")
                                cls = "invalid"
                        # correctness classification (independent of warm outcome)
                        if cls != "invalid":
                            sc = row.get("strict_correct")
                            if sc == "True":
                                cls = "scored_correct"
                            elif sc == "False":
                                cls = "scored_wrong"
                            else:
                                reasons.append("scored_first_true_but_strict_correct_not_boolean=%r" % sc)
                                cls = "invalid"

    five_phase = five_phase_from_counters(None, None)
    if arm in FIVE_PHASE_ARMS and cls in EXECUTED_STRUCTURALLY_VALID_CLASSES and a0 is not None:
        five_phase = five_phase_from_counters(a0.get("counters"), a0.get("decoder_calls"))

    ledger_a0_calls = a0.get("decoder_calls") if a0 is not None else None
    warm_runs_for_calls = ledger_index.get((cell_id, "warm"), [])
    ledger_warm_calls = warm_runs_for_calls[0].get("decoder_calls") if len(warm_runs_for_calls) == 1 else None

    return {
        "dataset": dataset, "id": id_, "seed": seed, "arm": arm,
        "cell_id": cell_id, "block": block, "host": reg_host, "gpu_uuid": reg_gpu,
        "class": cls, "reasons": reasons,
        "executed_segment": seg_name,
        "first_status": row.get("first_status") if row is not None else None,
        "quality_eligible": row.get("quality_eligible") if row is not None else None,
        "scored_first": row.get("scored_first") if row is not None else None,
        "strict_correct": row.get("strict_correct") if row is not None else None,
        "warm_status": row.get("warm_status") if row is not None else None,
        "accepted_warm_request_wall_s": parse_float_or_none(row.get("accepted_warm_request_wall_s")) if row is not None else None,
        "decoder_calls": parse_int_or_none(row.get("decoder_calls")) if row is not None else None,
        "canvases": parse_int_or_none(row.get("canvases")) if row is not None else None,
        "ledger_attempt0_decoder_calls": ledger_a0_calls,
        "ledger_warm_decoder_calls": ledger_warm_calls,
        "five_phase": five_phase,
    }


# --------------------------------------------------------------------------
# estimands
# --------------------------------------------------------------------------

def get_N_for_record(record, ledger_index):
    """N for the estimand tables: prefer the WARM ledger run's decoder_calls
    (that's the run W was actually timed on); fall back to the CSV's
    (attempt0/'first') decoder_calls, and say which was used."""
    runs = ledger_index.get((record["cell_id"], "warm"), [])
    if len(runs) == 1 and runs[0].get("decoder_calls") is not None:
        return int(runs[0]["decoder_calls"]), "ledger_warm_decoder_calls"
    return record["decoder_calls"], "csv_attempt0_decoder_calls"


def compute_estimand(dataset, arm_x, arm_y, cells_by_key, ledger_index):
    ids_seeds = sorted({(k[1], k[2]) for k in cells_by_key if k[0] == dataset})

    valid_pairs = []
    timing_pairs = []
    host_cross_fatal = []
    n_sources = set()

    for (id_, seed) in ids_seeds:
        kx = (dataset, id_, seed, arm_x)
        ky = (dataset, id_, seed, arm_y)
        rx = cells_by_key.get(kx)
        ry = cells_by_key.get(ky)
        if rx is None or ry is None:
            continue

        if rx["class"] in VALID_SCORED_CLASSES and ry["class"] in VALID_SCORED_CLASSES:
            valid_pairs.append((id_, seed, rx, ry))

        timing_ok = (
            rx["class"] in EXECUTED_STRUCTURALLY_VALID_CLASSES
            and ry["class"] in EXECUTED_STRUCTURALLY_VALID_CLASSES
            and rx.get("warm_status") == "accepted"
            and ry.get("warm_status") == "accepted"
            and rx.get("accepted_warm_request_wall_s") is not None
            and ry.get("accepted_warm_request_wall_s") is not None
        )
        if timing_ok:
            if rx["host"] != ry["host"] or rx["gpu_uuid"] != ry["gpu_uuid"]:
                host_cross_fatal.append({"id": id_, "seed": seed, "host_x": rx["host"], "host_y": ry["host"]})
                continue
            n_x, src_x = get_N_for_record(rx, ledger_index)
            n_y, src_y = get_N_for_record(ry, ledger_index)
            n_sources.add(src_x)
            n_sources.add(src_y)
            if n_x is None or n_y is None or n_x <= 0 or n_y <= 0:
                continue
            timing_pairs.append({
                "id": id_, "seed": seed, "host": rx["host"],
                "w_x": rx["accepted_warm_request_wall_s"], "w_y": ry["accepted_warm_request_wall_s"],
                "n_x": n_x, "n_y": n_y,
            })

    result = {
        "dataset": dataset, "arm_x": arm_x, "arm_y": arm_y,
        "n_valid_pairs": len(valid_pairs), "n_timing_pairs": len(timing_pairs),
        "host_cross_fatal": host_cross_fatal,
        "N_sources_used": sorted(n_sources),
    }

    if timing_pairs:
        w_ratios = [p["w_x"] / p["w_y"] for p in timing_pairs]
        n_ratios = [p["n_x"] / p["n_y"] for p in timing_pairs]
        wn_ratios = [(p["w_x"] / p["n_x"]) / (p["w_y"] / p["n_y"]) for p in timing_pairs]
        geo_w = geomean(w_ratios)
        geo_n = geomean(n_ratios)
        geo_wn = geomean(wn_ratios)
        result["geo_W_ratio"] = geo_w
        result["geo_N_ratio"] = geo_n
        result["geo_W_over_N_ratio"] = geo_wn
        result["geo_identity_abs_diff"] = abs(geo_w - geo_n * geo_wn)
        result["geo_identity_ok"] = result["geo_identity_abs_diff"] < IDENTITY_TOLERANCE

        sum_n_x = sum(p["n_x"] for p in timing_pairs)
        sum_n_y = sum(p["n_y"] for p in timing_pairs)
        result["sum_calls_ratio"] = (sum_n_x / sum_n_y) if sum_n_y else None

        by_host = defaultdict(lambda: {"sum_w_x": 0.0, "sum_w_y": 0.0, "sum_n_x": 0, "sum_n_y": 0, "n": 0})
        for p in timing_pairs:
            bucket = by_host[p["host"]]
            bucket["sum_w_x"] += p["w_x"]
            bucket["sum_w_y"] += p["w_y"]
            bucket["sum_n_x"] += p["n_x"]
            bucket["sum_n_y"] += p["n_y"]
            bucket["n"] += 1
        host_stratified = {}
        for host, b in by_host.items():
            host_stratified[host] = {
                "n_pairs": b["n"],
                "sum_W_x": b["sum_w_x"], "sum_W_y": b["sum_w_y"],
                "sum_N_x": b["sum_n_x"], "sum_N_y": b["sum_n_y"],
                "sum_W_ratio": (b["sum_w_x"] / b["sum_w_y"]) if b["sum_w_y"] else None,
                "sum_N_ratio": (b["sum_n_x"] / b["sum_n_y"]) if b["sum_n_y"] else None,
            }
        result["host_stratified"] = host_stratified
    else:
        result["geo_W_ratio"] = None
        result["geo_N_ratio"] = None
        result["geo_W_over_N_ratio"] = None
        result["geo_identity_abs_diff"] = None
        result["geo_identity_ok"] = None
        result["sum_calls_ratio"] = None
        result["host_stratified"] = {}

    disc_x_correct_y_wrong = sum(
        1 for (_, _, rx, ry) in valid_pairs if rx["strict_correct"] == "True" and ry["strict_correct"] == "False"
    )
    disc_x_wrong_y_correct = sum(
        1 for (_, _, rx, ry) in valid_pairs if rx["strict_correct"] == "False" and ry["strict_correct"] == "True"
    )
    result["discordant_x_correct_y_wrong"] = disc_x_correct_y_wrong
    result["discordant_x_wrong_y_correct"] = disc_x_wrong_y_correct

    return result


# --------------------------------------------------------------------------
# top-level audit driver (importable, used by both main() and tests)
# --------------------------------------------------------------------------

def run_audit(segments):
    """segments: list of Segment. Returns the full result dict."""
    union_reasons = []

    protocol_shas = {seg.name: seg.protocol_sha256 for seg in segments}
    distinct_shas = set(protocol_shas.values())
    protocol_consistent = len(distinct_shas) <= 1
    if not protocol_consistent:
        union_reasons.append("protocol.json sha256 differs across segments: %r" % (protocol_shas,))

    protocol_sha256 = next(iter(distinct_shas)) if distinct_shas else None

    binding_checks = {}
    for seg in segments:
        panel_sha = seg.binding.get("panel_protocol_sha256")
        ok = (panel_sha == protocol_sha256) if protocol_sha256 is not None else False
        binding_checks[seg.name] = {
            "panel_protocol_sha256": panel_sha,
            "matches_protocol_sha256": ok,
            "source_commits": seg.binding.get("source_commits"),
        }
        if not ok:
            union_reasons.append(
                "segment %s: binding.panel_protocol_sha256 (%r) != protocol sha256 (%r)"
                % (seg.name, panel_sha, protocol_sha256)
            )

    # Registry + block_assignments come from the (assumed identical) protocol.
    # Use the first segment's protocol; if protocols disagree this is already
    # flagged above, and results should be treated as unreliable (FAIL).
    registry, block_assignments, registry_reasons = build_registry(segments[0].protocol)
    union_reasons.extend(registry_reasons)

    seg_rows_by_key = {seg.name: seg.rows_by_key for seg in segments}

    ledger_index = defaultdict(list)
    for seg in segments:
        for run in seg.ledger_runs:
            ledger_index[(run.get("cell_id"), run.get("role"))].append(run)

    all_keys = set(registry.keys())
    for seg in segments:
        all_keys.update(seg.rows_by_key.keys())

    cells = []
    for key in sorted(all_keys):
        record = audit_cell(key, registry, block_assignments, seg_rows_by_key, ledger_index)
        cells.append(record)
        if record["class"] == "invalid":
            union_reasons.append(
                "cell dataset=%s id=%s seed=%s arm=%s INVALID: %s"
                % (record["dataset"], record["id"], record["seed"], record["arm"], "; ".join(record["reasons"]))
            )

    n_missing = sum(1 for c in cells if c["class"] == "missing")
    n_invalid = sum(1 for c in cells if c["class"] == "invalid")

    cells_by_key = {(c["dataset"], c["id"], c["seed"], c["arm"]): c for c in cells}

    estimands = []
    for dataset in DATASETS:
        for (arm_x, arm_y) in CONTRASTS:
            est = compute_estimand(dataset, arm_x, arm_y, cells_by_key, ledger_index)
            if est["host_cross_fatal"]:
                union_reasons.append(
                    "dataset=%s contrast %s vs %s: %d block(s) would pair across hosts: %r"
                    % (dataset, arm_x, arm_y, len(est["host_cross_fatal"]), est["host_cross_fatal"])
                )
            estimands.append(est)

    # per-dataset x arm summary
    per_dataset_arm = {}
    for dataset in DATASETS:
        per_dataset_arm[dataset] = {}
        for arm in ARMS:
            rows = [c for c in cells if c["dataset"] == dataset and c["arm"] == arm]
            n_correct = sum(1 for c in rows if c["class"] == "scored_correct")
            n_wrong = sum(1 for c in rows if c["class"] == "scored_wrong")
            n_failed = sum(1 for c in rows if c["class"] == "failed")
            n_unscored = sum(1 for c in rows if c["class"] == "unscored")
            n_missing_arm = sum(1 for c in rows if c["class"] == "missing")
            n_invalid_arm = sum(1 for c in rows if c["class"] == "invalid")
            valid_n = n_correct + n_wrong
            throughput_rows = [c for c in rows if c["class"] in EXECUTED_STRUCTURALLY_VALID_CLASSES]
            total_calls = sum(c["decoder_calls"] for c in throughput_rows if c["decoder_calls"] is not None)
            total_canvases = sum(c["canvases"] for c in throughput_rows if c["canvases"] is not None)
            calls_per_canvas = (total_calls / total_canvases) if total_canvases else None

            five_phase_totals = None
            if arm in FIVE_PHASE_ARMS:
                fp_rows = [c["five_phase"] for c in throughput_rows if c["five_phase"]["five_phase_fields_complete"]]
                if fp_rows:
                    b0 = sum(fp["B0"] for fp in fp_rows)
                    bo = sum(fp["BO"] for fp in fp_rows)
                    a = sum(fp["A"] for fp in fp_rows)
                    d = sum(fp["D"] for fp in fp_rows)
                    h = sum(fp["H"] for fp in fp_rows)
                    five_phase_totals = {
                        "n_runs": len(fp_rows),
                        "B0": b0, "BO": bo, "A": a, "D": d, "H": h,
                        "model_calls_total": (b0 + bo + a + d + h) / 5.0,
                        "observation_frequency": (bo + a) / total_calls if total_calls else None,
                        "conservation_violations": sum(1 for fp in fp_rows if fp["conservation_ok"] is False),
                    }
                else:
                    five_phase_totals = {"n_runs": 0}

            per_dataset_arm[dataset][arm] = {
                "n_correct": n_correct, "n_wrong": n_wrong, "n_scored_valid": valid_n,
                "n_failed": n_failed, "n_unscored": n_unscored,
                "n_missing": n_missing_arm, "n_invalid": n_invalid_arm,
                "total_decoder_calls": total_calls, "total_canvases": total_canvases,
                "calls_per_canvas": calls_per_canvas,
                "five_phase_totals": five_phase_totals,
            }

    union_verdict = "PASS" if (protocol_consistent and n_invalid == 0 and all(b["matches_protocol_sha256"] for b in binding_checks.values())) else "FAIL"

    result = {
        "generated_at": now_iso(),
        "header_note": HEADER_NOTE,
        "protocol_sha256": protocol_sha256,
        "protocol_id": segments[0].protocol.get("protocol_id"),
        "segments": {
            seg.name: {
                "scored_csv": seg.scored_csv,
                "ledger_paths": seg.ledger_paths,
                "protocol_path": seg.protocol_path,
                "binding_path": seg.binding_path,
                "protocol_sha256": seg.protocol_sha256,
                "binding": binding_checks[seg.name],
            }
            for seg in segments
        },
        "union_verdict": union_verdict,
        "union_reasons": union_reasons,
        "n_cells_total": len(cells),
        "n_missing": n_missing,
        "n_invalid": n_invalid,
        "cells": cells,
        "per_dataset_arm_summary": per_dataset_arm,
        "estimands": estimands,
    }
    return result


# --------------------------------------------------------------------------
# output writers
# --------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "id", "seed", "arm", "cell_id", "block", "host", "gpu_uuid",
    "class", "reasons", "executed_segment",
    "first_status", "quality_eligible", "scored_first", "strict_correct",
    "warm_status", "accepted_warm_request_wall_s",
    "decoder_calls", "canvases",
    "ledger_attempt0_decoder_calls", "ledger_warm_decoder_calls",
    "B0", "BO", "A", "D", "H", "model_calls", "conservation_ok", "BO_lt_B0",
    "observation_frequency",
]


def write_csv(path, cells):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for c in cells:
            fp = c["five_phase"]
            w.writerow([
                c["dataset"], c["id"], c["seed"], c["arm"], c["cell_id"], c["block"],
                c["host"], c["gpu_uuid"], c["class"], "; ".join(c["reasons"]), c["executed_segment"],
                c["first_status"], c["quality_eligible"], c["scored_first"], c["strict_correct"],
                c["warm_status"], c["accepted_warm_request_wall_s"],
                c["decoder_calls"], c["canvases"],
                c["ledger_attempt0_decoder_calls"], c["ledger_warm_decoder_calls"],
                fp["B0"], fp["BO"], fp["A"], fp["D"], fp["H"], fp["model_calls"],
                fp["conservation_ok"], fp["BO_lt_B0"], fp["observation_frequency"],
            ])


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "%.*f" % (nd, x)
    return str(x)


def write_md(path, result):
    lines = []
    lines.append("# v24 bootstrap6 output-numerics audit")
    lines.append("")
    lines.append("Generated: %s" % result["generated_at"])
    lines.append("")
    lines.append("> " + HEADER_NOTE)
    lines.append("")

    lines.append("## Union audit result: **%s**" % result["union_verdict"])
    lines.append("")
    lines.append("- protocol_id: `%s`" % result["protocol_id"])
    lines.append("- protocol sha256: `%s`" % result["protocol_sha256"])
    lines.append("- cells total: %d, missing: %d, invalid: %d" % (result["n_cells_total"], result["n_missing"], result["n_invalid"]))
    if result["union_reasons"]:
        lines.append("- reasons:")
        for r in result["union_reasons"]:
            lines.append("  - %s" % r)
    else:
        lines.append("- no violations found: exactly one segment executed each cell, all host/gpu/cell_id/ledger cross-checks agree.")
    lines.append("")

    lines.append("## Lineage")
    lines.append("")
    for seg_name, seg in result["segments"].items():
        b = seg["binding"]
        lines.append("- **%s**: protocol sha256 `%s`, binding panel_protocol_sha256 matches: %s, source_commits: `%s`"
                      % (seg_name, seg["protocol_sha256"], b["matches_protocol_sha256"], b["source_commits"]))
    lines.append("")

    lines.append("## Per-dataset corrected table")
    lines.append("")
    for dataset in DATASETS:
        lines.append("### %s" % dataset)
        lines.append("")
        header = "| arm | correct/valid | failed | missing | unscored | invalid | calls | canvases | calls/canvas | B0 layer-calls | BO layer-calls | A layer-calls | D layer-calls | H layer-calls | model-calls (sum/5) |"
        sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for arm in ARMS:
            s = result["per_dataset_arm_summary"][dataset][arm]
            fp = s["five_phase_totals"]
            if fp and fp.get("n_runs"):
                b0, bo, a, d, h, mc = fp["B0"], fp["BO"], fp["A"], fp["D"], fp["H"], fp["model_calls_total"]
            else:
                b0 = bo = a = d = h = mc = None
            lines.append("| %s | %d/%d | %d | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                arm, s["n_correct"], s["n_scored_valid"], s["n_failed"], s["n_missing"], s["n_unscored"], s["n_invalid"],
                s["total_decoder_calls"], s["total_canvases"], _fmt(s["calls_per_canvas"], 2),
                _fmt(b0, 0), _fmt(bo, 0), _fmt(a, 0), _fmt(d, 0), _fmt(h, 0), _fmt(mc, 1),
            ))
        lines.append("")

    lines.append("## Estimand table (paired, same-host/gpu, accepted-warm cells)")
    lines.append("")
    header = ("| dataset | arm_x | arm_y | n_pairs | geoW | geoN | geoW/N | identity_ok | sum_calls_ratio | "
               "discordant x-correct/y-wrong | discordant x-wrong/y-correct |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for est in result["estimands"]:
        lines.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %d | %d |" % (
            est["dataset"], est["arm_x"], est["arm_y"], est["n_timing_pairs"],
            _fmt(est["geo_W_ratio"]), _fmt(est["geo_N_ratio"]), _fmt(est["geo_W_over_N_ratio"]),
            _fmt(est["geo_identity_ok"]), _fmt(est["sum_calls_ratio"]),
            est["discordant_x_correct_y_wrong"], est["discordant_x_wrong_y_correct"],
        ))
    lines.append("")

    lines.append("### Host-stratified W/N decomposition (host confounded with seed)")
    lines.append("")
    for est in result["estimands"]:
        if not est["host_stratified"]:
            continue
        lines.append("- **%s: %s vs %s**" % (est["dataset"], est["arm_x"], est["arm_y"]))
        for host, hs in est["host_stratified"].items():
            lines.append("  - host `%s` (n=%d): sum W ratio=%s, sum N ratio=%s (sumW_x=%s sumW_y=%s sumN_x=%s sumN_y=%s)"
                          % (host, hs["n_pairs"], _fmt(hs["sum_W_ratio"]), _fmt(hs["sum_N_ratio"]),
                             _fmt(hs["sum_W_x"], 2), _fmt(hs["sum_W_y"], 2), hs["sum_N_x"], hs["sum_N_y"]))
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, result, extra):
    payload = dict(result)
    payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True, default=str)
        f.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", action="append", required=True,
                         help="NAME=SCORED_CSV,LEDGER1,LEDGER2,PROTOCOL,BINDING (repeatable)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)

    for out_path in (args.out_json, args.out_md, args.out_csv):
        if os.path.exists(out_path):
            print("refusing to overwrite existing output: %s" % out_path, file=sys.stderr)
            return 2

    segments = []
    for seg_arg in args.segment:
        name, paths = parse_segment_arg(seg_arg)
        scored_csv, ledger1, ledger2, protocol_path, binding_path = paths
        segments.append(Segment(name, scored_csv, ledger1, ledger2, protocol_path, binding_path))

    if len(segments) < 2:
        print("need at least two --segment entries for a union audit", file=sys.stderr)
        return 2

    result = run_audit(segments)

    input_sha256 = {}
    for seg in segments:
        for p in seg.input_paths():
            if p not in input_sha256:
                input_sha256[p] = sha256_file(p)
    script_sha256 = sha256_file(os.path.abspath(__file__))

    write_json(args.out_json, result, {"input_sha256": input_sha256, "script_sha256": script_sha256, "script_path": os.path.abspath(__file__)})
    write_md(args.out_md, result)
    write_csv(args.out_csv, result["cells"])

    print("union_verdict=%s cells=%d missing=%d invalid=%d" % (
        result["union_verdict"], result["n_cells_total"], result["n_missing"], result["n_invalid"]))
    return 0 if result["union_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
