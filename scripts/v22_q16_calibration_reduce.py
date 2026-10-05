"""Pure-Python (stdlib only) reducer for the v22 matched-Q16 calibration study.

Combines two cross-host geometry-capture receipts (one per calibration state,
labelled by the CLI e.g. ``mpk`` and ``dllm``) into a single reduced report
covering:

  * a pooled (sum-of-squares) relative-L2 error across the two states, never
    a naive average of per-state relative errors;
  * aggregate retained legal-pair "work" across the two states;
  * two frozen selection rules over the five calibration offsets recorded in
    the LongBench call-3 / layer-5 ``matched_q16_calibration`` block;
  * a descriptive (not a selection rule) log-linear interpolation of the
    pooled error ratio at work ratio == 1.0;
  * a bootstrap fill table over every group's call-0 layers 0 and 5; and
  * per-state calibration rows for CSV/JSON export.

This script performs no averaging shortcuts and raises rather than silently
substituting a value whenever a required invariant does not hold (mismatched
source_commit, a zero denominator, an unexpected offsets list, etc).

Ambiguity note on the CSV "two pooled rows per point" instruction: the two
pooled rows per point are tagged via the ``pooled_kind`` column as
``error`` (pooled relative_l2/error_ratio computed from part B, sum-of-squares)
and ``work`` (aggregate retained pairs/work_ratio computed from part C,
sum-of-counts). Row-level distributional fields (row p99/max, removed-mass
p99/max) are never pooled by simple sum/average and are left blank (None) on
both pooled rows; only per-host rows carry them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_OFFSETS = [0.0, -0.25, -0.5, -1.0, -2.0]
REQUIRED_DATASETS = {"aime26", "longbench_v2", "ruler4k"}
ERROR_BAND = 1.05
PER_STATE_CAP = 1.15


# ----------------------------------------------------------------------
# Hashing / IO helpers
# ----------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_receipt(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Structural navigation helpers
# ----------------------------------------------------------------------

def find_dataset_group(receipt, dataset):
    """Return (key, group) for the single group whose dataset == `dataset`.

    Raises ValueError unless exactly one such group exists.
    """
    groups = receipt.get("groups", {})
    matches = [(k, g) for k, g in groups.items() if g.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one group with dataset={dataset!r}, found {len(matches)}"
        )
    return matches[0]


def get_layer(group, call_idx, layer_idx):
    call = group["calls"][str(call_idx)]
    return call["layers"][str(layer_idx)]


def get_matched_calibration(lb_group):
    """Return the matched_q16_calibration block for LB call-3 / layer-5."""
    layer5 = get_layer(lb_group, 3, 5)
    comparison = layer5["geometry"]["comparison"]
    if "matched_q16_calibration" not in comparison:
        raise ValueError("longbench_v2 call-3 layer-5 has no matched_q16_calibration block")
    return comparison["matched_q16_calibration"]


def candidate_for_offset(mq, offset):
    for c in mq["q16_candidates"]:
        if c["offset"] == offset:
            return c
    raise ValueError(f"no q16 candidate found for offset={offset!r}")


# ----------------------------------------------------------------------
# Part A: verification
# ----------------------------------------------------------------------

def verify_receipts(receipts: dict):
    """receipts: {label: receipt_dict}. Returns (source_commit, {label: (group_key, group)}, {label: mq})."""
    if len(receipts) < 1:
        raise ValueError("no receipts supplied")

    source_commits = set()
    lb_groups = {}
    mqs = {}
    inherited_thresholds = set()

    for label, receipt in receipts.items():
        completeness = receipt.get("completeness", {})
        if completeness.get("status") != "complete":
            raise ValueError(f"receipt {label!r}: completeness.status != 'complete'")

        qual = receipt.get("qualification_tests", {})
        if qual.get("status") != "passed":
            raise ValueError(f"receipt {label!r}: qualification_tests.status != 'passed'")

        source_commits.add(receipt.get("source_commit"))

        datasets_present = {g.get("dataset") for g in receipt.get("groups", {}).values()}
        missing = REQUIRED_DATASETS - datasets_present
        if missing:
            raise ValueError(f"receipt {label!r}: missing required dataset groups {missing}")

        lb_key, lb_group = find_dataset_group(receipt, "longbench_v2")
        lb_groups[label] = (lb_key, lb_group)

        mq = get_matched_calibration(lb_group)
        mqs[label] = mq

        if mq.get("offsets") != EXPECTED_OFFSETS:
            raise ValueError(
                f"receipt {label!r}: offsets {mq.get('offsets')!r} != expected {EXPECTED_OFFSETS!r}"
            )

        inherited_thresholds.add(mq.get("inherited_threshold"))

    if len(source_commits) != 1:
        raise ValueError(f"receipts do not share one source_commit: {source_commits!r}")

    if len(inherited_thresholds) != 1:
        raise ValueError(
            f"receipts do not share one inherited_threshold: {inherited_thresholds!r}"
        )

    (source_commit,) = source_commits
    return source_commit, lb_groups, mqs


# ----------------------------------------------------------------------
# Part B: pooled error
# ----------------------------------------------------------------------

def pooled_relative_l2(abs_errors, references):
    """E = sqrt(sum(abs_err_i^2) / sum(ref_i^2)). Raises on zero denominator.

    Never a substitute for, or average of, the per-state relative errors.
    """
    if len(abs_errors) != len(references):
        raise ValueError("abs_errors and references must be the same length")
    denom = sum(r * r for r in references)
    if denom == 0:
        raise ValueError("sum of squared references is zero; refusing to divide")
    numer = sum(e * e for e in abs_errors)
    return math.sqrt(numer / denom)


# ----------------------------------------------------------------------
# Part C: aggregate work
# ----------------------------------------------------------------------

def aggregate_work(retained_pairs):
    return sum(retained_pairs)


# ----------------------------------------------------------------------
# Part D: per-state error ratio
# ----------------------------------------------------------------------

def per_state_error_ratio(relative_l2_point, relative_l2_coarse):
    if relative_l2_coarse == 0:
        raise ValueError("coarse relative_l2 is zero; refusing to divide")
    return relative_l2_point / relative_l2_coarse


# ----------------------------------------------------------------------
# Part E: Rule 1 (work-matching)
# ----------------------------------------------------------------------

def rule1_select(offsets, work_by_offset, work_coarse):
    """Offset whose aggregate W is closest to coarse W; tie -> larger W."""
    best_offset = None
    best_key = None
    for off in offsets:
        w = work_by_offset[off]
        mismatch = w - work_coarse
        distance = abs(mismatch)
        # sort key: minimize distance, then (on exact tie) prefer larger w
        key = (distance, -w)
        if best_key is None or key < best_key:
            best_key = key
            best_offset = off
    mismatch = work_by_offset[best_offset] - work_coarse
    ratio = work_by_offset[best_offset] / work_coarse if work_coarse != 0 else None
    return {
        "offset": best_offset,
        "work_mismatch": mismatch,
        "work_ratio": ratio,
    }


# ----------------------------------------------------------------------
# Part F: Rule 2 (minimum retained work under error caps)
# ----------------------------------------------------------------------

def rule2_select(offsets, pooled_error_ratio_by_offset, per_state_ratio_by_offset,
                  work_by_offset, error_band=ERROR_BAND, per_state_cap=PER_STATE_CAP):
    """Among offsets meeting the pooled+per-state error caps, choose the
    minimum aggregate W; tie -> larger W. Returns None if nothing qualifies.

    per_state_ratio_by_offset: {offset: {label: ratio}}
    """
    qualifying = []
    for off in offsets:
        pooled_ratio = pooled_error_ratio_by_offset[off]
        if pooled_ratio > error_band:
            continue
        per_state = per_state_ratio_by_offset[off]
        if any(r > per_state_cap for r in per_state.values()):
            continue
        qualifying.append(off)

    if not qualifying:
        return None

    # choose minimum W; tie -> larger W (a no-op when truly tied); use
    # |offset| ascending as a final deterministic tiebreak beyond spec.
    def key(off):
        w = work_by_offset[off]
        return (w, -w, abs(off))

    best_offset = min(qualifying, key=key)
    return {
        "offset": best_offset,
        "work": work_by_offset[best_offset],
        "pooled_error_ratio": pooled_error_ratio_by_offset[best_offset],
    }


# ----------------------------------------------------------------------
# Part G: nonmonotonic detection
# ----------------------------------------------------------------------

def nonmonotonic_pairs(offsets, work_by_offset):
    """List consecutive offsets (in the given, offset-descending order)
    where aggregate W decreases as the offset decreases."""
    out = []
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        wa, wb = work_by_offset[a], work_by_offset[b]
        if wb < wa:
            out.append({
                "from_offset": a,
                "to_offset": b,
                "from_work": wa,
                "to_work": wb,
            })
    return out


# ----------------------------------------------------------------------
# Part H: descriptive log-linear interpolation at work ratio == 1.0
# ----------------------------------------------------------------------

def interpolate_error_ratio_at_work_ratio_one(offsets, work_ratio_by_offset, error_ratio_by_offset):
    """Descriptive only -- NOT a measured value, NOT a selection rule.

    Finds the two *measured* offsets (adjacent in the given order) whose
    work ratios bracket 1.0, and log-linearly interpolates the pooled error
    ratio at work_ratio == 1.0 (linear in ln(work_ratio)). Returns None if no
    adjacent pair brackets 1.0.
    """
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        wa, wb = work_ratio_by_offset[a], work_ratio_by_offset[b]
        if wa == 1.0:
            return {
                "bracket_offsets": [a, a],
                "interpolated_error_ratio": error_ratio_by_offset[a],
            }
        if (wa - 1.0) * (wb - 1.0) <= 0:
            xa, xb = math.log(wa), math.log(wb)
            if xa == xb:
                continue
            ya, yb = error_ratio_by_offset[a], error_ratio_by_offset[b]
            t = (0.0 - xa) / (xb - xa)
            interp = ya + t * (yb - ya)
            return {
                "bracket_offsets": [a, b],
                "interpolated_error_ratio": interp,
            }
    return None


# ----------------------------------------------------------------------
# Part I: bootstrap fill table (call 0, layers 0 and 5, every group)
# ----------------------------------------------------------------------

def build_bootstrap_table(receipts_by_label):
    rows = []
    for label, receipt in receipts_by_label.items():
        for dataset in sorted(REQUIRED_DATASETS):
            _key, group = find_dataset_group(receipt, dataset)
            for layer_idx in ("0", "5"):
                layer = get_layer(group, 0, layer_idx)
                geometry = layer["geometry"]
                comparison = geometry["comparison"]
                parity = comparison.get("coarse_route_only_parity", {})
                geometries = {}
                for geom_name, geom_val in comparison.get("independent_sequential", {}).items():
                    ovf = geom_val["output_vs_full_fp32"]
                    row_stats = geom_val["output_relative_l2_per_row"]
                    removed_mass = geom_val["removed_mass_per_row"]
                    geometries[geom_name] = {
                        "kept_legal_pair_fraction": geom_val["kept_legal_pair_fraction"],
                        "relative_l2": ovf["relative_l2"],
                        "row_p99": row_stats["p99"],
                        "row_max": row_stats["max"],
                        "removed_mass_row_p99": removed_mass["p99"],
                    }
                rows.append({
                    "dataset": dataset,
                    "host": label,
                    "layer": layer_idx,
                    "layer_kind": layer.get("attention_kind"),
                    "query_sensitivity": geometry.get("query_sensitivity"),
                    "parity_status": parity.get("status"),
                    "geometries": geometries,
                })
    return rows


# ----------------------------------------------------------------------
# Part J: per-state calibration rows
# ----------------------------------------------------------------------

def build_calibration_rows(lb_groups, mqs):
    """Per-host (per-state) rows: one 'coarse' row + one row per offset."""
    rows = []
    for label, mq in mqs.items():
        group_key, _group = lb_groups[label]
        denom = mq["denominator_legal_pairs"]
        threshold_coarse = mq["inherited_threshold"]
        cb = mq["coarse_baseline"]
        retained_coarse = cb["retained_legal_pairs"]
        relative_l2_coarse = cb["output_vs_full_fp32"]["relative_l2"]

        def make_row(point_label, offset, threshold, retained, relative_l2,
                     row_p99, row_max, removed_p99, removed_max, bitmap_bytes,
                     near_threshold):
            return {
                "host": label,
                "group_key": group_key,
                "point": point_label,
                "offset": offset,
                "threshold": threshold,
                "retained_legal_pairs": retained,
                "retained_fraction_of_denominator": retained / denom,
                "work_ratio_to_state_coarse": retained / retained_coarse,
                "relative_l2": relative_l2,
                "error_ratio_to_state_coarse": per_state_error_ratio(relative_l2, relative_l2_coarse),
                "row_p99": row_p99,
                "row_max": row_max,
                "removed_mass_row_p99": removed_p99,
                "removed_mass_row_max": removed_max,
                "bitmap_bytes": bitmap_bytes,
                "near_threshold_groups": near_threshold,
            }

        rows.append(make_row(
            "coarse", None, threshold_coarse, retained_coarse, relative_l2_coarse,
            cb["output_relative_l2_per_row"]["p99"], cb["output_relative_l2_per_row"]["max"],
            cb["removed_mass_per_row"]["p99"], cb["removed_mass_per_row"]["max"],
            cb["bitmap_bytes"], cb["risk_groups_within_1e_3_of_threshold"],
        ))

        for off in mq["offsets"]:
            c = candidate_for_offset(mq, off)
            rows.append(make_row(
                "offset", off, c["threshold"], c["retained_legal_pairs"],
                c["output_vs_full_fp32"]["relative_l2"],
                c["output_relative_l2_per_row"]["p99"], c["output_relative_l2_per_row"]["max"],
                c["removed_mass_per_row"]["p99"], c["removed_mass_per_row"]["max"],
                c["bitmap_bytes"], c["risk_groups_within_1e_3_of_threshold"],
            ))
    return rows


# ----------------------------------------------------------------------
# Cross-state point summary (drives E/F/G/H and the pooled CSV rows)
# ----------------------------------------------------------------------

def build_point_summary(mqs):
    labels = sorted(mqs.keys())
    any_mq = mqs[labels[0]]
    offsets = any_mq["offsets"]
    threshold_coarse = any_mq["inherited_threshold"]

    def point_fields(point_label, offset):
        abs_errors, references, relative_l2s, retained = [], [], {}, {}
        threshold = None
        for label in labels:
            mq = mqs[label]
            if point_label == "coarse":
                d = mq["coarse_baseline"]
                threshold = mq["inherited_threshold"]
            else:
                d = candidate_for_offset(mq, offset)
                threshold = d["threshold"]
            ovf = d["output_vs_full_fp32"]
            abs_errors.append(ovf["absolute_error_l2"])
            references.append(ovf["reference_l2"])
            relative_l2s[label] = ovf["relative_l2"]
            retained[label] = d["retained_legal_pairs"]
        return {
            "threshold": threshold,
            "abs_errors": abs_errors,
            "references": references,
            "relative_l2": relative_l2s,
            "retained": retained,
        }

    coarse = point_fields("coarse", None)
    E_coarse = pooled_relative_l2(coarse["abs_errors"], coarse["references"])
    W_coarse = aggregate_work(list(coarse["retained"].values()))

    points = {
        "coarse": {
            "threshold": coarse["threshold"],
            "pooled_error": E_coarse,
            "aggregate_work": W_coarse,
            "pooled_error_ratio_to_coarse": 1.0,
            "work_ratio_to_coarse": 1.0,
            "per_state_relative_l2": coarse["relative_l2"],
            "per_state_error_ratio_to_coarse": {l: 1.0 for l in labels},
            "per_state_retained": coarse["retained"],
        }
    }

    work_by_offset = {}
    pooled_error_ratio_by_offset = {}
    per_state_ratio_by_offset = {}
    error_ratio_by_offset = {}
    work_ratio_by_offset = {}

    for off in offsets:
        p = point_fields("offset", off)
        E_off = pooled_relative_l2(p["abs_errors"], p["references"])
        W_off = aggregate_work(list(p["retained"].values()))
        pooled_ratio = per_state_error_ratio(E_off, E_coarse)
        per_state_ratio = {
            label: per_state_error_ratio(p["relative_l2"][label], coarse["relative_l2"][label])
            for label in labels
        }
        points[off] = {
            "threshold": p["threshold"],
            "pooled_error": E_off,
            "aggregate_work": W_off,
            "pooled_error_ratio_to_coarse": pooled_ratio,
            "work_ratio_to_coarse": W_off / W_coarse,
            "per_state_relative_l2": p["relative_l2"],
            "per_state_error_ratio_to_coarse": per_state_ratio,
            "per_state_retained": p["retained"],
        }
        work_by_offset[off] = W_off
        pooled_error_ratio_by_offset[off] = pooled_ratio
        per_state_ratio_by_offset[off] = per_state_ratio
        error_ratio_by_offset[off] = pooled_ratio
        work_ratio_by_offset[off] = W_off / W_coarse

    rule1 = rule1_select(offsets, work_by_offset, W_coarse)
    rule2 = rule2_select(offsets, pooled_error_ratio_by_offset, per_state_ratio_by_offset, work_by_offset)
    nonmono = nonmonotonic_pairs(offsets, work_by_offset)
    interp = interpolate_error_ratio_at_work_ratio_one(offsets, work_ratio_by_offset, error_ratio_by_offset)

    return {
        "labels": labels,
        "offsets": offsets,
        "threshold_coarse": threshold_coarse,
        "points": points,
        "rule1_work_matched": rule1,
        "rule2_min_work_under_caps": rule2,
        "nonmonotonic": nonmono,
        "descriptive_interpolation_not_measured": interp,
    }


# ----------------------------------------------------------------------
# CSV assembly
# ----------------------------------------------------------------------

CSV_FIELDS = [
    "host", "group_key", "point", "offset", "threshold",
    "retained_legal_pairs", "retained_fraction_of_denominator",
    "work_ratio", "relative_l2", "error_ratio",
    "row_p99", "row_max", "removed_mass_row_p99", "removed_mass_row_max",
    "bitmap_bytes", "near_threshold_groups", "pooled_kind",
]


def build_csv_rows(calibration_rows, point_summary):
    rows = []
    for r in calibration_rows:
        rows.append({
            "host": r["host"],
            "group_key": r["group_key"],
            "point": r["point"],
            "offset": r["offset"] if r["offset"] is not None else "coarse",
            "threshold": r["threshold"],
            "retained_legal_pairs": r["retained_legal_pairs"],
            "retained_fraction_of_denominator": r["retained_fraction_of_denominator"],
            "work_ratio": r["work_ratio_to_state_coarse"],
            "relative_l2": r["relative_l2"],
            "error_ratio": r["error_ratio_to_state_coarse"],
            "row_p99": r["row_p99"],
            "row_max": r["row_max"],
            "removed_mass_row_p99": r["removed_mass_row_p99"],
            "removed_mass_row_max": r["removed_mass_row_max"],
            "bitmap_bytes": r["bitmap_bytes"],
            "near_threshold_groups": r["near_threshold_groups"],
            "pooled_kind": "",
        })

    point_keys = ["coarse"] + point_summary["offsets"]
    for pk in point_keys:
        point = point_summary["points"][pk]
        offset_label = "coarse" if pk == "coarse" else pk
        rows.append({
            "host": "pooled", "group_key": "pooled", "point": "coarse" if pk == "coarse" else "offset",
            "offset": offset_label, "threshold": point["threshold"],
            "retained_legal_pairs": "", "retained_fraction_of_denominator": "",
            "work_ratio": "", "relative_l2": point["pooled_error"],
            "error_ratio": point["pooled_error_ratio_to_coarse"],
            "row_p99": "", "row_max": "", "removed_mass_row_p99": "", "removed_mass_row_max": "",
            "bitmap_bytes": "", "near_threshold_groups": "",
            "pooled_kind": "error",
        })
        rows.append({
            "host": "pooled", "group_key": "pooled", "point": "coarse" if pk == "coarse" else "offset",
            "offset": offset_label, "threshold": point["threshold"],
            "retained_legal_pairs": point["aggregate_work"], "retained_fraction_of_denominator": "",
            "work_ratio": point["work_ratio_to_coarse"], "relative_l2": "", "error_ratio": "",
            "row_p99": "", "row_max": "", "removed_mass_row_p99": "", "removed_mass_row_max": "",
            "bitmap_bytes": "", "near_threshold_groups": "",
            "pooled_kind": "work",
        })
    return rows


# ----------------------------------------------------------------------
# Top-level reduce
# ----------------------------------------------------------------------

def reduce_receipts(receipts_by_label: dict, reducer_source_path):
    source_commit, lb_groups, mqs = verify_receipts(receipts_by_label)
    point_summary = build_point_summary(mqs)
    bootstrap_rows = build_bootstrap_table(receipts_by_label)
    calibration_rows = build_calibration_rows(lb_groups, mqs)
    csv_rows = build_csv_rows(calibration_rows, point_summary)

    inputs = {}
    for label, receipt in receipts_by_label.items():
        inputs[label] = {
            "sha256": receipt.get("__source_sha256__"),
        }

    output = {
        "schema": "v22_q16_calibration_reduce_v1",
        "source_commit": source_commit,
        "reducer_sha256": sha256_file(reducer_source_path) if reducer_source_path else None,
        "inputs": inputs,
        "longbench_group_keys": {label: key for label, (key, _g) in lb_groups.items()},
        "longbench": point_summary,
        "bootstrap_fill_table": bootstrap_rows,
        "calibration_rows": calibration_rows,
        "quality_eligible": False,
        "note": (
            "These are diagnostic FP32 attention-output measurements on two "
            "answer-blind calibration states, not task quality."
        ),
    }
    return output, csv_rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_receipt_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--receipt must be LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--receipt must be LABEL=PATH, got {value!r}")
    return label, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", type=parse_receipt_arg, required=True,
                         help="LABEL=PATH, may be given multiple times")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)

    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)
    for p in (out_json_path, out_csv_path):
        if p.exists():
            print(f"refusing to overwrite existing output: {p}", file=sys.stderr)
            return 2

    receipts_by_label = {}
    for label, path in args.receipt:
        receipt = load_receipt(path)
        receipt["__source_sha256__"] = sha256_file(path)
        receipts_by_label[label] = receipt

    try:
        output, csv_rows = reduce_receipts(receipts_by_label, reducer_source_path=__file__)
    except ValueError as exc:
        print(f"reduction failed: {exc}", file=sys.stderr)
        return 1

    # strip the bookkeeping key we stashed on the receipt dict before any
    # accidental leakage; inputs are recorded via reduce_receipts already.
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        # sort_keys=False: the "points" mapping mixes a string key ("coarse")
        # with float offset keys; json.dump converts non-string keys to
        # strings during encoding but only *after* an unsorted pass, so
        # sort_keys=True would fail trying to compare float and str keys.
        json.dump(output, f, indent=2, sort_keys=False)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"wrote {out_json_path} and {out_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
