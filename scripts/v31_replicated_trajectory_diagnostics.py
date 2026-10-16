"""Descriptive trajectory diagnostics for completed, officially scored workloads.

Outcome-conditioned groups are post-treatment descriptions, not causal effects
or alternative primary estimands. No raw generation text is loaded.
"""
import argparse
import json
from pathlib import Path
import statistics

from scripts.v31_verify_replicated_results import read, require, sha, verify


METHODS = ("full_swap8", "residual_direction_guard", "residual",
           "full_swap8_direction_guard")
FIELDS = {"N": "denoise_forwards", "C": "canvases", "T": "output_tokens"}


def record_key(row):
    return f"{row['dataset']}|{row['index']}|{row['panel_seed']}|{row['repeat']}"


def outcome_group(control, candidate, full_credit):
    if all(x == full_credit for x in control + candidate):
        return "always_full_credit_both"
    if all(x == 0 for x in control + candidate):
        return "always_zero_both"
    delta = statistics.mean(candidate) - statistics.mean(control)
    if delta > 0:
        return "higher_mean_score"
    if delta < 0:
        return "lower_mean_score"
    return "equal_mixed_score"


def summarize_cells(cells):
    if not cells:
        return {"cells": 0}
    result = {"cells": len(cells)}
    for side in ("control", "candidate"):
        result[side] = {metric: statistics.mean(c[side][metric] for c in cells)
                        for metric in FIELDS}
    result["delta"] = {metric: result["candidate"][metric] - result["control"][metric]
                       for metric in FIELDS}
    result["N_changes"] = {
        "fewer": sum(c["candidate"]["N"] < c["control"]["N"] for c in cells),
        "equal": sum(c["candidate"]["N"] == c["control"]["N"] for c in cells),
        "more": sum(c["candidate"]["N"] > c["control"]["N"] for c in cells),
    }
    result["median_paired_delta_N"] = statistics.median(
        c["candidate"]["N"] - c["control"]["N"] for c in cells)
    control, candidate = result["control"], result["candidate"]
    ratio_c, ratio_v = control["N"] / control["C"], candidate["N"] / candidate["C"]
    canvas = (candidate["C"] - control["C"]) * (ratio_c + ratio_v) / 2
    per_canvas = (ratio_v - ratio_c) * (control["C"] + candidate["C"]) / 2
    require(abs(canvas + per_canvas - result["delta"]["N"]) < 1e-8,
            "trajectory product decomposition failed")
    result["symmetric_N_decomposition"] = {
        "canvas_count_term": canvas, "calls_per_canvas_term": per_canvas,
        "interpretation": "Exact aggregate product identity; not causal attribution",
    }
    return result


def diagnose(study, suite):
    verification = verify(study, suite)
    marker = next((study / "scoring" / suite).glob("attempt*/scoring_complete.json"))
    scores = read(marker.parent / "official.official.json")
    rows = {}
    for path in sorted((study / "shared_runs" / suite).glob(
            "shard*/repeat*/attempt*/exports/*/records.jsonl")):
        label = path.parent.name
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows.setdefault(label, {})[record_key(row)] = row
    full_credit = 100 if suite == "ruler128k" else 1
    keys = sorted(scores["r1_control_before"])
    control_labels = [f"r{rep}_control_{pos}" for rep in (1, 2)
                      for pos in ("before", "after")]
    output = {}
    for method in METHODS:
        candidate_labels = [f"r{rep}_{method}" for rep in (1, 2)]
        cells, groups = [], {}
        for key in keys:
            cell = {"key": key}
            for side, labels in (("control", control_labels),
                                 ("candidate", candidate_labels)):
                cell[side] = {metric: statistics.mean(rows[label][key][field] for label in labels)
                              for metric, field in FIELDS.items()}
            group = outcome_group([float(scores[x][key]) for x in control_labels],
                                  [float(scores[x][key]) for x in candidate_labels], full_credit)
            cell["outcome_group"] = group
            cells.append(cell)
            groups.setdefault(group, []).append(cell)
        ordered = sorted(cells, key=lambda c: c["candidate"]["N"] - c["control"]["N"])
        net_reduction = sum(c["control"]["N"] - c["candidate"]["N"] for c in cells)
        top_reduction = sum(c["control"]["N"] - c["candidate"]["N"] for c in ordered[:5])
        rr = [rows[label][key] for label in candidate_labels for key in keys]
        work = [r["receipts"]["adapter"].get("value_refinement_work", {}) for r in rr]
        anchors = sum(w.get("anchor_events", 0) for w in work)
        held = sum(w.get("held_calls", 0) for w in work)
        output[method] = {
            "unconditional": summarize_cells(cells),
            "outcome_conditioned": {group: summarize_cells(subset) for group, subset in sorted(groups.items())},
            "tail_sensitivity": {
                "net_sum_of_cell_mean_N_reductions": net_reduction,
                "top5_sum_of_N_reductions": top_reduction,
                "top5_share_of_net_saving": top_reduction / net_reduction if net_reduction > 0 else None,
                "top5_cells": ordered[:5],
                "excluding_top5": summarize_cells(ordered[5:]),
                "interpretation": "Post-hoc sensitivity, not a new primary estimate",
            },
            "intervention_coverage": {
                "trials": len(rr), "trials_with_anchor": sum(w.get("anchor_events", 0) > 0 for w in work),
                "trials_with_swaps": sum(w.get("swapped_blocks", 0) > 0 for w in work),
                "trials_with_correction": sum(w.get("corrected_calls", 0) > 0 for w in work),
                "anchor_events": anchors, "held_calls": held,
                "held_calls_per_anchor": held / anchors if anchors else None,
                "corrected_calls": sum(w.get("corrected_calls", 0) for w in work),
                "interpretation": "Receipt exposure counts, not causal treatment effects",
            },
        }
    return {
        "status": "complete_descriptive_diagnostic", "suite": suite,
        "verification": verification, "analysis_source_sha256": sha(Path(__file__)),
        "aggregation": "Average two candidate trials and four control bookends within each frozen input/seed cell",
        "inference": "Outcome conditioning and tail trimming are post-treatment, descriptive, and selection-biased",
        "methods": output, "timing": "paused",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.study.resolve(), args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        require(read(args.output) == result, "preserve existing diagnostic")
    else:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({"status": result["status"], "suite": args.suite, "methods": list(result["methods"])}))
