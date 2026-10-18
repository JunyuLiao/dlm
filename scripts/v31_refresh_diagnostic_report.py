"""Independent, CPU-only analysis of the bounded refresh diagnostic.

The intervention crosses CURRENT dense/sparse output (the sparse output uses
the old common support) with FUTURE fresh/held support.  Neither axis denotes
a cheap selector.  Instrumented wall times are diagnostic and are never
reported as deployable speedups.  Bootstrap units are questions, with all
seeds, engines, labels, and control bookends kept together.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


FACTORIAL_LABELS = ("dense_fresh", "dense_hold", "sparse_fresh", "sparse_hold")
CONTROL_LABELS = ("control_before", "control_after")
REQUIRED_LABELS = FACTORIAL_LABELS + CONTROL_LABELS
METRICS = ("score", "denoise_forwards", "canvases", "output_tokens", "wall_s", "decode_s")
CONTRASTS = {
    "sparse_minus_dense": {"sparse_fresh": .5, "sparse_hold": .5, "dense_fresh": -.5, "dense_hold": -.5},
    "fresh_minus_hold": {"dense_fresh": .5, "sparse_fresh": .5, "dense_hold": -.5, "sparse_hold": -.5},
    "interaction": {"sparse_fresh": 1., "sparse_hold": -1., "dense_fresh": -1., "dense_hold": 1.},
    "sparse_minus_dense_fresh": {"sparse_fresh": 1., "dense_fresh": -1.},
    "sparse_minus_dense_hold": {"sparse_hold": 1., "dense_hold": -1.},
    "fresh_minus_hold_dense": {"dense_fresh": 1., "dense_hold": -1.},
    "fresh_minus_hold_sparse": {"sparse_fresh": 1., "sparse_hold": -1.},
    "control_after_minus_before": {"control_after": 1., "control_before": -1.},
    **{f"{label}_minus_control_mean": {label: 1., "control_before": -.5, "control_after": -.5}
       for label in FACTORIAL_LABELS},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["suite"]), int(row["index"]), int(row["panel_seed"]), int(row["engine"]))


def _identity(key: tuple[str, int, int, int]) -> dict[str, Any]:
    return dict(zip(("suite", "index", "panel_seed", "engine"), key))


def _receipt(row: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = row.get("diagnostic_receipt", row.get("diagnostic", row.get("receipt", {})))
    if not isinstance(receipt, Mapping):
        return {}
    nested = receipt.get("refresh_diagnostic", receipt)
    return nested if isinstance(nested, Mapping) else {}


def _exposure(row: Mapping[str, Any]) -> bool | None:
    value = _receipt(row).get("exposed")
    return value if isinstance(value, bool) else None


def _prefix_status(group: Mapping[str, Mapping[str, Any]], labels: Sequence[str] = FACTORIAL_LABELS) -> str:
    """Do not equate missing proof with a matched prefix."""
    signatures = [_receipt(group[label]).get("prefix_signature") for label in labels]
    if any(not isinstance(value, str) or not value for value in signatures):
        return "unverified"
    return "matched" if len(set(signatures)) == 1 else "mismatched"


def _group_exposure(group: Mapping[str, Mapping[str, Any]]) -> str:
    values = [_exposure(group[label]) for label in FACTORIAL_LABELS]
    if any(value is None for value in values):
        return "unverified"
    if all(values):
        return "exposed"
    if not any(values):
        return "not_exposed"
    return "inconsistent"


def _validate(row: Mapping[str, Any]) -> dict[str, Any]:
    needed = ("suite", "index", "panel_seed", "engine", "label", "finish_reason", "output_hash") + METRICS
    _require(all(field in row for field in needed), "trial missing required fields")
    _require(isinstance(row["suite"], str) and bool(row["suite"]), "suite must be a nonempty string")
    _require(isinstance(row["label"], str) and bool(row["label"]), "label must be a nonempty string")
    for field in ("index", "panel_seed", "engine"):
        _require(isinstance(row[field], int) and not isinstance(row[field], bool), f"{field} must be an integer")
    _require(isinstance(row["output_hash"], str) and bool(row["output_hash"]), "output_hash must be a nonempty string")
    for field in METRICS:
        value = row[field]
        _require(isinstance(value, (int, float)) and math.isfinite(value), f"nonfinite/nonnumeric {field}")
        if field != "score":
            _require(value >= 0, f"negative {field}")
    for field in ("denoise_forwards", "canvases", "output_tokens"):
        _require(int(row[field]) == row[field], f"nonintegral {field}")
    # Deliberately copy only permitted numeric/structural fields: callers may
    # load scored records that also contain private prompts or generations.
    return {field: row[field] for field in needed} | {
        "diagnostic_receipt": dict(_receipt(row)),
        "post_event_states": row.get("post_event_states", row.get("refresh_post_event_states")) or [],
    }


def _intervention_status(group: Mapping[str, Mapping[str, Any]]) -> str:
    """The main study is all-five-GLOBAL; qualification has a separate phase."""
    missing = False
    for label in FACTORIAL_LABELS:
        receipt = _receipt(group[label])
        for field, expected in (("complete_scope", True), ("errors", []),
                                ("added_model_forwards", 0), ("added_sampler_calls", 0),
                                ("scope", [5, 11, 17, 23, 29])):
            if field not in receipt:
                missing = True
            elif receipt[field] != expected:
                return "failed"
    return "unverified" if missing else "passed"


def _state_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a, b = left.get("post_event_states", []), right.get("post_event_states", [])
    if not a or not b:
        return {"status": "unavailable", "compared_calls": 0}
    fields = ("call", "canvas", "argmax", "step", "encoder", "confident", "cuda_rng", "cpu_rng", "self_conditioning")
    first: dict[str, int] = {}
    compared = Counter()
    for offset, (x, y) in enumerate(zip(a, b), 1):
        for field in fields:
            if field in x and field in y:
                compared[field] += 1
                if x[field] != y[field] and field not in first:
                    first[field] = offset
    return {"status": "compared", "compared_calls": min(len(a), len(b)),
            "left_calls_recorded": len(a), "right_calls_recorded": len(b),
            "first_different_relative_call": first, "field_comparisons": dict(compared),
            "scope": "up to the recorded post-event calls; absence of a difference is not whole-trajectory equality"}


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {field: sum(row[field] for row in rows) for field in METRICS}
    return {
        "trials": len(rows),
        "means": {field: _mean(row[field] for row in rows) for field in METRICS},
        "totals": totals,
        "calls_per_canvas_ratio_of_totals": totals["denoise_forwards"] / totals["canvases"] if totals["canvases"] else None,
        "decode_per_call_amortized_s": totals["decode_s"] / totals["denoise_forwards"] if totals["denoise_forwards"] else None,
        "finish_reasons": dict(sorted(Counter(str(row["finish_reason"]) for row in rows).items())),
        "exposure": dict(Counter("exposed" if _exposure(row) else "not_exposed" if _exposure(row) is False else "unverified" for row in rows)),
    }


def _effect_rows(groups: Sequence[tuple[tuple[str, int, int, int], Mapping[str, Mapping[str, Any]]]],
                 weights: Mapping[str, float]) -> list[dict[str, Any]]:
    return [_identity(key) | {
        field: sum(weight * group[label][field] for label, weight in weights.items())
        for field in METRICS
    } for key, group in groups]


def _point_effect(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"paired_groups": len(rows), "question_clusters": len({row["index"] for row in rows}),
            "means": {field: _mean(row[field] for row in rows) for field in METRICS}}


def _effect_summary(rows: Sequence[Mapping[str, Any]], bootstrap_samples: int,
                    bootstrap_seed: int) -> dict[str, Any]:
    """Cluster resampling pools every retained group belonging to a question.

    This is the complete-paired-group estimand. Unequal cluster sizes are not
    silently converted to question-equal weighting. The denominator is
    resampled with the numerator, preserving the original estimator.
    """
    by_question: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["index"]].append(row)
    questions = sorted(by_question)
    sums = {index: {field: sum(row[field] for row in by_question[index]) for field in METRICS}
            for index in questions}
    draws: dict[str, list[float]] = {field: [] for field in METRICS}
    rng = random.Random(bootstrap_seed)
    if len(questions) >= 2:
        for _ in range(bootstrap_samples):
            sample = rng.choices(questions, k=len(questions))
            count = sum(len(by_question[index]) for index in sample)
            for field in METRICS:
                draws[field].append(sum(sums[index][field] for index in sample) / count)
    result = _point_effect(rows)
    result["bootstrap"] = {
        "unit": "question; all seeds and engines remain together",
        "samples": bootstrap_samples if len(questions) >= 2 else 0,
        "seed": bootstrap_seed,
        "ci95": {field: [_quantile(draws[field], .025), _quantile(draws[field], .975)] for field in METRICS},
        "interpretation": "descriptive percentile interval; no superiority, equivalence, or noninferiority claim",
    }
    result["by_seed"] = {str(seed): _point_effect([row for row in rows if row["panel_seed"] == seed])
                         for seed in sorted({row["panel_seed"] for row in rows})}
    result["by_engine"] = {str(engine): _point_effect([row for row in rows if row["engine"] == engine])
                           for engine in sorted({row["engine"] for row in rows})}
    result["question_effects"] = [dict(index=index, paired_groups=len(by_question[index]),
                                      means={field: sums[index][field] / len(by_question[index]) for field in METRICS})
                                    for index in questions]
    result["leave_one_question_out"] = [dict(omitted_index=index, **_point_effect([row for row in rows if row["index"] != index]))
                                         for index in questions]
    result["tails"] = {}
    for field in METRICS:
        total = sum(row[field] for row in rows)
        ranked = sorted(questions, key=lambda index: (-abs(sums[index][field]), index))
        result["tails"][field] = {
            "paired_delta_min": min((row[field] for row in rows), default=None),
            "paired_delta_max": max((row[field] for row in rows), default=None),
            "paired_absolute_delta_p95": _quantile([abs(row[field]) for row in rows], .95),
            "largest_absolute_question_contributions": [
                {"index": index, "sum_paired_delta": sums[index][field],
                 "signed_share_of_net_delta": sums[index][field] / total if total else None}
                for index in ranked[:5]],
            "share_note": "Signed shares may exceed one under cancellation; no observations are trimmed.",
        }
    return result


def summarize_trials(trials: Iterable[Mapping[str, Any]], *, bootstrap_samples: int = 10000,
                     bootstrap_seed: int = 20261018,
                     expected_cells: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize a bounded factorial without dropping adverse outcomes.

    expected_cells, when supplied, contains suite/index/panel_seed/engine.
    Additional labels (e.g. fixed16 headroom) are summarized separately and do
    not enter factorial contrasts. Missing groups/labels are reported, not
    mistaken for a completed experiment. Missing prefix signatures mean the
    causal pairing gate is unverified, never passed.
    """
    _require(isinstance(bootstrap_samples, int) and bootstrap_samples >= 0, "invalid bootstrap_samples")
    raw_rows = list(trials)
    excluded_phases = Counter(str(row.get("phase")) for row in raw_rows if row.get("phase", "factorial") != "factorial")
    rows = [_validate(row) for row in raw_rows if row.get("phase", "factorial") == "factorial"]
    _require(bool(rows), "no trials")
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    other: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = _key(row)
        identity = key + (row["label"],)
        _require(identity not in seen, f"duplicate trial {identity}")
        seen.add(identity)
        if row["label"] in REQUIRED_LABELS:
            groups[key][row["label"]] = row
        else:
            other[row["suite"]].append(row)
    expected = {_key(cell) for cell in expected_cells} if expected_cells is not None else set(groups)
    _require(not (set(groups) - expected), "observed factorial cell not in expected_cells")
    all_keys = sorted(expected | set(groups))
    missing = [_identity(key) | {"missing_labels": sorted(set(REQUIRED_LABELS) - set(groups.get(key, {})))}
               for key in all_keys if set(groups.get(key, {})) != set(REQUIRED_LABELS)]
    complete = [(key, groups[key]) for key in all_keys if set(groups.get(key, {})) == set(REQUIRED_LABELS)]
    status_counts = Counter(_prefix_status(group) for _, group in complete)
    exposure_counts = Counter(_group_exposure(group) for _, group in complete)
    integrity_counts = Counter(_intervention_status(group) for _, group in complete)
    all_prefix_ok = bool(complete) and not missing and status_counts["matched"] == len(complete)
    result: dict[str, Any] = {
        "status": ("complete_coverage" if expected_cells is not None else "complete_observed_groups")
        if not missing and bool(complete) else "partial_coverage",
        "scope": "bounded first-event causal diagnostic; instrumented timings are not deployable latency",
        "factor_definitions": {
            "current_output": "dense versus sparse using the old common support",
            "future_support": "fresh exact map versus held map; not a cheap-selector factor",
            "interaction": "(sparse_fresh - sparse_hold) - (dense_fresh - dense_hold)",
        },
        "score_units": "original official scorer units within each suite; suites are never pooled",
        "coverage": {"total_rows": len(rows), "excluded_phase_rows": dict(excluded_phases),
                     "factorial_rows": sum(len(group) for group in groups.values()),
                     "expected_groups": len(expected), "complete_groups": len(complete),
                     "expected_cells_supplied": expected_cells is not None, "missing": missing},
        "pairing_gate": {"status": "passed" if all_prefix_ok else "failed" if status_counts["mismatched"] else "unverified",
                         "prefix_counts": dict(status_counts),
                         "note": "All four factorial full-state prefix signatures must agree; control verification is separate and downstream divergence is not an exclusion rule."},
        "intervention_integrity": {"status": "failed" if integrity_counts["failed"] else "passed"
                                   if complete and not missing and integrity_counts["passed"] == len(complete) else "unverified",
                                   "group_counts": dict(integrity_counts)},
        "exposure_counts": dict(exposure_counts),
        "suites": {},
        "other_arms": {},
        "warnings": ["Small diagnostic cohorts do not establish benchmark superiority or noninferiority.",
                     "No automatic promotion, full503 resume, or deployable speedup follows from this report."],
    }
    if missing:
        result["warnings"].append("Contrasts use only explicitly enumerated complete groups; incomplete groups remain in coverage and arm summaries.")
    if expected_cells is None:
        result["warnings"].append("No frozen expected-cell list was supplied; complete observed groups do not establish complete study coverage.")
    if not all_prefix_ok:
        result["warnings"].append("Full-state prefix pairing is not established for the complete study; causal attribution is blocked.")
    if exposure_counts["inconsistent"] or exposure_counts["unverified"]:
        result["warnings"].append("Intervention exposure is inconsistent or unverified; inspect receipts before attributing effects.")
    if result["intervention_integrity"]["status"] != "passed":
        result["warnings"].append("All-five-GLOBAL scope and zero added model/sampler calls are not fully verified; no method promotion is supported.")
    for suite in sorted({key[0] for key in all_keys}):
        selected = [(key, group) for key, group in complete if key[0] == suite]
        suite_rows = [row for row in rows if row["suite"] == suite and row["label"] in REQUIRED_LABELS]
        effects = {name: _effect_rows(selected, weights) for name, weights in CONTRASTS.items()}
        subset_keys = {
            name: {key for key, group in selected if _group_exposure(group) == name}
            for name in ("exposed", "not_exposed", "inconsistent", "unverified")
        }
        prefix_matched = {key for key, group in selected if _prefix_status(group) == "matched"}
        contrasts = {}
        for name, effect in effects.items():
            stats = _effect_summary(effect, bootstrap_samples, bootstrap_seed)
            stats["exposure_subsets"] = {
                exposure: _point_effect([row for row in effect if _key(row) in keys])
                for exposure, keys in subset_keys.items()
            }
            stats["prefix_matched_subset_descriptive_only"] = _point_effect([row for row in effect if _key(row) in prefix_matched])
            contrasts[name] = stats
        bookends = []
        state_groups = []
        for key, group in selected:
            before, after = (group[label] for label in CONTROL_LABELS)
            bookends.append(_identity(key) | {
                "output_hash_equal": before["output_hash"] == after["output_hash"],
                "finish_reason_equal": before["finish_reason"] == after["finish_reason"],
                "prefix_status": _prefix_status(group), "exposure": _group_exposure(group),
                "control_prefix_status": _prefix_status(group, CONTROL_LABELS),
                "all_six_prefix_status": _prefix_status(group, REQUIRED_LABELS),
                "noop_hash_matches_both_controls": (group["dense_fresh"]["output_hash"] == before["output_hash"] == after["output_hash"]),
                "noop_hash_differs_despite_control_agreement": (before["output_hash"] == after["output_hash"]
                                                               and group["dense_fresh"]["output_hash"] != before["output_hash"]),
                "after_minus_before": {field: after[field] - before[field] for field in METRICS},
            })
            state_groups.append(_identity(key) | {
                "prefix_status": _prefix_status(group), "exposure": _group_exposure(group),
                "dense_fresh_vs_dense_hold": _state_comparison(group["dense_fresh"], group["dense_hold"]),
                "sparse_fresh_vs_sparse_hold": _state_comparison(group["sparse_fresh"], group["sparse_hold"]),
                "dense_fresh_vs_sparse_fresh": _state_comparison(group["dense_fresh"], group["sparse_fresh"]),
                "dense_hold_vs_sparse_hold": _state_comparison(group["dense_hold"], group["sparse_hold"]),
                "control_before_vs_control_after": _state_comparison(before, after),
                "dense_fresh_vs_control_before": _state_comparison(group["dense_fresh"], before),
            })
        noise = {field: {"mean_absolute_delta": _mean(abs(row["after_minus_before"][field]) for row in bookends),
                         "p95_absolute_delta": _quantile([abs(row["after_minus_before"][field]) for row in bookends], .95),
                         "max_absolute_delta": max((abs(row["after_minus_before"][field]) for row in bookends), default=None)}
                 for field in METRICS}
        result["suites"][suite] = {
            "paired_groups": len(selected), "question_clusters": len({key[1] for key, _ in selected}),
            "arms": {label: _arm_summary([row for row in suite_rows if row["label"] == label]) for label in REQUIRED_LABELS},
            "contrasts": contrasts,
            "post_event_state_comparisons": state_groups,
            "control_repeatability": {"pairs": len(bookends),
                                      "output_hash_disagreements": sum(not row["output_hash_equal"] for row in bookends),
                                      "score_disagreements": sum(row["after_minus_before"]["score"] != 0 for row in bookends),
                                      "control_prefix_counts": dict(Counter(row["control_prefix_status"] for row in bookends)),
                                      "all_six_prefix_counts": dict(Counter(row["all_six_prefix_status"] for row in bookends)),
                                      "noop_hash_differs_despite_control_agreement": sum(row["noop_hash_differs_despite_control_agreement"] for row in bookends),
                                      "noise_absolute": noise, "groups": bookends},
            "nonexposure_check": {
                "groups": len(subset_keys["not_exposed"]),
                "factorial_output_hash_disagreements": sum(
                    len({group[label]["output_hash"] for label in FACTORIAL_LABELS}) != 1
                    for key, group in selected if key in subset_keys["not_exposed"]),
                "note": "A nonexposed group should have no treatment effect; disagreement is retained and compared with runtime variation.",
            },
        }
    for suite, suite_rows in sorted(other.items()):
        result["other_arms"][suite] = {label: _arm_summary([row for row in suite_rows if row["label"] == label])
                                         for label in sorted({row["label"] for row in suite_rows})}
    return result


def summarize_headroom_trials(trials: Iterable[Mapping[str, Any]], *,
                              dynamic_family: str = "dynamic", fixed_family: str = "fixed16") -> dict[str, Any]:
    """Separate clean requests from audited spans; never infer critical path.

    Rows carry family/pass_name/repeat and optionally native_record (the
    corresponding sanitized records.jsonl entry). Engine IDs deliberately do
    not define matching: families use different engines. No accuracy CI is
    calculated for this tiny, deliberately selected headroom cohort.
    """
    rows = []
    for source in trials:
        if source.get("phase", "headroom") != "headroom":
            continue
        row = _validate(source)
        for field in ("family", "pass_name"):
            _require(field in source, f"headroom row missing {field}")
            row[field] = source[field]
        row["repeat"] = source.get("sample_repeat", source.get("repeat"))
        _require(isinstance(row["repeat"], int), "headroom row missing integer sample_repeat")
        if row["family"] == "shared":
            _require(row["label"] in ("dynamic", "fixed16"), "unknown shared headroom arm")
            row["family"] = row["label"]
        _require(row["pass_name"] in ("clean", "audit", "profile"), "unknown headroom pass")
        if row["pass_name"] == "profile":
            row["pass_name"] = "audit"
        record = source.get("native_record")
        if record is None:
            record = {field: source[field] for field in ("cuda_graph_captures", "value_clean_timing", "peak_cuda_allocated_bytes") if field in source}
            record["receipts"] = {"adapter": source.get("native_receipt") or {}}
        _require(isinstance(record, Mapping), "headroom native_record must be a mapping")
        row["native_record"] = record
        rows.append(row)
    clean = [row for row in rows if row["pass_name"] == "clean"]
    audited = [row for row in rows if row["pass_name"] == "audit"]
    result: dict[str, Any] = {
        "scope": "selected batch-one headroom diagnostic; clean timing and audited work remain separate",
        "trials": len(rows), "clean_trials": len(clean), "audit_trials": len(audited),
        "dynamic_family": dynamic_family, "fixed_family": fixed_family,
        "clean_arms": {}, "paired_clean": {}, "audit_arms": {},
        "critical_path_headroom": {"status": "unresolved", "request_latency_gain_bound": None,
            "reason": "Inclusive CUDA spans and exclusive activity categories do not identify removable request-critical-path time or overlap with unaffected work."},
        "warnings": ["No population accuracy, noninferiority, or larger-batch claim follows from this selected cohort.",
                     "A refresh-span/request ratio is a descriptive audited-time proxy, not a rigorous speedup bound.",
                     "Do not add GLOBAL parent spans to discovery, buffer, or sparse-consumer child spans."]}
    clean_by_family: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
    for row in clean:
        key = (row["suite"], row["index"], row["panel_seed"], row["repeat"])
        _require(key not in clean_by_family[row["family"]], "duplicate headroom family/cell/repeat")
        clean_by_family[row["family"]][key] = row
    for family, cells in sorted(clean_by_family.items()):
        rr = list(cells.values())
        validation = Counter()
        for row in rr:
            record = row["native_record"]
            adapter = ((record.get("receipts") or {}).get("adapter") or {})
            if not record:
                validation["unverified"] += 1
            elif record.get("cuda_graph_captures") is None:
                validation["unverified"] += 1
            elif record.get("cuda_graph_captures") != 0 or adapter.get("headroom"):
                validation["failed"] += 1
            elif record.get("value_clean_timing") not in (True, 1):
                validation["unverified"] += 1
            else:
                validation["passed"] += 1
        result["clean_arms"][family] = _arm_summary(rr) | {"timing_verification": dict(validation),
            "peak_cuda_allocated_bytes_max": max((row["native_record"].get("peak_cuda_allocated_bytes", 0) or 0 for row in rr), default=None),
            "question_count": len({(row["suite"], row["index"]) for row in rr}),
            "request_values": [{**_identity(_key(row)), "repeat": row["repeat"],
                                 **{field: row[field] for field in METRICS}, "finish_reason": row["finish_reason"]}
                               for row in rr]}
    reference = clean_by_family.get(dynamic_family, {})
    for family, cells in sorted(clean_by_family.items()):
        if family == dynamic_family:
            continue
        common = sorted(set(cells) & set(reference))
        pairs = []
        for key in common:
            candidate, control = cells[key], reference[key]
            pairs.append(dict(suite=key[0], index=key[1], panel_seed=key[2], repeat=key[3],
                candidate_engine=candidate["engine"], control_engine=control["engine"],
                delta={field: candidate[field] - control[field] for field in METRICS},
                ratio={field: candidate[field] / control[field] if control[field] else None
                       for field in ("denoise_forwards", "wall_s", "decode_s", "output_tokens")},
                output_hash_equal=candidate["output_hash"] == control["output_hash"],
                score_equal=candidate["score"] == control["score"]))
        result["paired_clean"][family + "_minus_" + dynamic_family] = {
            "pairs": len(pairs), "missing_candidate_cells": len(set(reference) - set(cells)),
            "missing_reference_cells": len(set(cells) - set(reference)),
            "mean_deltas": {field: _mean(pair["delta"][field] for pair in pairs) for field in METRICS},
            "ratios_of_totals": {field: (sum(cells[key][field] for key in common) / sum(reference[key][field] for key in common))
                                 if sum(reference[key][field] for key in common) else None
                                 for field in ("denoise_forwards", "wall_s", "decode_s", "output_tokens")},
            "groups": pairs,
                "interpretation": "Matched input/seed/repetition; dynamic/fixed16 share an engine, substrate controls may use separate engines. Request-level descriptive comparison; runtime variation remains visible."}
    for family in sorted({row["family"] for row in audited}):
        rr = [row for row in audited if row["family"] == family]
        spans: dict[str, Counter] = defaultdict(Counter)
        physical: dict[str, Counter] = defaultdict(Counter)
        local = Counter()
        phase_calls = Counter()
        inventory = Counter()
        categories: dict[str, Counter] = defaultdict(Counter)
        profiler_requests = []
        proxies = []
        physical_requests = 0
        for row in rr:
            record = row["native_record"]
            adapter = ((record.get("receipts") or {}).get("adapter") or {})
            headroom = adapter.get("headroom") or {}
            physical_requests += bool(adapter.get("value_phase_tiles"))
            inventory["passed" if headroom.get("complete_scope_inventory") is True else "unavailable_or_incomplete"] += 1
            for phase, count in headroom.get("phase_calls", {}).items():
                phase_calls[phase] += count
            for label, values in headroom.get("spans", {}).items():
                for field in ("calls", "cpu_inclusive_ms", "cpu_exclusive_own_scopes_ms", "cuda_inclusive_span_ms"):
                    if field in values:
                        spans[label][field] += values[field]
            for phase, values in adapter.get("value_phase_tiles", {}).items():
                for field in ("calls", "eligible_tiles", "kept_tiles", "skipped_tiles"):
                    if field in values:
                        physical[phase][field] += values[field]
            for field in ("native_local_calls_audited", "native_local_eligible_rectangles", "native_local_skipped_tiles"):
                if field in adapter:
                    local[field] += adapter[field]
            span = headroom.get("spans", {}).get("denoise/global_refresh", {})
            refresh_ms = span.get("cuda_inclusive_span_ms", 0 if headroom.get("complete_scope_inventory") else None)
            proxies.append({**_identity(_key(row)), "repeat": row["repeat"],
                "audit_request_s": row["wall_s"], "global_refresh_inclusive_ms": refresh_ms,
                "refresh_span_over_audit_request": refresh_ms / (row["wall_s"] * 1000)
                    if refresh_ms is not None and row["wall_s"] else None,
                "is_critical_path_bound": False})
            profiler = headroom.get("profiler")
            if isinstance(profiler, Mapping):
                for label, values in profiler.get("exclusive_activity_categories", {}).items():
                    for field in ("activities", "activity_sum_ms", "activity_union_ms"):
                        if field in values:
                            categories[label][field] += values[field]
                profiler_requests.append({**_identity(_key(row)), "repeat": row["repeat"],
                    **{field: profiler.get(field) for field in ("gpu_activity_available", "cuda_activities", "scope_linked_activities", "unlinked_activities", "devices")}})
        eligible = sum(values.get("eligible_tiles", 0) for values in physical.values())
        kept = sum(values.get("kept_tiles", 0) for values in physical.values())
        local_eligible = local.get("native_local_eligible_rectangles", 0)
        result["audit_arms"][family] = {
            "requests": len(rr), "complete_scope_inventory": dict(inventory), "native_phase_calls": dict(phase_calls),
            "physical_receipt_requests": physical_requests,
            "inclusive_spans_by_label": {key: dict(value) for key, value in sorted(spans.items())},
            "global_physical_tiles_by_phase": {key: dict(value) for key, value in sorted(physical.items())},
            "global_physical_totals": {"eligible_tiles": eligible, "kept_tiles": kept,
                                       "skipped_tiles": eligible - kept, "skipped_fraction": (eligible - kept) / eligible if eligible else None},
            "local_physical_totals": dict(local),
            "overall_skipped_fraction": (eligible - kept) / (eligible + local_eligible) if eligible + local_eligible else None,
            "overall_denominator": "audited decoder GLOBAL Q128/KV64 full support plus LOCAL window-intersecting rectangles; not CUDA CTA counts",
            "refresh_time_proxy": proxies,
            "exclusive_activity_categories": {key: dict(value) for key, value in sorted(categories.items())},
            "profiled_requests": profiler_requests,
            "interpretation": "Each span label is separate. GPU activity categories are exclusive assignments but simultaneous kernels may overlap; do not sum category unions into request time. Gaps are not proof of host idle."}
    return result


def _format(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Bounded refresh diagnostic", "", report["scope"], "",
             f"Coverage: {report['coverage']['complete_groups']}/{report['coverage']['expected_groups']} complete paired groups. "
             f"Full-state prefix pairing: **{report['pairing_gate']['status']}**.", "",
             "Current output uses dense attention or the old common sparse support; future support is freshly selected or held. "
             "This experiment does not evaluate a cheap selector.", ""]
    for suite, result in report["suites"].items():
        lines.extend([f"## {suite}", "", f"{result['paired_groups']} paired groups; {result['question_clusters']} question clusters.", "",
                      "| Contrast | Score delta | Calls delta | Calls 95% cluster interval | Request delta, diagnostic s |", "|---|---:|---:|---|---:|"])
        for name in ("sparse_minus_dense", "fresh_minus_hold", "interaction", "control_after_minus_before"):
            contrast = result["contrasts"][name]
            ci = contrast["bootstrap"]["ci95"]["denoise_forwards"]
            lines.append(f"| {name} | {_format(contrast['means']['score'])} | {_format(contrast['means']['denoise_forwards'])} | "
                         f"[{_format(ci[0])}, {_format(ci[1])}] | {_format(contrast['means']['wall_s'])} |")
        control = result["control_repeatability"]
        lines.extend(["", f"Control bookends disagree in output hash in {control['output_hash_disagreements']}/{control['pairs']} groups "
                      f"and in official score in {control['score_disagreements']}/{control['pairs']} groups.", "",
                      "JSON includes every simple contrast, seed/engine strata, exposure groups, missing coverage, "
                      "question contributions, leave-one-question-out sensitivity, and A/A variation. All adverse and capped outcomes remain included.", ""])
    if report.get("headroom"):
        lines.extend(render_headroom_markdown(report["headroom"]).splitlines())
        lines.append("")
    lines.extend(["Score deltas retain each suite's official units. Intervals are descriptive; nonsignificance does not establish equivalence. "
                  "Instrumented time is not a deployable latency estimate, and fewer calls do not establish speedup.", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def render_headroom_markdown(report: Mapping[str, Any]) -> str:
    lines = ["## Separate clean runtime headroom", "", report["scope"], "",
             "| Arm | Requests | Mean score | Mean calls | Mean request s |", "|---|---:|---:|---:|---:|"]
    for family, arm in report["clean_arms"].items():
        lines.append(f"| {family} | {arm['trials']} | {_format(arm['means']['score'])} | "
                     f"{_format(arm['means']['denoise_forwards'])} | {_format(arm['means']['wall_s'])} |")
    lines.extend(["", "JSON reports paired dynamic/fixed16 ratios and substrate-control differences. "
                  "Audit spans, physical rectangles, and profiler activity are separate. "
                  "The removable request-critical-path bound remains **unresolved**: inclusive spans and disjoint activity labels do not establish overlap or latency savings.", ""])
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_paths(study: Path | None, scored_paths: Sequence[Path]) -> tuple[list[Path], list[Mapping[str, Any]] | None, dict[str, str]]:
    """Explicit paths or frozen plan directories; never recursively scan runs."""
    paths = list(scored_paths)
    bindings: dict[str, str] = {}
    expected = None
    if study is not None:
        protocol_path = study / "protocol.json"
        protocol = _read_json(protocol_path)
        bindings["protocol.json"] = _sha(protocol_path)
        if "expected_cells" in protocol:
            expected = protocol["expected_cells"]
        if not paths:
            for plan in protocol.get("plans", []):
                if plan.get("phase") != "factorial":
                    continue
                folder = plan.get("folder")
                _require(isinstance(folder, str) and bool(folder), "plan folder missing")
                path = study / folder / "scored.json"
                if path.exists():
                    paths.append(path)
    _require(bool(paths), "no scored files; supply --scored or completed protocol plans")
    _require(len({path.resolve() for path in paths}) == len(paths), "duplicate scored input path")
    for ordinal, path in enumerate(paths, 1):
        # Bind bytes without publishing user/host/private directory names.
        name = f"scored{ordinal:03d}:{path.name}"
        if study is not None:
            try:
                name = str(path.resolve().relative_to(study.resolve()))
            except ValueError:
                pass
        bindings[name] = _sha(path)
    return paths, expected, bindings


def _headroom_from_study(study: Path) -> tuple[dict[str, Any] | None, dict[str, str]]:
    protocol = _read_json(study / "protocol.json")
    rows, execution = [], []
    bindings = {}
    for plan in protocol.get("plans", []):
        if plan.get("phase") != "headroom":
            continue
        path = study / plan["folder"] / "scored.json"
        if not path.exists():
            execution.append(dict(engine=plan["engine"], status="no_scored_rows"))
            continue
        data = _read_json(path)
        _require(isinstance(data, list), "headroom scored payload must be a list")
        for row in data:
            _require(row.get("phase") == "headroom" and row.get("engine") == plan["engine"]
                     and row.get("suite") == plan["suite"] and row.get("family") == plan["family"]
                     and row.get("pass_name") == plan["pass_name"], "headroom plan/row identity mismatch")
        marker = path.parent / "attempt_complete.json"
        status = "unverified_or_partial"
        if marker.exists():
            receipt = _read_json(marker)
            _require(receipt.get("protocol_sha256") == _sha(study / "protocol.json"), "headroom protocol changed")
            _require(receipt.get("files_sha256", {}).get("scored.json") == _sha(path), "headroom scored bytes changed")
            status = receipt.get("status", "unverified")
            bindings[str(marker.relative_to(study))] = _sha(marker)
        execution.append(dict(engine=plan["engine"], status=status, measured_rows=len(data),
                              planned_rows=plan.get("measured_count")))
        rows.extend(data)
        bindings[str(path.relative_to(study))] = _sha(path)
    if not rows:
        return None, bindings
    report = summarize_headroom_trials(rows)
    report["engine_coverage"] = execution
    return report, bindings


def run(study: Path | None, output: Path, *, scored_paths: Sequence[Path] = (),
        bootstrap_samples: int = 10000, bootstrap_seed: int = 20261018) -> dict[str, Any]:
    _require(not output.exists(), "output exists; preserve immutable report")
    paths, expected, bindings = _load_paths(study, scored_paths)
    trials = []
    for path in paths:
        payload = _read_json(path)
        _require(isinstance(payload, list), f"scored file must contain a list: {path}")
        marker = path.parent / "attempt_complete.json"
        if marker.exists():
            receipt = _read_json(marker)
            _require(receipt.get("files_sha256", {}).get("scored.json") == _sha(path), "factorial scored bytes changed")
            if study is not None:
                _require(receipt.get("protocol_sha256") == _sha(study / "protocol.json"), "factorial protocol changed")
        trials.extend(payload)
    report = summarize_trials(trials, bootstrap_samples=bootstrap_samples,
                              bootstrap_seed=bootstrap_seed, expected_cells=expected)
    if study is not None:
        headroom, extra_bindings = _headroom_from_study(study)
        if headroom is not None:
            report["headroom"] = headroom
        bindings.update(extra_bindings)
    report["input_sha256"] = bindings
    report["analysis_source_sha256"] = _sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.json"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown = output / "summary.md"
    markdown.write_text(render_markdown(report))
    # This marker describes report generation only, never campaign completion.
    (output / "analysis_complete.json").write_text(json.dumps({
        "status": "analysis_complete", "study_coverage": report["status"],
        "prefix_pairing": report["pairing_gate"]["status"],
        "input_sha256": bindings, "analysis_source_sha256": report["analysis_source_sha256"],
        "files_sha256": {path.name: _sha(path) for path in (summary, markdown)},
        "full503_resume_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    return report


def run_headroom(study: Path, output: Path) -> dict[str, Any]:
    """Write a separate immutable headroom report before factorial completion."""
    _require(not output.exists(), "output exists; preserve immutable report")
    report, bindings = _headroom_from_study(study)
    _require(report is not None, "no headroom scored rows")
    bindings["protocol.json"] = _sha(study / "protocol.json")
    report["input_sha256"] = bindings
    report["analysis_source_sha256"] = _sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "summary.md").write_text(render_headroom_markdown(report) + "\n")
    (output / "analysis_complete.json").write_text(json.dumps(dict(status="headroom_analysis_complete",
        scope="report completion only; no campaign promotion", input_sha256=bindings,
        files_sha256={name: _sha(output / name) for name in ("summary.json", "summary.md")}), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--scored", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261018)
    parser.add_argument("--headroom-only", action="store_true")
    args = parser.parse_args()
    if args.headroom_only:
        _require(args.study is not None, "--headroom-only requires --study")
        result = run_headroom(args.study, args.output)
        print(json.dumps({"status": "headroom_analysis_complete", "trials": result["trials"]}))
        return
    result = run(args.study, args.output, scored_paths=args.scored,
                 bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "pairing_gate": result["pairing_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
