"""Gold-separated offline scoring and redacted accounting for frozen v21 panels.

First successful receipts alone enter task quality. Only accepted warm receipts
enter timing. The 12-request CP1 diagnostic is work-only and never selects a
numeric method by answer quality. No prompts, answers or completions are written.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v18_protocol import sha
from scripts.v20_score import cluster_interval, score_firsts, verify_sources
from scripts.v21_run import strict_v21_warm, validate_protocol


def _read(path: Path):
    return json.loads(path.read_text())


def load_records(protocol: dict, binding_path: Path, ledger_paths: list[Path]) -> dict[str, dict]:
    expected = {execution_key(e): e for e in protocol["schedule"]}
    if len(expected) != len(protocol["schedule"]):
        raise ValueError("duplicate frozen execution key")
    binding_sha = sha(binding_path.read_bytes())
    hosts = {(a["host"], a["gpu_uuid"]) for a in protocol["block_assignments"].values()}
    found = {}
    for path in ledger_paths:
        starts = ends = 0
        active_host = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("event")
            if kind == "start":
                starts += 1
                if (starts != ends + 1 or event.get("protocol_id") != protocol["protocol_id"] or
                        event.get("binding_sha256") != binding_sha or
                        event.get("model_revision") != protocol["model_revision"] or
                        (event.get("host"), event.get("gpu_uuid")) not in hosts):
                    raise ValueError("scoring ledger has uncertain/source-drifted worker start")
                active_host = (event["host"], event["gpu_uuid"])
            elif kind == "worker_end":
                ends += 1
                if ends != starts or (event.get("host"), event.get("gpu_uuid")) != active_host:
                    raise ValueError("scoring ledger has unmatched worker end")
                active_host = None
            elif kind == "run":
                key = event.get("execution_key")
                spec = expected.get(key)
                if starts != ends + 1 or spec is None or key in found or active_host != (spec["host"], spec["gpu_uuid"]):
                    raise ValueError("foreign/duplicate/writerless scoring execution")
                if any(event.get(k) != spec[k] for k in
                       ("index", "block", "dataset", "arm", "id", "seed", "role", "repeat", "cell_id", "host", "gpu_uuid")):
                    raise ValueError("scoring execution differs from frozen schedule")
                if event.get("generation_seed") != spec["seed"]:
                    raise ValueError("actual generation seed differs")
                if protocol["panel_kind"] == "zero_pruning_diagnostic" and (
                        event.get("quality_eligible") is not False or event.get("timing_eligible") is not False):
                    raise ValueError("diagnostic receipt falsely claims evaluation eligibility")
                found[key] = event
        if starts != ends:
            raise ValueError("worker ledger remains open; score only after writers close")
    return found


def _dist(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    def q(p):
        x = (len(ordered) - 1) * p
        lo, hi = int(x), min(int(x) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)
    return {"n": len(values), "mean": statistics.mean(values), "median": q(.5),
            "p90": q(.9), "max": ordered[-1]}


def build_cells(protocol: dict, records: dict[str, dict], quality: dict[str, dict]) -> dict[tuple, dict]:
    grouped = defaultdict(dict)
    for spec in protocol["schedule"]:
        key = (spec["dataset"], spec["id"], spec["seed"], spec["arm"])
        grouped[key][spec["role"]] = (spec, records.get(execution_key(spec)))
    cells = {}
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    for key, pair in grouped.items():
        first_spec, first = pair["attempt0"]
        warm = pair.get("warm", (None, None))[1]
        score = quality.get(first_spec["cell_id"])
        if score is not None and (diagnostic or first is None or not first.get("ok")):
            raise ValueError("quality attaches to ineligible or failed first execution")
        if diagnostic and warm is not None:
            raise ValueError("diagnostic has unexpected warm execution")
        accepted = False
        if warm is not None:
            recomputed = strict_v21_warm(first, warm)
            if warm.get("acceptance") != recomputed:
                raise ValueError("stored warm acceptance differs from strict v21 identity")
            accepted = recomputed["accepted"]
        cells[key] = {"first": first, "warm": warm, "quality": score, "warm_accepted": accepted,
                      "cell_id": first_spec["cell_id"], "block": first_spec["block"],
                      "host": first_spec["host"], "gpu_uuid": first_spec["gpu_uuid"],
                      "dataset": key[0], "id": key[1], "seed": key[2], "arm": key[3]}
    return cells


def block_completeness(protocol: dict, records: dict[str, dict], cells: dict) -> dict:
    by_block = defaultdict(list)
    for e in protocol["schedule"]:
        by_block[e["block"]].append(e)
    failed, partial = [], []
    recorded = first_success = warm_accepted = valid = 0
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    for block, specs in sorted(by_block.items()):
        rows = [records.get(execution_key(e)) for e in specs]
        missing = any(r is None for r in rows)
        failure = any(r is not None and not r.get("ok") for r in rows)
        if not missing:
            recorded += 1
        if all(r is not None and r.get("ok") for e, r in zip(specs, rows) if e["role"] == "attempt0"):
            first_success += 1
        qkey = (specs[0]["dataset"], specs[0]["id"], specs[0]["seed"])
        block_cells = [cells[(qkey[0], qkey[1], qkey[2], arm)] for arm in protocol["arms"]]
        if not diagnostic and all(c["warm_accepted"] for c in block_cells):
            warm_accepted += 1
        if missing:
            partial.append(block)
        if failure or any(c["warm"] is not None and not c["warm_accepted"] for c in block_cells):
            failed.append(block)
        if not missing and not failure and (diagnostic or all(c["warm_accepted"] for c in block_cells)):
            valid += 1
    return {"planned_blocks": len(by_block), "recorded_all_blocks": recorded,
            "successful_first_all_arms_blocks": first_success,
            "accepted_warm_all_arms_blocks": None if diagnostic else warm_accepted,
            "complete_valid_blocks": valid, "failed_block_ids": failed, "partial_block_ids": partial}


def _stop_counts(record: dict) -> Counter:
    phases = ((record.get("phase_evidence") or {}).get("per_canvas") or
              record.get("per_canvas_stopping") or [])
    counts = Counter()
    if not phases:
        counts["unrecorded"] = record.get("canvases", 0)
        return counts
    for row in phases:
        native, cap = bool(row.get("native_stop")), bool(row.get("iteration_cap"))
        counts["both" if native and cap else "native_only" if native else "cap_only" if cap else "neither"] += 1
    if phases and len(phases) != record.get("canvases"):
        raise ValueError("phase/canvas count differs")
    return counts


def arm_groups(protocol: dict, cells: dict) -> list[dict]:
    by_group = defaultdict(list)
    for (dataset, _, _, arm), cell in cells.items():
        by_group[(dataset, cell["host"], arm)].append(cell)
    output = []
    for (dataset, host, arm), group in sorted(by_group.items()):
        first = [c["first"] for c in group if c["first"] and c["first"].get("ok")]
        scored = [c["quality"] for c in group if c["quality"] is not None]
        warms = [c["warm"] for c in group if c["warm_accepted"]]
        calls = [r["decoder_calls"] for r in first]
        canvases = [r["canvases"] for r in first]
        tokens = [r["output_tokens"] for r in first]
        if any(type(n) is not int or n <= 0 for n in calls + canvases):
            raise ValueError("successful first lacks positive decoder work")
        per_canvas = [n for r in first for n in r.get("per_canvas_calls", [])]
        if any(sum(r.get("per_canvas_calls", [])) != r["decoder_calls"] for r in first):
            raise ValueError("decoder calls do not equal sum of per-canvas calls")
        stops = Counter()
        for r in first:
            stops.update(_stop_counts(r))
        if sum(stops.values()) != sum(canvases):
            raise ValueError("disjoint stop categories do not cover all canvases")
        router = Counter()
        measured = 0
        for r in first:
            p = r.get("router_phase_evidence") or {}
            if all(type(p.get(k)) is int for k in ("A", "D", "H")):
                if sum(p[k] for k in ("A", "D", "H")) != p.get("attention_calls"):
                    raise ValueError("A/D/H work does not conserve attention calls")
                router.update({k: p[k] for k in ("A", "D", "H")})
                measured += 1
        warm_wall = [r["api_wall_s"] for r in warms]
        spans = [(r.get("phase_evidence") or {}).get("prefill_end_to_finish_gpu_s") for r in warms]
        spans = [x for x in spans if type(x) in (int, float) and math.isfinite(x) and x > 0]
        total_calls, total_canvases = sum(calls), sum(canvases)
        quality_eligible = protocol["panel_kind"] != "zero_pruning_diagnostic"
        output.append({"dataset": dataset, "host": host, "arm": arm,
                       "unique_question_count": len({c["id"] for c in group}),
                       "planned_first_cells": len(group),
                       "first_present": sum(c["first"] is not None for c in group),
                       "first_success": len(first), "first_failed": sum(c["first"] is not None and not c["first"].get("ok") for c in group),
                       "first_missing": sum(c["first"] is None for c in group),
                       "quality_eligible": quality_eligible, "scored_first": len(scored),
                       "score_mean": statistics.mean(q["score"] for q in scored) if scored else None,
                       "strict_correct": sum(bool(q["strict_correct"]) for q in scored) if quality_eligible else None,
                       "task_correct": sum(bool(q["task_correct"]) for q in scored) if quality_eligible else None,
                       "task_at_cap": sum(bool(q["task_correct"] and q["capped"]) for q in scored) if quality_eligible else None,
                       "parsed": sum(bool(q["parsed"]) for q in scored) if quality_eligible else None,
                       "eos_wrong": sum(bool(q["eos"] and not q["strict_correct"]) for q in scored) if quality_eligible else None,
                       "termination": dict(sorted(Counter(r.get("termination", "unknown") for r in first).items())),
                       "output_tokens_by_termination": {term: _dist([r["output_tokens"] for r in first
                                                                      if r.get("termination") == term])
                                                       for term in sorted({r.get("termination") for r in first})},
                       "output_tokens": _dist(tokens), "canvases": _dist(canvases), "decoder_calls": _dist(calls),
                       "exact_work": {"output_tokens": sum(tokens), "canvases": total_canvases,
                                      "decoder_calls": total_calls,
                                      "pooled_calls_per_canvas": total_calls / total_canvases if total_canvases else None,
                                      "per_canvas_calls_distribution": _dist(per_canvas)},
                       "stop_categories": {k: stops[k] for k in ("native_only", "cap_only", "both", "neither", "unrecorded")},
                       "router_phase_layer_calls": {"A": router["A"], "D": router["D"], "H": router["H"],
                                                    "measured_first_requests": measured} if measured else None,
                       "warm_present": sum(c["warm"] is not None for c in group),
                       "warm_failed": sum(c["warm"] is not None and not c["warm"].get("ok") for c in group),
                       "warm_accepted": len(warms) if quality_eligible else None,
                       "warm_whole_request_s": _dist(warm_wall) if quality_eligible else None,
                       "warm_prefill_end_to_finish_cuda_span_s": _dist(spans) if quality_eligible else None,
                       "amortized_warm_wall_s_per_first_decoder_call":
                           sum(warm_wall) / total_calls if warm_wall and len(warm_wall) == len(first) and total_calls else None,
                       "quality_denominator_note": "successful first requests only; no warm score",
                       "timing_note": "accepted warm whole request; CUDA span includes host gaps/later work; neither is direct forward"})
    return output


def _error_class(record: dict | None) -> str | None:
    if record is None or record.get("ok"):
        return None
    value = str(record.get("error") or "")
    head = value.split(":", 1)[0]
    if head.startswith("timeout>"):
        return "Timeout"
    if head.isidentifier() and head.endswith(("Error", "Exception")):
        return head
    return "UnclassifiedFailure"


def cell_rows(protocol: dict, cells: dict) -> list[dict]:
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    rows = []
    for key, cell in sorted(cells.items()):
        first, warm, q = cell["first"], cell["warm"], cell["quality"]
        first_ok = bool(first and first.get("ok"))
        if warm is None:
            warm_status = "not_planned" if diagnostic else "missing"
        elif not warm.get("ok"):
            warm_status = "failed"
        elif cell["warm_accepted"]:
            warm_status = "accepted"
        else:
            warm_status = "rejected"
        stops = _stop_counts(first) if first_ok else Counter()
        phase = (first or {}).get("router_phase_evidence") or {}
        wall = warm.get("api_wall_s") if cell["warm_accepted"] else None
        span = ((warm.get("phase_evidence") or {}).get("prefill_end_to_finish_gpu_s")
                if cell["warm_accepted"] else None)
        calls = first.get("decoder_calls") if first_ok else None
        canvases = first.get("canvases") if first_ok else None
        rows.append({"dataset": key[0], "id": key[1], "seed": key[2], "arm": key[3],
                     "block": cell["block"], "cell_id": cell["cell_id"],
                     "host": cell["host"], "gpu_uuid": cell["gpu_uuid"],
                     "first_status": "missing" if first is None else "success" if first_ok else "failed",
                     "first_error_class": _error_class(first),
                     "quality_eligible": not diagnostic, "scored_first": q is not None,
                     "first_score": q["score"] if q else None,
                     "strict_correct": q["strict_correct"] if q else None,
                     "task_correct": q["task_correct"] if q else None,
                     "parsed": q["parsed"] if q else None, "eos": q["eos"] if q else None,
                     "capped": q["capped"] if q else None,
                     "task_at_cap": bool(q["task_correct"] and q["capped"]) if q else None,
                     "termination": first.get("termination") if first_ok else None,
                     "output_tokens": first.get("output_tokens") if first_ok else None,
                     "canvases": canvases, "decoder_calls": calls,
                     "mean_calls_per_canvas": calls / canvases if calls is not None and canvases else None,
                     "stop_native_only": stops["native_only"], "stop_cap_only": stops["cap_only"],
                     "stop_both": stops["both"], "stop_neither": stops["neither"],
                     "stop_unrecorded": stops["unrecorded"],
                     "router_A": phase.get("A"), "router_D": phase.get("D"), "router_H": phase.get("H"),
                     "first_cold_request_wall_s": first.get("api_wall_s") if first_ok else None,
                     "warm_status": warm_status, "warm_error_class": _error_class(warm),
                     "warm_rejection_reasons": (warm.get("acceptance") or {}).get("reasons") if warm is not None else None,
                     "accepted_warm_request_wall_s": wall,
                     "accepted_warm_prefill_end_to_finish_cuda_span_s": span})
    return rows


def paired_ratios(protocol: dict, cells: dict) -> list[dict]:
    if protocol["panel_kind"] == "zero_pruning_diagnostic":
        return []
    references = protocol.get("reference_arms") or [a for a in ("D_native", "T_scope", "B_A8_matched") if a in protocol["arms"]]
    if not references:
        references = protocol["arms"][:1]
    if any(a not in protocol["arms"] for a in references) or len(set(references)) != len(references):
        raise ValueError("reference arms differ from frozen arm list")
    output = []
    for dataset in protocol["ids"]:
        for arm in protocol["arms"]:
            for ref in references:
                if arm == ref:
                    continue
                pairs = []
                score_q, time_q = defaultdict(list), defaultdict(list)
                for (d, qid, seed, a), c in cells.items():
                    if (d, a) != (dataset, arm):
                        continue
                    other = cells[(d, qid, seed, ref)]
                    if (c["host"], c["gpu_uuid"]) != (other["host"], other["gpu_uuid"]):
                        raise ValueError("paired methods crossed frozen GPU")
                    if c["quality"] is not None and other["quality"] is not None:
                        score_q[qid].append(c["quality"]["score"] - other["quality"]["score"])
                    if c["warm_accepted"] and other["warm_accepted"]:
                        left, right = c["warm"]["api_wall_s"], other["warm"]["api_wall_s"]
                        if min(left, right) <= 0:
                            raise ValueError("accepted warm wall time must be positive")
                        pairs.append((c["host"], left, right, c["first"]["decoder_calls"],
                                      other["first"]["decoder_calls"], c["first"]["canvases"],
                                      other["first"]["canvases"]))
                        time_q[qid].append(math.log(left / right))
                hosts = {}
                for host in sorted({p[0] for p in pairs}):
                    rows = [p for p in pairs if p[0] == host]
                    lt, rt, lc, rc, lv, rv = (sum(p[i] for p in rows) for i in range(1, 7))
                    hosts[host] = {"paired_cells": len(rows),
                                   "warm_wall_ratio_of_totals": lt / rt,
                                   "warm_wall_geometric_ratio": math.exp(statistics.mean(math.log(p[1] / p[2]) for p in rows)),
                                   "decoder_call_ratio_of_totals": lc / rc,
                                   "canvas_ratio_of_totals": lv / rv,
                                   "pooled_calls_per_canvas_ratio": (lc / lv) / (rc / rv),
                                   "amortized_warm_wall_per_call_ratio": (lt / lc) / (rt / rc)}
                seed = int(sha(f"v21/{protocol['protocol_id']}/{dataset}/{arm}/{ref}")[:12], 16)
                score, score_ci = cluster_interval(score_q, seed=seed)
                time, time_ci = cluster_interval(time_q, seed=seed + 1, transform=math.exp)
                output.append({"dataset": dataset, "arm": arm, "reference": ref,
                               "paired_quality_cells": sum(len(x) for x in score_q.values()),
                               "paired_accepted_warm_cells": len(pairs), "by_host": hosts,
                               "paired_score_delta": score, "paired_score_delta_95_exploratory": score_ci,
                               "question_cluster_warm_ratio": time,
                               "question_cluster_warm_ratio_95_exploratory": time_ci,
                               "uncertainty_note": "question bootstrap keeps seed repeats in one cluster; descriptive small-panel interval"})
    return output


def summarize(protocol: dict, records: dict[str, dict], quality: dict[str, dict], *,
              protocol_sha: str, binding_sha: str, ruler_task_by_id: dict[str, str] | None = None) -> dict:
    validate_protocol(protocol)
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    if diagnostic and quality:
        raise ValueError("diagnostic quality may not enter candidate selection summary")
    if not diagnostic:
        first_specs = {e["cell_id"]: e for e in protocol["schedule"] if e["role"] == "attempt0"}
        expected_quality = {cid for cid, e in first_specs.items()
                            if (record := records.get(execution_key(e))) is not None and record.get("ok")}
        if set(quality) != expected_quality:
            raise ValueError("qualified first score inventory omits or adds a successful cell")
    cells = build_cells(protocol, records, quality)
    groups = arm_groups(protocol, cells)
    ruler_macro = {}
    for arm in protocol["arms"]:
        selected = [(qid, c) for (dataset, qid, _, a), c in cells.items()
                    if dataset == "ruler4k" and a == arm]
        task_scores = defaultdict(list)
        if ruler_task_by_id and selected and all(q in ruler_task_by_id for q, _ in selected) and len({ruler_task_by_id[q] for q, _ in selected}) == 13 and all(
                c["quality"] is not None for _, c in selected):
            for qid, c in selected:
                task_scores[ruler_task_by_id[qid]].append(c["quality"]["score"])
        ruler_macro[arm] = statistics.mean(statistics.mean(v) for v in task_scores.values()) if len(task_scores) == 13 else None
    return {"schema": "v21_conditional_scored_summary_v1", "panel_kind": protocol["panel_kind"],
            "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
            "binding_sha256": binding_sha, "planned_executions": len(protocol["schedule"]),
            "recorded_executions": len(records), "executions_complete": len(records) == len(protocol["schedule"]),
            "quality_eligible": not diagnostic, "timing_eligible": not diagnostic,
            "block_completeness": block_completeness(protocol, records, cells),
            "groups": groups, "cells": cell_rows(protocol, cells),
            "paired_ratios": paired_ratios(protocol, cells),
            "ruler_official_13_task_macro_by_arm": ruler_macro,
            "ruler_task_macro_note": "RULER 13-task macro only when all selected task/seed first outputs have qualified scores; group scores are host-stratified, not a substitute for task macro",
            "scoring_note": "Unchanged v20 qualified scorer on successful first immutable receipts; gold and raw text excluded",
            "diagnostic_note": "CP1 first-only natural no-pruning work; quality and timing ineligible for method promotion" if diagnostic else None}


def score(protocol_path: Path, binding_path: Path, ledgers: list[Path], gold_paths: dict[str, Path] | None,
          ruler_root: Path | None, private_roots: dict[str, Path] | None) -> dict:
    protocol = _read(protocol_path)
    validate_protocol(protocol)
    binding = _read(binding_path)
    if (binding.get("schema") != "v21_conditional_binding_v1" or binding.get("status") != "frozen" or
            binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes())):
        raise ValueError("scorer binding/protocol byte identity drift")
    records = load_records(protocol, binding_path, ledgers)
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    if diagnostic:
        quality = {}
    else:
        # A panel without some task (v23 bootstrap6 defers RULER) binds only its own gold.
        gold_paths = ({d: p for d, p in gold_paths.items() if d in protocol["ids"]}
                      if gold_paths else gold_paths)
        if not gold_paths or not ruler_root or not private_roots or set(gold_paths) != set(protocol["ids"]):
            raise ValueError("eligible panel requires exact pinned gold/scorer paths and private receipt roots")
        if "source_identity" not in protocol:
            # v21/v23 panels inherit gold/scorer identity from the v20 protocol they pin by hash.
            from scripts.v21_run import V20_PROTOCOL
            raw = V20_PROTOCOL.read_bytes()
            if sha(raw) != protocol.get("v20_protocol_sha256"):
                raise ValueError("pinned v20 protocol identity drift")
            protocol = dict(protocol, source_identity=json.loads(raw)["source_identity"])
        from scripts.v27_datasets import EXTRA_GOLD, base_task
        long_sets = [d for d in protocol["ids"] if d in EXTRA_GOLD]
        gold = verify_sources(protocol, {d: p for d, p in gold_paths.items() if d not in long_sets})
        gold.update(_long_ruler_gold(protocol, gold_paths, ruler_root,
                                     [d for d in long_sets if base_task(d) == "ruler4k"]))
        gold.update(_long_lb_gold(protocol, gold_paths, [d for d in long_sets if base_task(d) == "longbench_v2"]))
        quality = score_firsts(protocol, records, gold, ruler_root=ruler_root, private_roots=private_roots)
    task_by_id = ({qid: row["task"] for qid, row in gold["ruler4k"].items()}
                  if not diagnostic and "ruler4k" in gold and protocol["ids"].get("ruler4k") else None)
    return summarize(protocol, records, quality, protocol_sha=sha(protocol_path.read_bytes()),
                     binding_sha=sha(binding_path.read_bytes()), ruler_task_by_id=task_by_id)


def _long_ruler_gold(protocol: dict, gold_paths: dict, ruler_root, datasets) -> dict:
    """v27 RULER 32K/64K: id -> outputs gold, sha pinned in the frozen protocol; the task base
    (official scorer key) comes from RULER's own synthetic.yaml for the task named in the id."""
    if not datasets:
        return {}
    import yaml
    config = yaml.safe_load((Path(ruler_root) / "scripts" / "synthetic.yaml").read_text())
    out = {}
    for dataset in datasets:
        raw = gold_paths[dataset].read_bytes()
        if sha(raw) != protocol.get("extra_gold_sha256", {}).get(dataset):
            raise ValueError(f"{dataset} gold identity drift")
        rows = json.loads(raw)
        table = {}
        for qid in protocol["ids"][dataset]:
            task = qid.split("/", 1)[1].split("_", 2)[2].rsplit("_p", 1)[0]
            table[qid] = dict(outputs=rows[qid], task_base=str(config[task]["task"]), task=task)
        out[dataset] = table
    return out


def _long_lb_gold(protocol: dict, gold_paths: dict, datasets) -> dict:
    """v27 LongBench-v2 32K/64K bins: scorer-only gold (id -> gold entry, the base LB format),
    sha pinned in the frozen protocol; scored by the unchanged v15 LB contract."""
    out = {}
    for dataset in datasets:
        raw = gold_paths[dataset].read_bytes()
        if sha(raw) != protocol.get("extra_gold_sha256", {}).get(dataset):
            raise ValueError(f"{dataset} gold identity drift")
        table = json.loads(raw)
        if not isinstance(table, dict) or not set(protocol["ids"][dataset]) <= set(table):
            raise ValueError(f"{dataset} scorer-only gold must cover the selected ids")
        out[dataset] = table
    return out


def write_redacted(summary: dict, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_bytes((json.dumps(summary, indent=2, sort_keys=True,
                                                             allow_nan=False) + "\n").encode())
    cell_columns = ("dataset", "id", "seed", "arm", "block", "cell_id", "host", "gpu_uuid",
                    "first_status", "first_error_class", "quality_eligible", "scored_first", "first_score",
                    "strict_correct", "task_correct", "task_at_cap", "parsed", "eos", "capped",
                    "termination", "output_tokens", "canvases", "decoder_calls", "mean_calls_per_canvas",
                    "stop_native_only", "stop_cap_only", "stop_both", "stop_neither", "stop_unrecorded",
                    "router_A", "router_D", "router_H", "first_cold_request_wall_s", "warm_status",
                    "warm_error_class", "warm_rejection_reasons", "accepted_warm_request_wall_s",
                    "accepted_warm_prefill_end_to_finish_cuda_span_s")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=cell_columns, lineterminator="\n")
    writer.writeheader()
    for row in summary["cells"]:
        flat = {k: row[k] for k in cell_columns}
        flat["warm_rejection_reasons"] = json.dumps(flat["warm_rejection_reasons"], separators=(",", ":"))
        writer.writerow(flat)
    (out_prefix.with_suffix(".csv")).write_bytes(stream.getvalue().encode())
    columns = ("dataset", "host", "arm", "unique_question_count", "planned_first_cells", "first_present", "first_success", "first_failed",
               "first_missing", "scored_first", "score_mean", "strict_correct", "task_correct", "task_at_cap",
               "warm_present", "warm_failed", "warm_accepted", "output_tokens_mean", "canvases_mean",
               "decoder_calls_mean", "warm_whole_request_s_mean", "decoder_calls_total", "canvases_total",
               "pooled_calls_per_canvas", "router_A", "router_D", "router_H")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for g in summary["groups"]:
        work = g["exact_work"]
        phase = g["router_phase_layer_calls"] or {}
        row = {k: g.get(k) for k in columns if k in g}
        row.update(output_tokens_mean=(g["output_tokens"] or {}).get("mean"),
                   canvases_mean=(g["canvases"] or {}).get("mean"),
                   decoder_calls_mean=(g["decoder_calls"] or {}).get("mean"),
                   warm_whole_request_s_mean=(g["warm_whole_request_s"] or {}).get("mean"),
                   decoder_calls_total=work["decoder_calls"], canvases_total=work["canvases"],
                   pooled_calls_per_canvas=work["pooled_calls_per_canvas"],
                   router_A=phase.get("A"), router_D=phase.get("D"), router_H=phase.get("H"))
        writer.writerow(row)
    out_prefix.with_name(out_prefix.name + ".groups.csv").write_bytes(stream.getvalue().encode())
    lines = ["# v21 冻结面板离线汇总（模板）", "",
             f"协议：`{summary['protocol_id']}`；类别：`{summary['panel_kind']}`。已记录 {summary['recorded_executions']}/{summary['planned_executions']} 次执行。",
             "", "质量仅统计成功的首次生成；时间仅统计严格接受的 warm 请求。绝对时间按 GPU 主机分开；同 GPU 配对比值小于 1 表示请求耗时较少。",
             "", "| 数据集 | 主机 | 方法 | 首次成功/计划 | 首次评分 | 严格正确 | warm 接受 | 平均 warm 请求秒 | 总调用 | 总 canvas |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        wall = (g["warm_whole_request_s"] or {}).get("mean")
        lines.append(f"| {g['dataset']} | {g['host']} | {g['arm']} | {g['first_success']}/{g['planned_first_cells']} | "
                     f"{g['scored_first']} | {g['strict_correct'] if g['strict_correct'] is not None else 'N/A'} | "
                     f"{g['warm_accepted'] if g['warm_accepted'] is not None else 'N/A'} | "
                     f"{format(wall, '.3f') if wall is not None else 'N/A'} | "
                     f"{g['exact_work']['decoder_calls']} | {g['exact_work']['canvases']} |")
    lines += ["", "请求的 CUDA 时间跨度从首次编码器 forward 结束至最终事件，包含主机间隙与后续工作，并非同步的纯解码墙钟。完整缺失/失败块与配对 bootstrap 描述性区间见 JSON。"]
    if summary["panel_kind"] == "zero_pruning_diagnostic":
        lines += ["", "此 12 次 CP1 自然诊断没有 warm 配对，质量及计时均不进入候选方法选择。"]
    (out_prefix.with_suffix(".md")).write_bytes(("\n".join(lines) + "\n").encode())


def gold_paths_from(ruler, aime, longbench, extra) -> dict[str, Path] | None:
    """Gold paths actually given (a long-RULER-only panel passes only --extra-gold); the scorer
    still requires them to equal the panel's datasets exactly."""
    paths = {d: Path(p) for d, p in (("ruler4k", ruler), ("aime26", aime), ("longbench_v2", longbench)) if p}
    paths.update({d: Path(p) for d, p in (x.split("=", 1) for x in extra)})
    return paths or None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--ledger", type=Path, action="append", required=True)
    p.add_argument("--private-roots", type=Path)
    p.add_argument("--ruler-gold", type=Path)
    p.add_argument("--aime-gold", type=Path)
    p.add_argument("--longbench-gold", type=Path)
    p.add_argument("--ruler-root", type=Path)
    p.add_argument("--extra-gold", action="append", default=[], help="v27 long RULER: dataset=path")
    p.add_argument("--out-prefix", type=Path, required=True)
    a = p.parse_args()
    gold_paths = gold_paths_from(a.ruler_gold, a.aime_gold, a.longbench_gold, a.extra_gold)
    private_roots = {h: Path(v) for h, v in _read(a.private_roots).items()} if a.private_roots else None
    result = score(a.protocol, a.binding, a.ledger, gold_paths, a.ruler_root, private_roots)
    write_redacted(result, a.out_prefix)
    print(json.dumps({"kind": result["panel_kind"], "recorded": result["recorded_executions"],
                      "planned": result["planned_executions"], "complete": result["executions_complete"]}))


if __name__ == "__main__":
    main()
