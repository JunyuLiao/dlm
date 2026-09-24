#!/usr/bin/env python3
"""Render an answer-free, four-question native reuse smoke report.

The summarizer is the source of quality and timing rows. Dispatch audits only
add GLOBAL/LOCAL physical work for exact matching quality attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = ("native_dense", "fresh_junyu_T", "M1", "M3")
EXPECTED_QUESTIONS = 4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1729


def _load(path: Path, schema: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema!r}")
    return data


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(rows: list[dict], key: str, unit: str = "", digits: int = 1) -> str:
    values = [row.get(key) for row in rows]
    if not values or any(not _number(value) for value in values):
        return "N/A"
    return f"{statistics.mean(values):,.{digits}f}{unit}"


def _sum(rows: list[dict], key: str) -> int | None:
    values = [row.get(key) for row in rows]
    if not values or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return None
    return sum(values)


def _count(rows: list[dict], key: str) -> str:
    values = [row.get(key) for row in rows]
    if not values or any(not isinstance(value, bool) for value in values):
        return "N/A"
    return f"{sum(values)}/{len(values)}"


def _correct(rows: list[dict]) -> str:
    values = [row.get("correct") for row in rows]
    if not values or any(not isinstance(value, bool) for value in values):
        return "N/A"
    return f"{sum(values)}/{len(values)}"


def _integer(value: int | None) -> str:
    return f"{value:,}" if value is not None else "N/A"


def _percent(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "N/A"
    return f"{100 * numerator / denominator:.1f}%"


def _audit_matches(row: dict, audit: dict) -> bool:
    counters = audit.get("counters", {})
    physical = audit.get("physical_decision_units", {})
    global_units = physical.get("global", {})
    local_units = physical.get("local", {})
    if audit.get("dispatch_audit") != "PASS":
        return False
    if (audit.get("id"), audit.get("seed"), audit.get("arm"), audit.get("source_fingerprint")) != (
        row.get("question"), row.get("seed"), row.get("arm"), row.get("source_fingerprint")
    ):
        return False
    if audit.get("per_canvas_calls") != row.get("per_canvas_calls"):
        return False
    if audit.get("decoder_calls") != row.get("total_calls"):
        return False
    if counters.get("current_qk_elements") != row.get("current_qk_elements"):
        return False
    if counters.get("reused_qk_elements") != row.get("reused_score_elements"):
        return False
    for field in ("score_refresh_calls", "decision_refresh_calls"):
        if counters.get(field) != row.get(field):
            return False
    for field in ("eligible", "skipped"):
        if not all(isinstance(part.get(field), int) for part in (global_units, local_units)):
            return False
    return (global_units["eligible"] + local_units["eligible"] == row.get("pv_physical_eligible")
            and global_units["skipped"] + local_units["skipped"] == row.get("pv_physical_skipped"))


def _bootstrap_pairs(pairs: list[dict], key: str, seed: int) -> str:
    by_question: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        value = pair.get(key)
        if _number(value) and value > 0:
            by_question[str(pair["question"])].append(math.log(value))
    if not by_question:
        return "N/A"
    question_logs = [statistics.mean(values) for _, values in sorted(by_question.items())]
    point = math.exp(statistics.mean(question_logs))
    rng = random.Random(seed)
    draws = sorted(math.exp(statistics.mean(rng.choices(question_logs, k=len(question_logs))))
                   for _ in range(BOOTSTRAP_DRAWS))
    lo = draws[int(0.025 * (BOOTSTRAP_DRAWS - 1))]
    hi = draws[int(0.975 * (BOOTSTRAP_DRAWS - 1))]
    return f"{point:.2f}× [{lo:.2f}, {hi:.2f}]"


def _key(row: dict) -> tuple:
    return row.get("question"), row.get("seed"), row.get("arm"), row.get("source_fingerprint")


def render(summary: dict, audit_documents: list[dict], summary_path: Path | None = None,
           audit_paths: list[Path] | None = None) -> str:
    if summary.get("schema") != "numerical_qk_summary_v1":
        raise ValueError("expected numerical_qk_summary_v1")
    rows = summary.get("records")
    if not isinstance(rows, list):
        raise ValueError("summary.records must be a list")
    seen: set[tuple] = set()
    logical_attempts: set[tuple] = set()
    for row in rows:
        logical_key = row.get("question"), row.get("seed"), row.get("arm")
        if row.get("arm") not in ARMS or _key(row) in seen or logical_key in logical_attempts:
            raise ValueError("unexpected arm or duplicate quality attempt")
        seen.add(_key(row))
        logical_attempts.add(logical_key)
    if len({row.get("question") for row in rows}) > EXPECTED_QUESTIONS:
        raise ValueError("this report is limited to the four-question smoke")
    if len({row.get("seed") for row in rows}) > 1:
        raise ValueError("this report is limited to the one-seed smoke")
    audits: dict[tuple, dict] = {}
    for document in audit_documents:
        if document.get("schema") != "numerical_reuse_dispatch_audit_v1":
            raise ValueError("expected numerical_reuse_dispatch_audit_v1")
        for audit in document.get("records", []):
            key = (audit.get("id"), audit.get("seed"), audit.get("arm"), audit.get("source_fingerprint"))
            if key in audits:
                raise ValueError("duplicate dispatch audit")
            audits[key] = audit
    matched_audits = {key: audits[key] for key in seen & audits.keys()
                      if _audit_matches(next(row for row in rows if _key(row) == key), audits[key])}
    rejected_audits = len(seen & audits.keys()) - len(matched_audits)
    unmatched_audits = len(audits.keys() - seen)

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    lines = ["# Native QK reuse: four-question smoke report", "",
             f"Quality scope: {len(set(row['question'] for row in rows))}/{EXPECTED_QUESTIONS} questions, "
             f"{len(rows)} unique attempt-0 receipts; one seed in the checkpoint where present. "
             "Missing attempts are incomplete data, not failures. Timing retries never enter the quality denominator.", "",
             "| Arm | Coverage | Correct | Caps | Unparsed | Mean whole wall (s) | Mean device timeline (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        arm_rows = by_arm[arm]
        coverage = f"{len(arm_rows)}/{EXPECTED_QUESTIONS}"
        lines.append(f"| {arm} | {coverage} | {_correct(arm_rows)} | {_count(arm_rows, 'capped')} | "
                     f"{_count(arm_rows, 'unparsed')} | {_mean(arm_rows, 'whole_wall_seconds')} | "
                     f"{_mean(arm_rows, 'device_generation_timeline_seconds')} |")
    lines.extend(["", "Device timeline starts after initial encoder prefill and ends after generation; "
                  "it includes host gaps and later encoder/commit work. Missing event measurements are N/A. "
                  "CPU generation and time-between-tokens (including burst events) are N/A.", "",
                  "| Arm | Output tokens (mean) | Decoder calls (total; mean/request) | Canvases (total; mean/request) | True per-canvas decoder calls (median [min, max]) |",
                  "|---|---:|---:|---:|---:|"])
    for arm in ARMS:
        arm_rows = by_arm[arm]
        calls = _sum(arm_rows, "total_calls")
        canvases = _sum(arm_rows, "canvas_count")
        per_canvas = [call for row in arm_rows for call in row.get("per_canvas_calls", [])]
        per_canvas_text = (f"{statistics.median(per_canvas):g} [{min(per_canvas)}, {max(per_canvas)}]"
                           if per_canvas else "N/A")
        lines.append(f"| {arm} | {_mean(arm_rows, 'output_tokens', digits=0)} | "
                     f"{_integer(calls)}; {_mean(arm_rows, 'total_calls')} | "
                     f"{_integer(canvases)}; {_mean(arm_rows, 'canvas_count')} | {per_canvas_text} |")

    lines.extend(["", "QK work is counted in score elements. Physical PV skip is counted in eligible block decisions; "
                  "neither is a measured DRAM-byte or instruction reduction.", "",
                  "| Arm | QK reused / (current + reused) | Physical PV skipped / eligible | GLOBAL PV skip | LOCAL PV skip | Cache refresh / decision refresh |",
                  "|---|---:|---:|---:|---:|---:|"])
    for arm in ARMS:
        arm_rows = by_arm[arm]
        current = _sum(arm_rows, "current_qk_elements")
        reused = _sum(arm_rows, "reused_score_elements")
        eligible = _sum(arm_rows, "pv_physical_eligible")
        skipped = _sum(arm_rows, "pv_physical_skipped")
        refresh = _sum(arm_rows, "score_refresh_calls")
        decisions = _sum(arm_rows, "decision_refresh_calls")
        accepted = [matched_audits[_key(row)] for row in arm_rows if _key(row) in matched_audits]
        physical = [audit["physical_decision_units"] for audit in accepted]
        def subset(direction: str) -> str:
            if not physical:
                return "N/A"
            numerator = sum(item[direction]["skipped"] for item in physical)
            denominator = sum(item[direction]["eligible"] for item in physical)
            return f"{_percent(numerator, denominator)} ({len(physical)}/{len(arm_rows)} audited)"
        qk = _percent(reused, current + reused) if current is not None and reused is not None else "N/A"
        cache = f"{_integer(refresh)} / {_integer(decisions)}" if refresh is not None and decisions is not None else "N/A"
        lines.append(f"| {arm} | {qk} | {_percent(skipped, eligible)} | {subset('global')} | "
                     f"{subset('local')} | {cache} |")

    timing_rows = summary.get("summary", {}).get("timing_controls", [])
    timing_pairs = summary.get("summary", {}).get("paired_timing_controls", [])
    valid_pairs = [pair for pair in timing_pairs if pair.get("cross_load_token_match") is True
                   and pair.get("cross_load_call_match") is True and pair.get("performance_qualified") is False]
    lines.extend(["", f"Separate dense timing controls: {len(timing_rows)} retry receipts; "
                  "the original dense quality rows above remain unchanged. All reported comparisons are "
                  "exploratory because cold/JIT state is unknown.",
                  f"Dense timing-control absolute means: whole wall {_mean(timing_rows, 'whole_wall_seconds', ' s', 3)}; "
                  f"initial-prefill-excluded device timeline {_mean(timing_rows, 'device_generation_timeline_seconds', ' s', 3)}.", "",
                  "| Matched comparison | Questions | Whole-wall dense retry / method (geomean [bootstrap 95% CI]) | Device-timeline dense retry / method (geomean [bootstrap 95% CI]) |",
                  "|---|---:|---:|---:|"])
    for arm in ARMS[1:]:
        pairs = [pair for pair in valid_pairs if pair.get("arm") == arm]
        lines.append(f"| dense timing control vs {arm} | {len(set(pair['question'] for pair in pairs))}/{EXPECTED_QUESTIONS} | "
                     f"{_bootstrap_pairs(pairs, 'whole_wall_speedup', BOOTSTRAP_SEED)} | "
                     f"{_bootstrap_pairs(pairs, 'device_timeline_speedup', BOOTSTRAP_SEED + 1)} |")
    lines.extend(["", f"Dispatch audits matched exactly to quality attempts: {len(matched_audits)}; "
                  f"unmatched: {unmatched_audits}; rejected on consistency: {rejected_audits}. "
                  "GLOBAL/LOCAL percentages use only matched audits. "
                  "The dispatch audit verifies calls and routing, not physical memory traffic.", "",
                  "Limits: four questions and one seed do not establish quality noninferiority or a stable speedup. "
                  "Pinned Junyu applies a query-relative LOCAL lower window bound; installed native SDPA attends its supplied truncated prefix plus canvas. "
                  "T/M1/M3 inherit the Junyu rule, so dense/T differences cannot all be attributed to numerical reuse. "
                  "CUDA graph performance is unqualified. The report is numerical only; it does not infer method success."])
    if summary_path is not None:
        digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        lines.extend(["", f"Source summary SHA-256: `{digest}`."])
    for path in audit_paths or []:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"Dispatch audit SHA-256: `{digest}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dispatch-audit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = _load(args.summary, "numerical_qk_summary_v1")
    audits = [_load(path, "numerical_reuse_dispatch_audit_v1") for path in args.dispatch_audit]
    rendered = render(summary, audits, args.summary, args.dispatch_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
