"""Offline, gold-separated v20 scorer for first84 and full paired schedules."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v18_protocol import sha
from scripts.v20_panel import ARMS, HISTORICAL
from scripts.v20_run import strict_v20_warm


SCORER_FILES = {
    "ruler4k": "src/dllm/evaluation/ruler/official.py",
    "aime26": "experiments/diffusion_gemma_aime26_modes/protocol.py",
    "longbench_v2": "scripts/v15_longbench_task.py",
}
CONTRACT_FILES = {
    "ruler4k": "scripts/v18_protocol.py",
    "aime26": "scripts/v18_protocol.py",
    "longbench_v2": "results/numerical_qk_longcontext_scored_20260926/panel_selection.json",
}


def _read(path: Path):
    return json.loads(path.read_text())


def verify_sources(protocol: dict, gold_paths: dict[str, Path]) -> dict[str, dict]:
    root = Path(__file__).resolve().parents[1]
    gold = {}
    for dataset, path in gold_paths.items():
        expected = protocol["source_identity"]["datasets"][dataset]
        if sha(path.read_bytes()) != expected["gold_sha256"]:
            raise ValueError(f"{dataset} gold identity drift")
        if sha((root / SCORER_FILES[dataset]).read_bytes()) != expected["scorer_sha256"]:
            raise ValueError(f"{dataset} scorer source drift")
        if sha((root / CONTRACT_FILES[dataset]).read_bytes()) != expected["task_contract_sha256"]:
            raise ValueError(f"{dataset} task contract drift")
        raw = _read(path)
        if dataset == "longbench_v2":
            if not isinstance(raw, dict):
                raise ValueError("LB scorer-only gold must be a dictionary")
            gold[dataset] = raw
        else:
            if not isinstance(raw, list):
                raise ValueError("RULER/AIME original gold source must be a row list")
            gold[dataset] = {r["id"]: r for r in raw}
        if not set(protocol["ids"][dataset]) <= set(gold[dataset]):
            raise ValueError(f"{dataset} selected gold IDs missing")
    return gold


def load_records(protocol: dict, binding_path: Path, ledger_paths: list[Path], *, historical: bool = False) -> dict[str, dict]:
    schedule = protocol["historical_extension"]["schedule"] if historical else protocol["schedule"]
    expected = {execution_key(e): e for e in schedule}
    if len(expected) != (100 if historical else 700):
        raise ValueError("duplicate or incomplete frozen schedule")
    if historical and (protocol["historical_extension"].get("arm") != HISTORICAL or
                       any(e["arm"] != HISTORICAL for e in schedule)):
        raise ValueError("historical arm differs from frozen extension")
    binding_sha = sha(binding_path.read_bytes())
    found, starts = {}, []
    for path in ledger_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "start":
                protocol_id = protocol["protocol_id"] + ("/historical" if historical else "")
                if event.get("protocol_id") != protocol_id or event.get("binding_sha256") != binding_sha:
                    raise ValueError("worker protocol/binding identity drift")
                starts.append(event)
            elif event.get("event") == "run":
                key = event.get("execution_key")
                if key not in expected or key in found:
                    raise ValueError("foreign or duplicate execution")
                spec = expected[key]
                if any(event.get(k) != spec[k] for k in ("arm", "id", "seed", "block", "role", "repeat", "cell_id")):
                    raise ValueError("run differs from frozen schedule")
                assignment = protocol["block_assignments"][str(spec["block"])]
                if (event.get("host") != assignment["host"] or event.get("gpu_uuid") != assignment["gpu_uuid"] or
                    (historical and (spec.get("host"), spec.get("gpu_uuid")) !=
                     (assignment["host"], assignment["gpu_uuid"]))):
                    raise ValueError("run on wrong host/GPU")
                if event.get("generation_seed") != spec["seed"]:
                    raise ValueError("actual generation seed differs")
                found[key] = event
    if found and not starts:
        raise ValueError("no worker source/identity start receipt")
    return found


def _first_receipt(record: dict, spec: dict, private_roots: dict[str, Path] | None = None) -> dict:
    path = record.get("private_receipt")
    if not path:
        raise ValueError("successful first execution lacks immutable private receipt")
    path = Path(path)
    if private_roots and record["host"] in private_roots:
        if path.name != "attempt00.json":
            raise ValueError("first receipt filename differs from frozen execution role")
        path = private_roots[record["host"]] / "cells" / spec["cell_id"] / path.name
    if not path.is_file():
        raise ValueError("successful first execution lacks immutable private receipt")
    receipt = _read(path)
    if (receipt.get("id"), receipt.get("seed")) != (spec["id"], spec["seed"]):
        raise ValueError("receipt identity mismatch")
    if receipt.get("fingerprint") != record.get("fingerprint") or receipt.get("prompt_token_hash") != record.get("prompt_token_hash"):
        raise ValueError("receipt config/prompt identity mismatch")
    if sha(json.dumps(receipt["completion_tokens"], separators=(",", ":"))) != record.get("completion_token_hash"):
        raise ValueError("first token hash mismatch")
    if [c["decoder_calls"] for c in receipt["per_canvas"]] != record.get("per_canvas_calls"):
        raise ValueError("first per-canvas calls mismatch")
    if receipt.get("termination_reason") != record.get("termination"):
        raise ValueError("first termination mismatch")
    if receipt.get("total_decoder_calls") != record.get("decoder_calls") or receipt.get("output_tokens") != record.get("output_tokens"):
        raise ValueError("first work/output count mismatch")
    return receipt


def cluster_interval(by_question: dict[str, list[float]], *, seed: int, resamples: int = 2000,
                     transform=lambda x: x) -> tuple[float | None, list[float] | None]:
    """Question-cluster exploratory interval; seed repeats stay within their question."""
    values = [statistics.mean(rows) for _, rows in sorted(by_question.items()) if rows]
    if not values:
        return None, None
    point = transform(statistics.mean(values))
    rng = random.Random(seed)
    boots = sorted(transform(statistics.mean(rng.choice(values) for _ in values)) for _ in range(resamples))
    return point, [boots[int(.025 * (resamples - 1))], boots[int(.975 * (resamples - 1))]]


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    def quantile(p):
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return dict(n=len(ordered), min=ordered[0], median=quantile(.5), p90=quantile(.9), max=ordered[-1])


def score_firsts(protocol: dict, records: dict[str, dict], gold: dict[str, dict], *, ruler_root: Path,
                 private_roots: dict[str, Path] | None = None,
                 schedule: list[dict] | None = None) -> dict[str, dict]:
    from dllm.evaluation.ruler import official
    from scripts.v18_summarize import score_one
    from scripts import v15_longbench_task as lb_task

    official.verify_checkout(ruler_root)
    ruler_scorers = official.load_scorers(ruler_root)
    scored, lb_pending = {}, []
    for spec in protocol["schedule"] if schedule is None else schedule:
        if spec["role"] != "attempt0":
            continue
        record = records.get(execution_key(spec))
        if record is None or not record.get("ok"):
            continue
        receipt = _first_receipt(record, spec, private_roots)
        dataset = spec["dataset"]
        if dataset == "longbench_v2":
            lb_pending.append((spec, record, receipt))
        else:
            row = score_one(dataset, gold[dataset][spec["id"]], receipt, record,
                            ruler_scorers=ruler_scorers)
            if dataset == "aime26":
                from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
                answer = gold[dataset][spec["id"]]
                expected = answer.get("expected", answer.get("answer"))
                row["task_correct"] = bool(numeric_score(final_response(receipt["raw_completion"], True),
                                                          str(expected))["correct"])
            else:
                row["task_correct"] = bool(row["correct"])
            row["strict_correct"] = bool(row["task_correct"] and row["eos"])
            scored[spec["cell_id"]] = row
    if lb_pending:
        results = lb_task.score([receipt["raw_completion"] for _, _, receipt in lb_pending],
                                [gold["longbench_v2"][spec["id"]] for spec, _, _ in lb_pending],
                                [record["termination"] for _, record, _ in lb_pending])
        for (spec, record, _), row in zip(lb_pending, results):
            scored[spec["cell_id"]] = dict(score=float(row["task_correct"]), correct=bool(row["task_correct"]),
                                           task_correct=bool(row["task_correct"]),
                                           strict_correct=bool(row["strict_correct"]), parsed=bool(row["parsed"]),
                                           capped=record["termination"] == "length", eos=record["termination"] == "eos")
    return scored


def summarize_subset(protocol: dict, records: dict[str, dict], quality: dict[str, dict],
                     *, first84: bool) -> dict:
    schedule = protocol["schedule"][:84] if first84 else protocol["schedule"]
    by_block = defaultdict(list)
    for spec in schedule:
        by_block[spec["block"]].append(spec)
    block_counts = Counter()
    failed_blocks, partial_blocks = [], []
    for block, entries in sorted(by_block.items()):
        rows = [(spec, records.get(execution_key(spec))) for spec in entries]
        missing = any(record is None for _, record in rows)
        failed = any(record is not None and not record.get("ok") for _, record in rows)
        firsts = {spec["cell_id"]: record for spec, record in rows if spec["role"] == "attempt0"}
        warms = {spec["cell_id"]: record for spec, record in rows if spec["role"] == "warm"}
        if len(entries) == 2 * len(ARMS) and not missing:
            block_counts["recorded_all14_blocks"] += 1
        if len(firsts) == len(ARMS) and all(row is not None and row.get("ok") for row in firsts.values()):
            block_counts["successful_first_all7_blocks"] += 1
        strict_pairs = (len(firsts) == len(warms) == len(ARMS) and
                        all(firsts[cell] is not None and warm is not None and
                            strict_v20_warm(firsts[cell], warm)["accepted"] and
                            warm.get("acceptance", {}).get("accepted")
                            for cell, warm in warms.items()))
        if strict_pairs:
            block_counts["strictwarm_all7_blocks"] += 1
        if any(record is not None and spec["role"] == "warm" and
               not record.get("acceptance", {}).get("accepted") for spec, record in rows):
            failed = True
        if failed:
            failed_blocks.append(block)
        if missing:
            partial_blocks.append(block)
        if len(entries) == 2 * len(ARMS) and not missing and not failed and strict_pairs:
            block_counts["complete_valid_pair_blocks"] += 1
    groups = defaultdict(dict)
    for spec in schedule:
        groups[(spec["dataset"], spec["id"], spec["seed"], spec["arm"])][spec["role"]] = (
            spec, records.get(execution_key(spec)))
    cells = {}
    for (dataset, qid, seed, arm), pair in groups.items():
        first_spec, first = pair.get("attempt0", (None, None))
        _, warm = pair.get("warm", (None, None))
        accepted = bool(first and warm and strict_v20_warm(first, warm)["accepted"] and
                        warm.get("acceptance", {}).get("accepted"))
        if warm and warm.get("acceptance") and warm["acceptance"] != strict_v20_warm(first, warm):
            raise ValueError("warm acceptance drift")
        cells[(dataset, qid, seed, arm)] = dict(first=first, warm=warm,
                                                quality=quality.get(first_spec["cell_id"]),
                                                warm_s=warm.get("api_wall_s") if accepted else None)
    by_dataset = {}
    for dataset in protocol["ids"]:
        arm_rows = {}
        for arm in ARMS:
            selected = [c for (d, _, _, a), c in cells.items() if d == dataset and a == arm]
            scores = [c["quality"]["score"] for c in selected if c["quality"] is not None]
            correct = [c["quality"]["correct"] for c in selected if c["quality"] is not None]
            first_ok = [c["first"] for c in selected if c["first"] and c["first"].get("ok")]
            warm_s = [c["warm_s"] for c in selected if c["warm_s"] is not None]
            per_canvas = [n for r in first_ok for n in r.get("per_canvas_calls", [])]
            term = Counter(r.get("termination", "unknown") for r in first_ok)
            by_host_latency = defaultdict(lambda: dict(warm_request_s=[], prefill_end_to_finish_gpu_s=[]))
            for c in selected:
                if c["warm_s"] is None:
                    continue
                warm = c["warm"]
                host = warm["host"]
                by_host_latency[host]["warm_request_s"].append(c["warm_s"])
                span = (warm.get("phase_evidence") or {}).get("prefill_end_to_finish_gpu_s")
                if type(span) in (int, float) and math.isfinite(span) and span > 0:
                    by_host_latency[host]["prefill_end_to_finish_gpu_s"].append(span)
            host_latency = {host: dict(accepted_warm_requests=len(v["warm_request_s"]),
                                       warm_whole_request_s_mean=statistics.mean(v["warm_request_s"]),
                                       warm_prefill_end_to_finish_cuda_event_span_s_mean=(
                                           statistics.mean(v["prefill_end_to_finish_gpu_s"])
                                           if v["prefill_end_to_finish_gpu_s"] else None),
                                       qualified_cuda_spans=len(v["prefill_end_to_finish_gpu_s"]))
                            for host, v in sorted(by_host_latency.items())}
            cap_canvases = sum(sum(bool(x.get("iteration_cap")) for x in (r.get("phase_evidence") or {}).get("per_canvas", []))
                               for r in first_ok)
            native_stop_canvases = sum(sum(bool(x.get("native_stop")) for x in (r.get("phase_evidence") or {}).get("per_canvas", []))
                                       for r in first_ok)
            router_totals = Counter()
            router_present = 0
            for r in first_ok:
                evidence = r.get("router_phase_evidence")
                if not evidence:
                    continue
                if all(type(evidence.get(k)) is int for k in ("A", "D", "H")):
                    router_present += 1
                    router_totals.update({k: evidence[k] for k in ("A", "D", "H")})
            fallback = [(r.get("counters") or {}).get("unsupported_mask_refreshes") for r in first_ok]
            fallback = [x for x in fallback if type(x) is int]
            total_canvases = sum(r.get("canvases", 0) for r in first_ok)
            total_calls = sum(r.get("decoder_calls", 0) for r in first_ok)
            early_counts = dict(call0=sum(n >= 1 for n in per_canvas), call1=sum(n >= 2 for n in per_canvas),
                                call2plus=sum(max(0, n - 2) for n in per_canvas))
            scored_rows = [c["quality"] for c in selected if c["quality"] is not None]
            eos_wrong = sum(bool(q.get("eos")) and not bool(q.get("strict_correct", q["correct"])) for q in scored_rows)
            task_at_cap = sum(bool(q.get("capped")) and bool(q.get("task_correct", q["correct"])) for q in scored_rows)
            arm_rows[arm] = dict(planned_cells=len(selected), first_present=sum(c["first"] is not None for c in selected),
                                 first_success=len(first_ok), first_failed=sum(c["first"] is not None and not c["first"].get("ok") for c in selected),
                                 first_missing=sum(c["first"] is None for c in selected),
                                 warm_failed=sum(c["warm"] is not None and not c["warm"].get("ok") for c in selected),
                                 scored=len(scores), score_mean=statistics.mean(scores) if scores else None,
                                 ruler_official_task_macro=(statistics.mean(scores) if dataset == "ruler4k" and
                                                             len({q for d, q, _, a in cells if d == dataset and a == arm}) == 13
                                                             and len(scores) == len(selected) else None),
                                 correct=sum(correct), parsed=sum(bool(c["quality"]["parsed"]) for c in selected if c["quality"]),
                                 strict_correct=sum(bool(q.get("strict_correct", q["correct"] and q.get("eos"))) for q in scored_rows),
                                 task_at_cap=task_at_cap, eos_wrong=eos_wrong,
                                 unparsed=sum(not bool(q["parsed"]) for q in scored_rows),
                                 termination=dict(sorted(term.items())), request_output_cap=term["length"],
                                 iteration_cap_canvases=cap_canvases, native_stop_canvases=native_stop_canvases,
                                 warm_accepted=len(warm_s), absolute_latency_by_host=host_latency,
                                 decoder_calls_total=total_calls,
                                 decoder_calls_per_request_mean=(statistics.mean(r["decoder_calls"] for r in first_ok) if first_ok else None),
                                 canvases_total=total_canvases,
                                 decoder_calls_per_canvas_pooled=(total_calls / total_canvases if total_canvases else None),
                                 decoder_calls_per_canvas_per_request=_distribution([
                                     r["decoder_calls"] / r["canvases"] for r in first_ok if r.get("canvases", 0) > 0]),
                                 calls_per_canvas_distribution=dict(sorted(Counter(per_canvas).items())),
                                 decoder_call_positions=early_counts,
                                 router_phase_layer_calls=(dict(A=router_totals["A"], D=router_totals["D"],
                                                                H=router_totals["H"], measured_requests=router_present)
                                                           if router_present else None),
                                 unsupported_mask_refreshes_total=(sum(fallback) if len(fallback) == len(first_ok) and fallback else None),
                                 phase_by_early_call_position="N/A: aggregate request counters do not identify per-call router phase",
                                 output_tokens_total=sum(r.get("output_tokens", 0) for r in first_ok),
                                 output_tokens_per_canvas_pooled=(sum(r.get("output_tokens", 0) for r in first_ok) / total_canvases
                                                                  if total_canvases else None),
                                 output_tokens_per_canvas_per_request=_distribution([
                                     r["output_tokens"] / r["canvases"] for r in first_ok if r.get("canvases", 0) > 0]),
                                 gpu_timeline_note="accepted warm CUDA event span from first actual encoder forward end to final event after generate; includes host launch gaps and later encoder/commit work, not synchronized prefill-excluded generation wall",
                                 initial_prefill_excluded_generation_s="N/A: no qualified boundary")
        ratios = {}
        for arm in ARMS:
            if arm == "D_native":
                continue
            for reference in dict.fromkeys(("D_native", "T_scope", "B_A8_matched")):
                if arm == reference:
                    continue
                matched = []
                score_by_question = defaultdict(list)
                time_by_question = defaultdict(list)
                for (d, qid, seed, a), c in cells.items():
                    if d != dataset or a != arm:
                        continue
                    other = cells.get((d, qid, seed, reference))
                    if not other:
                        continue
                    f, g = c["first"], other["first"]
                    if f and g and (f["host"], f["gpu_uuid"]) != (g["host"], g["gpu_uuid"]):
                        raise ValueError("timing pair crossed GPUs")
                    if c["quality"] is not None and other["quality"] is not None:
                        score_by_question[qid].append(c["quality"]["score"] - other["quality"]["score"])
                    if c["warm_s"] is None or other["warm_s"] is None:
                        continue
                    matched.append((f["host"], c["warm_s"], other["warm_s"], f.get("decoder_calls"), g.get("decoder_calls")))
                    time_by_question[qid].append(math.log(c["warm_s"] / other["warm_s"]))
                per_host = {}
                for host in sorted({x[0] for x in matched}):
                    rows = [x for x in matched if x[0] == host]
                    tm, tr = sum(x[1] for x in rows), sum(x[2] for x in rows)
                    cm, cr = sum(x[3] for x in rows), sum(x[4] for x in rows)
                    per_host[host] = dict(paired_cells=len(rows), total_request_time_ratio=tm / tr,
                                          geometric_request_time_ratio=math.exp(statistics.mean(math.log(x[1] / x[2]) for x in rows)),
                                          decoder_call_ratio=cm / cr if cr else None,
                                          amortized_wall_per_call_ratio=(tm / cm) / (tr / cr) if cm and cr else None)
                draw_seed = int(sha(f"v20/{dataset}/{arm}/{reference}")[:12], 16)
                score_delta, score_ci = cluster_interval(score_by_question, seed=draw_seed)
                time_ratio, time_ci = cluster_interval(time_by_question, seed=draw_seed + 1, transform=math.exp)
                ratios[f"{arm}/{reference}"] = dict(paired_cells=len(matched), by_host=per_host,
                                                      geometric_ratio=(math.exp(statistics.mean(math.log(x[1] / x[2]) for x in matched))
                                                                       if matched else None),
                                                      paired_score_delta=score_delta, paired_score_delta_95_exploratory=score_ci,
                                                      question_cluster_time_ratio=time_ratio,
                                                      question_cluster_time_ratio_95_exploratory=time_ci,
                                                      uncertainty_note="question-cluster bootstrap across seed repeats; small development panel, not noninferiority")
        by_dataset[dataset] = dict(questions=len({q for d, q, _, _ in cells if d == dataset}),
                                   seeds=sorted({s for d, _, s, _ in cells if d == dataset}),
                                   arms=arm_rows, paired_request_time_ratios=ratios,
                                   score_note="RULER official partial-credit task scores require task macro; no pooled cross-dataset accuracy")
    return dict(schema="v20_fan_scored_summary_v1", subset="first84" if first84 else "full700",
                planned_executions=len(schedule), recorded_executions=sum(execution_key(e) in records for e in schedule),
                executions_complete=all(execution_key(e) in records for e in schedule),
                planned_blocks=len(by_block), recorded_all14_blocks=block_counts["recorded_all14_blocks"],
                successful_first_all7_blocks=block_counts["successful_first_all7_blocks"],
                strictwarm_all7_blocks=block_counts["strictwarm_all7_blocks"],
                complete_valid_pair_blocks=block_counts["complete_valid_pair_blocks"],
                failed_block_ids=failed_blocks, partial_block_ids=partial_blocks,
                block_completeness_note="ledger-row completeness includes failures; valid paired blocks require seven successful first outputs and seven accepted warm executions",
                datasets=by_dataset, timing_note="accepted warm whole request; model load excluded; ratios method/reference <1 faster")


def summarize_historical(protocol: dict, binding: dict, core_records: dict[str, dict],
                         historical_records: dict[str, dict], core_quality: dict[str, dict],
                         historical_quality: dict[str, dict]) -> dict:
    """Separate G75/native comparison on frozen question-seed/GPU pairs."""
    if binding.get("historical_qualified") is not True or binding.get("historical_scope") != "ALL_NATIVE_LEGAL":
        raise ValueError("historical transplant lacks independently qualified ALL_NATIVE_LEGAL scope")
    extension = protocol["historical_extension"]
    schedule = extension["schedule"]
    if extension.get("arm") != HISTORICAL or len(schedule) != 100:
        raise ValueError("historical extension inventory drift")
    native = {(e["dataset"], e["id"], e["seed"], e["role"]): e for e in protocol["schedule"]
              if e["arm"] == "D_native"}
    if len(native) != 100:
        raise ValueError("native comparator inventory drift")
    grouped = defaultdict(dict)
    for spec in schedule:
        key = (spec["dataset"], spec["id"], spec["seed"], spec["role"])
        reference = native.get(key)
        if reference is None or (spec["host"], spec["gpu_uuid"], spec["block"]) != (
                reference["host"], reference["gpu_uuid"], reference["block"]):
            raise ValueError("historical/native comparison is not frozen on the same question-seed GPU")
        grouped[(spec["dataset"], spec["id"], spec["seed"])][spec["role"]] = (spec, reference)
    if len(grouped) != 50 or any(set(pair) != {"attempt0", "warm"} for pair in grouped.values()):
        raise ValueError("historical first/warm pairing drift")
    output = {}
    for dataset in protocol["ids"]:
        selected = [(key, pair) for key, pair in grouped.items() if key[0] == dataset]
        scores, native_scores, paired_score_deltas, ratios = [], [], [], []
        host_times = defaultdict(list)
        first_present = first_success = first_failed = warm_present = warm_accepted = 0
        native_warm_accepted = paired_quality = paired_time = 0
        strict_correct = task_at_cap = eos_wrong = unparsed = 0
        missing_blocks, failed_blocks = [], []
        for (_, qid, seed), pair in selected:
            first_spec, ref_first_spec = pair["attempt0"]
            warm_spec, ref_warm_spec = pair["warm"]
            first = historical_records.get(execution_key(first_spec))
            warm = historical_records.get(execution_key(warm_spec))
            ref_first = core_records.get(execution_key(ref_first_spec))
            ref_warm = core_records.get(execution_key(ref_warm_spec))
            if first is not None: first_present += 1
            if warm is not None: warm_present += 1
            if first is not None and first.get("ok"): first_success += 1
            if first is not None and not first.get("ok"): first_failed += 1
            if first is None or warm is None:
                missing_blocks.append(first_spec["block"])
            for row in (first, warm, ref_first, ref_warm):
                if row and (row["host"], row["gpu_uuid"]) != (first_spec["host"], first_spec["gpu_uuid"]):
                    raise ValueError("historical/native receipt crossed GPUs")
            if any(row is not None and not row.get("ok") for row in (first, warm)):
                failed_blocks.append(first_spec["block"])
            hist_q = historical_quality.get(first_spec["cell_id"])
            native_q = core_quality.get(ref_first_spec["cell_id"])
            if hist_q is not None and (first is None or not first.get("ok")):
                raise ValueError("historical quality lacks a successful first receipt")
            if native_q is not None and (ref_first is None or not ref_first.get("ok")):
                raise ValueError("native quality lacks a successful first receipt")
            if hist_q is not None:
                scores.append(hist_q["score"])
                strict_correct += bool(hist_q.get("strict_correct", hist_q["correct"] and hist_q.get("eos")))
                task_at_cap += bool(hist_q.get("task_correct", hist_q["correct"])) and bool(hist_q.get("capped"))
                eos_wrong += bool(hist_q.get("eos")) and not bool(hist_q.get("strict_correct", hist_q["correct"]))
                unparsed += not bool(hist_q["parsed"])
            if native_q is not None:
                native_scores.append(native_q["score"])
            if hist_q is not None and native_q is not None:
                paired_quality += 1
                paired_score_deltas.append(hist_q["score"] - native_q["score"])
            def accepted(a, b):
                if b is None:
                    return False
                expected = strict_v20_warm(a, b)
                if b.get("acceptance") and b["acceptance"] != expected:
                    raise ValueError("historical/native warm acceptance drift")
                return expected["accepted"] and b.get("acceptance", {}).get("accepted") is True
            hist_ok, native_ok = accepted(first, warm), accepted(ref_first, ref_warm)
            warm_accepted += hist_ok
            native_warm_accepted += native_ok
            if hist_ok:
                host_times[first_spec["host"]].append(warm["api_wall_s"])
            if hist_ok and native_ok:
                paired_time += 1
                ratios.append((first_spec["host"], warm["api_wall_s"], ref_warm["api_wall_s"]))
        by_host = {}
        for host in sorted({x[0] for x in ratios} | set(host_times)):
            matched = [x for x in ratios if x[0] == host]
            by_host[host] = dict(accepted_historical_warm=len(host_times[host]),
                                 historical_warm_whole_request_s_mean=(statistics.mean(host_times[host])
                                                                      if host_times[host] else None),
                                 paired_native_warm_cells=len(matched),
                                 paired_total_request_ratio=(sum(x[1] for x in matched) / sum(x[2] for x in matched)
                                                             if matched else None),
                                 paired_geometric_request_ratio=(math.exp(statistics.mean(math.log(x[1] / x[2])
                                                                                           for x in matched)) if matched else None))
        output[dataset] = dict(planned_cells=len(selected), first_present=first_present,
                               first_success=first_success, first_failed=first_failed,
                               first_missing=len(selected) - first_present, warm_present=warm_present,
                               warm_accepted=warm_accepted, native_warm_accepted=native_warm_accepted,
                               paired_quality_cells=paired_quality, paired_accepted_warm_cells=paired_time,
                               historical_score_mean=statistics.mean(scores) if scores else None,
                               native_score_mean=statistics.mean(native_scores) if native_scores else None,
                               paired_score_delta_mean=(statistics.mean(paired_score_deltas)
                                                        if paired_score_deltas else None),
                               ruler_official_task_macro=(statistics.mean(scores) if dataset == "ruler4k" and
                                                          len(selected) == 26 and len(scores) == 26 else None),
                               strict_correct=strict_correct, task_at_cap=task_at_cap,
                               eos_wrong=eos_wrong, unparsed=unparsed,
                               missing_block_ids=sorted(set(missing_blocks)),
                               failed_block_ids=sorted(set(failed_blocks)), by_host=by_host,
                               same_gpu_request_time_ratio=(math.exp(statistics.mean(math.log(x[1] / x[2])
                                                                                     for x in ratios)) if ratios else None))
    return dict(schema="v20_historical_nativeQ128_separate_v1", arm=HISTORICAL,
                historical_scope="ALL_NATIVE_LEGAL", core_scope=binding["scope"],
                planned_executions=100, recorded_executions=sum(execution_key(e) in historical_records for e in schedule),
                execution_rows_complete=all(execution_key(e) in historical_records for e in schedule),
                datasets=output, temporal_note="historical/native first and warm requests share frozen question-seed GPU; extension runs after core and may have temporal drift",
                accounting_note="historical 100 executions are outside core700 and first84; ratios require both accepted warm requests")


def summarize(protocol_path: Path, binding_path: Path, ledgers: list[Path], gold_paths: dict[str, Path],
              ruler_root: Path, *, private_roots: dict[str, Path] | None = None,
              historical_ledgers: list[Path] | None = None,
              historical_private_roots: dict[str, Path] | None = None) -> dict:
    protocol = _read(protocol_path)
    if protocol.get("schema") != "v20_fan_panel_v1" or len(protocol.get("schedule", [])) != 700:
        raise ValueError("wrong frozen v20 panel")
    binding = _read(binding_path)
    if binding.get("status") != "frozen" or binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes()):
        raise ValueError("bound protocol identity mismatch")
    gold = verify_sources(protocol, gold_paths)
    records = load_records(protocol, binding_path, ledgers)
    quality = score_firsts(protocol, records, gold, ruler_root=ruler_root, private_roots=private_roots)
    result = dict(protocol_id=protocol["protocol_id"], panel_sha256=sha(protocol_path.read_bytes()),
                binding_sha256=sha(binding_path.read_bytes()), first84=summarize_subset(protocol, records, quality, first84=True),
                full=summarize_subset(protocol, records, quality, first84=False))
    if historical_ledgers is not None:
        if historical_private_roots is None:
            raise ValueError("historical private archive roots must be explicit")
        historical_records = load_records(protocol, binding_path, historical_ledgers, historical=True)
        historical_quality = score_firsts(protocol, historical_records, gold, ruler_root=ruler_root,
                                          private_roots=historical_private_roots,
                                          schedule=protocol["historical_extension"]["schedule"])
        result["historical_extension"] = summarize_historical(protocol, binding, records, historical_records,
                                                              quality, historical_quality)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "binding", "ruler-gold", "aime-gold", "longbench-gold", "ruler-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ledger", type=Path, action="append", required=True)
    p.add_argument("--private-roots", type=Path, help="JSON map of frozen host id to local private archive root")
    p.add_argument("--historical-ledger", type=Path, action="append",
                   help="separate optional G75L30_nativeQ128 worker ledger")
    p.add_argument("--historical-private-roots", type=Path,
                   help="JSON map of host id to separate historical private archive root")
    a = p.parse_args()
    private_roots = {host: Path(path) for host, path in _read(a.private_roots).items()} if a.private_roots else None
    historical_roots = ({host: Path(path) for host, path in _read(a.historical_private_roots).items()}
                        if a.historical_private_roots else None)
    result = summarize(a.protocol, a.binding, a.ledger,
                       {"ruler4k": a.ruler_gold, "aime26": a.aime_gold, "longbench_v2": a.longbench_gold},
                       a.ruler_root, private_roots=private_roots,
                       historical_ledgers=a.historical_ledger, historical_private_roots=historical_roots)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.out.exists() and a.out.read_text() != payload:
        raise ValueError("refusing to overwrite different scored summary")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(payload)
    print(json.dumps({k: {"recorded": result[k]["recorded_executions"], "complete": result[k]["executions_complete"]}
                      for k in ("first84", "full")}, sort_keys=True))


if __name__ == "__main__":
    main()
