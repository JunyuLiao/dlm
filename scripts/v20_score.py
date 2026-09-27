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
from scripts.v20_panel import ARMS
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


def load_records(protocol: dict, binding_path: Path, ledger_paths: list[Path]) -> dict[str, dict]:
    expected = {execution_key(e): e for e in protocol["schedule"]}
    if len(expected) != 700:
        raise ValueError("duplicate or incomplete core schedule")
    binding_sha = sha(binding_path.read_bytes())
    found, starts = {}, []
    for path in ledger_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "start":
                if event.get("protocol_id") != protocol["protocol_id"] or event.get("binding_sha256") != binding_sha:
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
                if event.get("host") != assignment["host"] or event.get("gpu_uuid") != assignment["gpu_uuid"]:
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


def score_firsts(protocol: dict, records: dict[str, dict], gold: dict[str, dict], *, ruler_root: Path,
                 private_roots: dict[str, Path] | None = None) -> dict[str, dict]:
    from dllm.evaluation.ruler import official
    from scripts.v18_summarize import score_one
    from scripts import v15_longbench_task as lb_task

    official.verify_checkout(ruler_root)
    ruler_scorers = official.load_scorers(ruler_root)
    scored, lb_pending = {}, []
    for spec in protocol["schedule"]:
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
            scored[spec["cell_id"]] = score_one(dataset, gold[dataset][spec["id"]], receipt, record,
                                                 ruler_scorers=ruler_scorers)
    if lb_pending:
        results = lb_task.score([receipt["raw_completion"] for _, _, receipt in lb_pending],
                                [gold["longbench_v2"][spec["id"]] for spec, _, _ in lb_pending],
                                [record["termination"] for _, record, _ in lb_pending])
        for (spec, record, _), row in zip(lb_pending, results):
            scored[spec["cell_id"]] = dict(score=float(row["task_correct"]), correct=bool(row["task_correct"]),
                                           strict_correct=bool(row["strict_correct"]), parsed=bool(row["parsed"]),
                                           capped=record["termination"] == "length", eos=record["termination"] == "eos")
    return scored


def summarize_subset(protocol: dict, records: dict[str, dict], quality: dict[str, dict],
                     *, first84: bool) -> dict:
    schedule = protocol["schedule"][:84] if first84 else protocol["schedule"]
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
            gpu_timeline = [(r.get("phase_evidence") or {}).get("prefill_end_to_finish_gpu_s") for r in first_ok]
            gpu_timeline = [s for s in gpu_timeline if isinstance(s, (int, float)) and s > 0]
            cap_canvases = sum(sum(bool(x.get("iteration_cap")) for x in (r.get("phase_evidence") or {}).get("per_canvas", []))
                               for r in first_ok)
            arm_rows[arm] = dict(planned_cells=len(selected), first_present=sum(c["first"] is not None for c in selected),
                                 first_success=len(first_ok), first_failed=sum(c["first"] is not None and not c["first"].get("ok") for c in selected),
                                 scored=len(scores), score_mean=statistics.mean(scores) if scores else None,
                                 ruler_official_task_macro=(statistics.mean(scores) if dataset == "ruler4k" and
                                                             len({q for d, q, _, a in cells if d == dataset and a == arm}) == 13
                                                             and len(scores) == len(selected) else None),
                                 correct=sum(correct), parsed=sum(bool(c["quality"]["parsed"]) for c in selected if c["quality"]),
                                 termination=dict(sorted(term.items())), iteration_cap_canvases=cap_canvases,
                                 warm_accepted=len(warm_s), warm_request_s_mean=statistics.mean(warm_s) if warm_s else None,
                                 decoder_calls_total=sum(r.get("decoder_calls", 0) for r in first_ok),
                                 decoder_calls_per_request_mean=(statistics.mean(r["decoder_calls"] for r in first_ok) if first_ok else None),
                                 canvases_total=sum(r.get("canvases", 0) for r in first_ok),
                                 calls_per_canvas_distribution=dict(sorted(Counter(per_canvas).items())),
                                 output_tokens_total=sum(r.get("output_tokens", 0) for r in first_ok),
                                 first_encoder_forward_end_to_finish_gpu_timeline_s_mean=(
                                     statistics.mean(gpu_timeline) if gpu_timeline else None),
                                 gpu_timeline_note="CUDA event span includes host gaps and later commit work; not prefill-excluded generation",
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
                datasets=by_dataset, timing_note="accepted warm whole request; model load excluded; ratios method/reference <1 faster")


def summarize(protocol_path: Path, binding_path: Path, ledgers: list[Path], gold_paths: dict[str, Path],
              ruler_root: Path, *, private_roots: dict[str, Path] | None = None) -> dict:
    protocol = _read(protocol_path)
    if protocol.get("schema") != "v20_fan_panel_v1" or len(protocol.get("schedule", [])) != 700:
        raise ValueError("wrong frozen v20 panel")
    binding = _read(binding_path)
    if binding.get("status") != "frozen" or binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes()):
        raise ValueError("bound protocol identity mismatch")
    gold = verify_sources(protocol, gold_paths)
    records = load_records(protocol, binding_path, ledgers)
    quality = score_firsts(protocol, records, gold, ruler_root=ruler_root, private_roots=private_roots)
    return dict(protocol_id=protocol["protocol_id"], panel_sha256=sha(protocol_path.read_bytes()),
                binding_sha256=sha(binding_path.read_bytes()), first84=summarize_subset(protocol, records, quality, first84=True),
                full=summarize_subset(protocol, records, quality, first84=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "binding", "ruler-gold", "aime-gold", "longbench-gold", "ruler-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ledger", type=Path, action="append", required=True)
    p.add_argument("--private-roots", type=Path, help="JSON map of frozen host id to local private archive root")
    a = p.parse_args()
    private_roots = {host: Path(path) for host, path in _read(a.private_roots).items()} if a.private_roots else None
    result = summarize(a.protocol, a.binding, a.ledger,
                       {"ruler4k": a.ruler_gold, "aime26": a.aime_gold, "longbench_v2": a.longbench_gold},
                       a.ruler_root, private_roots=private_roots)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.out.exists() and a.out.read_text() != payload:
        raise ValueError("refusing to overwrite different scored summary")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(payload)
    print(json.dumps({k: {"recorded": result[k]["recorded_executions"], "complete": result[k]["executions_complete"]}
                      for k in ("first84", "full")}, sort_keys=True))


if __name__ == "__main__":
    main()
