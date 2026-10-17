"""Reproduce the October 17 tables from public numeric evidence, without CUDA.

This validates the committed scoring outputs and audit records. It cannot
rescore private generations or validate private prompt/model bytes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/v31_value_decision_reuse_20261015"
SUITES = ("aime26", "longbench_v2", "math500", "ruler128k")
CONTROL = "current_v31_control"
REFERENCES = ("dense_full_fix51994", "dense_piecewise", "allkept_fa4")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, separators=(",", ":"))
        else:
            result[name] = item
    return result


def csv_file(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def labels_for(arm):
    if arm == CONTROL:
        return [f"r{r}_control_{p}" for r in (1, 2) for p in ("before", "after")]
    if arm in REFERENCES:
        return [arm]
    return [f"r{r}_{arm}" for r in (1, 2)]


def check_arm(result, labels, records, scores, scale):
    rows = [r for label in labels for r in records[label]]
    require(result["problem_trials"] == len(rows), "unequal repetition count")
    expected_score = scale * statistics.mean(float(v) for label in labels for v in scores[label].values())
    require(math.isclose(result["score_points"], expected_score, abs_tol=1e-10), "score arithmetic")
    work = result["work"]
    for metric, field in (("N", "denoise_forwards"), ("C", "canvases"), ("T", "output_tokens")):
        total = sum(row[field] for row in rows)
        require(work[metric] == total, f"{metric} total")
        require(math.isclose(work[f"mean_{metric}"], total / len(rows), abs_tol=1e-10), f"mean {metric}")
    for field in ("global_eligible_tiles", "global_kept_tiles", "overall_eligible_rectangles", "overall_skipped_rectangles"):
        require(work[field] == sum(r["receipts"]["adapter"][field] for r in rows), "physical work")
    require(all(r["cuda_graph_captures"] == 0 for r in rows), "measured graph capture")
    require(all(r["receipts"]["adapter"]["native_local_skipped_tiles"] == 0 for r in rows), "LOCAL skip")
    require(all(r["receipts"]["adapter"].get("value_refinement_work", {}).get("budget_errors", 0) == 0 for r in rows), "budget error")


def publish(study, output):
    native = study / "native_replicated_20261016_attempt001"
    done = read(native / "final_complete.json")
    require(done["status"] == "complete", "native study incomplete")
    require(done["protocol_sha256"] == sha(native / "protocol.json"), "changed protocol")
    require(done["development_promotion_sha256"] == sha(native / "development_promotion.json"), "changed promotion")
    output.mkdir(parents=True, exist_ok=False)
    inputs, counts = {}, {}

    def load(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    native_rows, phase_rows, comparisons, repeats, noise, exposure = [], [], [], [], [], []
    lines = ["# October 17 native statistics", "", "Scores are repeated evaluations, not answer ensembles. RULER is the official task-average 0–100 score; other scores are accuracy %. All timings below are instrumented audit measurements. Clean timing remains paused.", ""]
    for suite in SUITES:
        report_path = native / "reports" / suite / "summary.json"
        require(done["report_sha256"][suite] == sha(report_path), "changed native report")
        report = load(report_path)
        scoring = native / "scoring" / suite / "attempt001"
        proof = load(scoring / "scoring_complete.json")
        require(proof["protocol_sha256"] == done["protocol_sha256"], "scorer protocol")
        for rel, digest in proof["outputs"].items():
            require(sha(scoring / rel) == digest, "changed scoring output")
        scores = load(scoring / "official.official.json")
        records = {}
        paths = list((native / "shared_runs" / suite).glob("shard*/repeat*/attempt*/exports/*/records.jsonl"))
        paths += list((native / "runs" / suite).glob("*/shard*/attempt*/records.jsonl"))
        for path in sorted(paths):
            inputs[str(path.relative_to(ROOT))] = sha(path)
            label = path.parent.name if path.parent.name != "attempt001" else path.parents[2].name
            records.setdefault(label, []).extend(json.loads(line) for line in path.read_text().splitlines())
        require(set(scores) == set(records) and len(scores) == 15, "score/record labels")
        expected = read(native / "protocol.json")["suites"][suite]["count"]
        for label, rr in records.items():
            keys = [f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}" for r in rr]
            require(len(keys) == len(set(keys)) == expected and set(keys) == set(scores[label]), "cohort coverage")
        counts[suite] = dict(cells=expected, labels=len(scores), scored_trials=expected * len(scores))
        lines += [f"## {suite}", "", "| Arm | Score | Trials | N/trial | Median N | p95 N | N/C | C/trial | T/trial | Cap % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, result in report["summaries"].items():
            check_arm(result, labels_for(arm), records, scores, 1 if suite == "ruler128k" else 100)
            work = result["work"]
            native_rows.append(dict(suite=suite, arm=arm, **flatten(result)))
            for phase, data in work["phases"].items():
                phase_rows.append(dict(suite=suite, arm=arm, phase=phase, problem_trials=result["problem_trials"], **flatten(data),
                                       eligible_per_trial=data["eligible_tiles"] / result["problem_trials"], kept_per_trial=data["kept_tiles"] / result["problem_trials"]))
            lines.append(f"| {arm} | {result['score_points']:.3f} | {result['problem_trials']} | {work['mean_N']:.3f} | {work['median_N']:.2f} | {result['p95_N']:.2f} | {work['N_per_C']:.3f} | {work['mean_C']:.3f} | {work['mean_T']:.2f} | {100*result['cap_rate']:.2f} |")
        lines += ["", "| Candidate | Metric | Difference | Problem/task 95% CI | Engine-block 95% CI | Repeat 1 | Repeat 2 |", "|---|---|---:|---|---|---:|---:|"]
        for item in report["comparisons"]:
            comparisons.append(dict(suite=suite, **flatten(item)))
            for i, metric in enumerate(item["metrics"]):
                value = item["problem_or_task_cluster"]["delta"][i]
                ci = item["problem_or_task_cluster"]["ci95"][i]
                ec = item["paired_engine_block"]["ci95"][i]
                lines.append(f"| {item['arm']} | {metric} | {value:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | [{ec[0]:.4f}, {ec[1]:.4f}] | {item['replicate_delta']['1'][i]:+.4f} | {item['replicate_delta']['2'][i]:+.4f} |")
        lines += ["", "Instrumented audit cost only; requests and generation spans differ from synchronized clean timing.", "", "| Arm | Audit generation S (s) | Audit request W (s) | Audit S/N (s) | GLOBAL sparsity % | Overall sparsity % |", "|---|---:|---:|---:|---:|---:|"]
        for arm, result in report["summaries"].items():
            w = result["work"]
            lines.append(f"| {arm} | {w['audit_decode_s']:.3f} | {w['audit_wall_s']:.3f} | {w['audit_S_per_N']:.5f} | {100*w['global_sparsity']:.3f} | {100*w['overall_sparsity']:.3f} |")
            for item in report["separate_repeat_positions"][arm]:
                repeats.append(dict(suite=suite, arm=arm, **item))
        noise.append(dict(suite=suite, **flatten(report["control_noise"])))
        diagnostic = load(native / "reports" / suite / "paired_trajectory_diagnostics.json")
        for arm, value in diagnostic["methods"].items():
            exposure.append(dict(suite=suite, arm=arm, **flatten(value)))
        lines += [""]
    outputs = {"native_results.csv": native_rows, "native_phases.csv": phase_rows, "native_comparisons.csv": comparisons,
               "native_repeats.csv": repeats, "native_control_noise.csv": noise, "native_trajectory_exposure.csv": exposure}
    for filename, rows in outputs.items():
        csv_file(output / filename, rows)
    for filename, source in (("operator_screen.csv", "operator_summary.json"), ("whole_layer.csv", "whole_layer/summary.json"),
                             ("reuse_guard.csv", "reuse_guard/summary.json")):
        csv_file(output / filename, [flatten(row) for row in load(study / source)["metrics"]])
    csv_file(output / "reuse_guard_by_layer.csv", [flatten(row) for row in load(study / "reuse_guard/layer_analysis.json")])
    decisions = []
    for panel, source in (("initial", "decision_analysis/summary.json"), ("coverage", "coverage_analysis.json")):
        for suite, data in load(study / source)["workloads"].items():
            for row in data["aggregates"]:
                decisions.append(dict(panel=panel, suite=suite, **flatten(row)))
    csv_file(output / "same_state_decisions.csv", decisions)
    (output / "native_statistics.md").write_text("\n".join(lines))
    result = dict(status="complete_public_analysis", source_sha256=sha(Path(__file__)),
                  inputs_sha256=inputs, cohorts=counts, timing="paused",
                  scope="Public numeric arithmetic, coverage, receipts and scoring-output hash verification; no private generation rescoring or private input/model verification",
                  outputs_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    require(sum(c["scored_trials"] for c in counts.values()) == 5370, "planned generation count")
    with (output / "final_complete.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "cohorts": counts, "output_files": len(result["outputs_sha256"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.study.resolve(), args.output.resolve())
