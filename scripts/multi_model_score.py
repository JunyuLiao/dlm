"""Official task scoring and paired question-clustered campaign aggregation."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def final_text(text):
    # No Gemma channel markers are required on these model families.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    return text


def main():
    from scripts.v31_official import aime_numeric_score
    from scripts import v27_humaneval as he
    from dllm.evaluation.ruler.official import load_scorers, verify_checkout
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", default="analysis001")
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    out = root / a.output
    out.mkdir(parents=True, exist_ok=False)
    numeric, numeric_source = aime_numeric_score()
    ruler_source = verify_checkout("reference/RULER")
    scorer = load_scorers("reference/RULER")
    he.sandbox_preflight()
    scoring_identity = {
        "aggregation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aime_numeric": numeric_source,
        "humaneval_source_sha256": hashlib.sha256(Path(he.__file__).read_bytes()).hexdigest(),
        "humaneval_extractor": he.EXTRACTOR,
        "humaneval_program": he.PROGRAM,
        "humaneval_sandbox": he.SANDBOX,
        "ruler": ruler_source,
        "bootstrap_seed": 738, "bootstrap_resamples": 10000,
        "inference_unit": "problem_cluster",
        "final_text_rule": "after_last_closed_think_or_raw; unclosed_think_empty",
    }
    (out / "scoring_identity.json").write_text(json.dumps(scoring_identity, indent=2) + "\n")
    tables = collections.defaultdict(list)
    sources = []
    accepted_shards = set()
    for marker in sorted((root / "runs").glob("*/*/*/shard*/attempt*/final_complete.json")):
        leaf = marker.parent
        shard_identity = str(leaf.parent.relative_to(root))
        if shard_identity in accepted_shards:
            continue  # First complete immutable attempt is the canonical shard.
        accepted_shards.add(shard_identity)
        config = json.loads((leaf / "config.json").read_text())
        dataset = json.loads((root / "private/data" / config["model"] / (config["suite"] + ".json")).read_text())
        byid = {r["id"]: r for r in dataset}
        records = [json.loads(line) for line in (leaf / "private/generation.private.jsonl").read_text().splitlines()]
        for r in records:
            row = byid[r["id"]]
            text = final_text(r["text"])
            suite = config["suite"]
            if suite == "aime26":
                quality = float(numeric(text, row["gold"])["correct"])
                detail = "numeric_exact"
            elif suite == "humaneval":
                code, detail = he.extract_code("<channel|>" + text, row["gold"]["entry_point"])
                quality = float(he.run_test(he.program_for(row["gold"], code))[0]) if code else 0.
            else:
                quality = scorer[row["task_base"]]([row.get("answer_prefix", "") + text], [row["gold"]]) / 100.
                detail = "official_ruler"
            sanitized = {k: v for k, v in r.items() if k not in ("text", "tokens")}
            sanitized.update(quality=quality, score_detail=detail, arm=config["arm"],
                             model=config["model"], suite=suite)
            tables[(config["model"], suite, config["arm"])].append(sanitized)
        sources.append({"config_sha256": hashlib.sha256((leaf / "config.json").read_bytes()).hexdigest(),
                        "records_sha256": hashlib.sha256((leaf / "records.jsonl").read_bytes()).hexdigest()})
    summaries = {}
    ruler_tasks = {}
    expected_panels = {(m, s, arm)
                       for m in ("llada22", "ultrallada", "trado")
                       for s in (("aime26", "humaneval") if m == "trado" else ("aime26", "humaneval", "ruler128k"))
                       for arm in ("dense_native", "allkept_fa4", "mass_topk_reuse_r4", "value_v2_gaussian32_reuse_r4", "value_v3a_singleton_reuse_r4")}
    if not a.allow_partial and set(tables) != expected_panels:
        raise ValueError(f"missing complete panels: {sorted(expected_panels - set(tables))}")
    for (model, suite, arm), rows in tables.items():
        keys = [(r["id"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate records {model}/{suite}/{arm}")
        expected = 90 if suite == "aime26" else 164 if suite == "humaneval" else 130
        if not a.allow_partial and len(rows) != expected:
            raise ValueError(f"incomplete {model}/{suite}/{arm}: {len(rows)}/{expected}")
        counts = collections.Counter()
        problem_outputs = collections.defaultdict(set)
        for r in rows:
            counts.update(r["counts"])
            problem_outputs[r["id"]].add(r["output_sha256"])
        n = counts["denoising_forwards"]
        c = counts["canvases"]
        consumer_sparsity = 1. - counts["kept_tiles"] / counts["eligible_tiles"]
        summary = {"requests": len(rows), "expected": expected,
                   "accuracy": statistics.mean(r["quality"] for r in rows),
                   "full_credit": sum(r["quality"] == 1. for r in rows),
                   "unique_problems": len(problem_outputs),
                   "problems_with_distinct_outputs_across_seeds": sum(len(v) > 1 for v in problem_outputs.values()),
                   "repeated_same_problem_output_requests": len(rows) - sum(len(v) for v in problem_outputs.values()),
                   "N": n, "C": c, "pooled_N_per_C": n / c,
                   "mean_request_N": n / len(rows),
                   "mean_request_C": c / len(rows),
                   "mean_request_N_per_C": statistics.mean(r["counts"]["denoising_forwards"] / r["counts"]["canvases"] for r in rows),
                   "T": sum(r["output_tokens"] for r in rows),
                   "mean_request_T": statistics.mean(r["output_tokens"] for r in rows),
                   "pooled_output_tokens_per_denoising_call": sum(r["output_tokens"] for r in rows) / n,
                   "finished": sum(r["finished"] for r in rows),
                   "without_eos_at_cap": sum(not r["finished"] and r["output_tokens"] == r["generation_cap"] for r in rows),
                   "consumer_sparsity": consumer_sparsity,
                   "sparse_target_range": [.48, .52] if "reuse" in arm else None,
                   "sparse_target_satisfied": .48 <= consumer_sparsity <= .52 if "reuse" in arm else None,
                   "panel_complete": len(rows) == expected,
                   "counts": dict(counts)}
        summaries[f"{model}/{suite}/{arm}"] = summary
        if suite == "ruler128k":
            dataset = json.loads((root / "private/data" / model / "ruler128k.json").read_text())
            task_ids = {r["id"]: r["task"] for r in dataset}
            by_task = collections.defaultdict(list)
            for row in rows:
                by_task[task_ids[row["id"]]].append(row)
            ruler_tasks[f"{model}/{arm}"] = {
                task: {"requests": len(values), "official_mean_score": statistics.mean(r["quality"] for r in values),
                       "full_credit": sum(r["quality"] == 1. for r in values),
                       "mean_N": statistics.mean(r["counts"]["denoising_forwards"] for r in values),
                       "mean_C": statistics.mean(r["counts"]["canvases"] for r in values),
                       "mean_T": statistics.mean(r["output_tokens"] for r in values)}
                for task, values in sorted(by_task.items())}
    paired = {}
    rng = np.random.default_rng(738)
    for (model, suite, arm), rows in tables.items():
        for ref_arm in ("mass_topk_reuse_r4", "dense_native", "allkept_fa4"):
            reference = tables.get((model, suite, ref_arm))
            if not reference or arm == ref_arm:
                continue
            ref = {(r["id"], r["seed"]): r for r in reference}
            clusters = collections.defaultdict(list)
            wins = losses = 0
            for r in rows:
                if (r["id"], r["seed"]) in ref:
                    other = ref[(r["id"], r["seed"])]
                    delta = r["quality"] - other["quality"]
                    wins += delta > 0
                    losses += delta < 0
                    rn, rc = r["counts"]["denoising_forwards"], r["counts"]["canvases"]
                    on, oc = other["counts"]["denoising_forwards"], other["counts"]["canvases"]
                    clusters[r["id"]].append([delta, rn - on, rc - oc,
                                              r["output_tokens"] - other["output_tokens"],
                                              np.log((rn / rc) / (on / oc)), np.log(rn / on)])
            matrix = np.array([np.mean(v, axis=0) for v in clusters.values()])
            boot = matrix[rng.integers(len(matrix), size=(10000, len(matrix)))].mean(1)
            sparse_arms = [x for x in (arm, ref_arm) if "reuse" in x]
            paired[f"{model}/{suite}/{arm}_vs_{ref_arm}"] = {"questions": len(matrix),
                "both_panels_complete": all(summaries[f"{model}/{suite}/{x}"]["panel_complete"] for x in (arm, ref_arm)),
                "all_sparse_arms_at_target": all(summaries[f"{model}/{suite}/{x}"]["sparse_target_satisfied"] for x in sparse_arms) if sparse_arms else None,
                "accuracy_delta": float(matrix[:, 0].mean()),
                "accuracy_delta_ci95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
                "calls_delta": float(matrix[:, 1].mean()),
                "calls_delta_ci95": np.quantile(boot[:, 1], [.025, .975]).tolist(),
                "canvases_delta": float(matrix[:, 2].mean()),
                "canvases_delta_ci95": np.quantile(boot[:, 2], [.025, .975]).tolist(),
                "output_tokens_delta": float(matrix[:, 3].mean()),
                "output_tokens_delta_ci95": np.quantile(boot[:, 3], [.025, .975]).tolist(),
                "paired_geomean_N_per_C_ratio": float(np.exp(matrix[:, 4].mean())),
                "paired_geomean_N_per_C_ratio_ci95": np.exp(np.quantile(boot[:, 4], [.025, .975])).tolist(),
                "paired_geomean_N_ratio": float(np.exp(matrix[:, 5].mean())),
                "paired_geomean_N_ratio_ci95": np.exp(np.quantile(boot[:, 5], [.025, .975])).tolist(),
                "quality_wins": int(wins), "quality_losses": int(losses)}
    for name, data in [("summary.json", summaries), ("paired_vs_v31.json", paired),
                       ("ruler_task_summary.json", ruler_tasks), ("sources.json", sources)]:
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    (out / "scored_records.jsonl").write_text("".join(json.dumps(r) + "\n" for rows in tables.values() for r in rows))
    lines = ["# Three-model value evaluation", "", "Instrumented execution; no clean latency claim.", "",
             "| Model / task / arm | Score | N | C | N/C | T | Consumer sparsity |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, r in sorted(summaries.items()):
        lines.append(f"| {key} | {100*r['accuracy']:.2f}% ({r['requests']}) | {r['N']} | {r['C']} | {r['pooled_N_per_C']:.3f} | {r['T']} | {100*r['consumer_sparsity']:.2f}% |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "analysis_complete.json").write_text(json.dumps({"status": "complete" if not a.allow_partial else "partial", "panels": len(summaries)}) + "\n")


if __name__ == "__main__":
    main()
