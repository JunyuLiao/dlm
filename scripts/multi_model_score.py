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
    from dllm.evaluation.ruler.official import load_scorers
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", default="analysis001")
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    out = root / a.output
    out.mkdir(parents=True, exist_ok=False)
    numeric, numeric_source = aime_numeric_score()
    scorer = load_scorers("reference/RULER")
    he.sandbox_preflight()
    tables = collections.defaultdict(list)
    sources = []
    for marker in sorted((root / "runs").glob("*/*/*/shard*/attempt*/final_complete.json")):
        leaf = marker.parent
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
                code, detail = he.extract_code("<channel|>final\n" + text, row["gold"]["entry_point"])
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
    for (model, suite, arm), rows in tables.items():
        keys = [(r["id"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate records {model}/{suite}/{arm}")
        expected = 90 if suite == "aime26" else 164 if suite == "humaneval" else 130
        if not a.allow_partial and len(rows) != expected:
            raise ValueError(f"incomplete {model}/{suite}/{arm}: {len(rows)}/{expected}")
        counts = collections.Counter()
        for r in rows:
            counts.update(r["counts"])
        n = counts["denoising_forwards"]
        c = counts["canvases"]
        summary = {"requests": len(rows), "expected": expected,
                   "accuracy": statistics.mean(r["quality"] for r in rows),
                   "full_credit": sum(r["quality"] == 1. for r in rows),
                   "N": n, "C": c, "pooled_N_per_C": n / c,
                   "mean_request_N_per_C": statistics.mean(r["counts"]["denoising_forwards"] / r["counts"]["canvases"] for r in rows),
                   "T": sum(r["output_tokens"] for r in rows),
                   "finished": sum(r["finished"] for r in rows),
                   "consumer_sparsity": 1. - counts["kept_tiles"] / counts["eligible_tiles"],
                   "counts": dict(counts)}
        summaries[f"{model}/{suite}/{arm}"] = summary
    paired = {}
    rng = np.random.default_rng(738)
    for (model, suite, arm), rows in tables.items():
        reference = tables.get((model, suite, "mass_topk_reuse_r4"))
        if not reference or arm == "mass_topk_reuse_r4":
            continue
        ref = {(r["id"], r["seed"]): r for r in reference}
        clusters = collections.defaultdict(list)
        for r in rows:
            if (r["id"], r["seed"]) in ref:
                other = ref[(r["id"], r["seed"])]
                clusters[r["id"]].append([r["quality"] - other["quality"],
                                          r["counts"]["denoising_forwards"] - other["counts"]["denoising_forwards"]])
        matrix = np.array([np.mean(v, axis=0) for v in clusters.values()])
        boot = matrix[rng.integers(len(matrix), size=(10000, len(matrix)))].mean(1)
        paired[f"{model}/{suite}/{arm}"] = {"questions": len(matrix),
            "accuracy_delta": float(matrix[:, 0].mean()),
            "accuracy_delta_ci95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
            "calls_delta": float(matrix[:, 1].mean()),
            "calls_delta_ci95": np.quantile(boot[:, 1], [.025, .975]).tolist()}
    for name, data in [("summary.json", summaries), ("paired_vs_v31.json", paired), ("sources.json", sources)]:
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
