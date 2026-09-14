"""Strict report regeneration from completed paired condition shards."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .config import conditions


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ratio(n: float, d: float) -> float:
    return n / d if d else 0.0


def _token_agreement(a: list[int], b: list[int]) -> float:
    length = max(len(a), len(b))
    return _ratio(sum(index < len(a) and index < len(b) and a[index] == b[index] for index in range(length)), length) if length else 1.0


def _blasst_aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for row in rows for call in (row.get("blasst_steps") or [])]
    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = sum(int(item.get("eligible_tiles", 0)) for item in items); skipped = sum(int(item.get("skipped_tiles", 0)) for item in items)
        elements = sum(int(item.get("valid_elements", 0)) for item in items); skipped_elements = sum(int(item.get("skipped_valid_elements", 0)) for item in items)
        weights = [int(item.get("retained_attention_mass_rows", 0)) for item in items]
        mass = sum(float(item.get("retained_dense_attention_mass", 0) or 0) * w for item, w in zip(items, weights))
        values = [float(item["effective_blasst_lambda"]) for item in items if item.get("effective_blasst_lambda") not in (None, "")]
        return {"eligible_tiles": eligible, "skipped_tiles": skipped, "retained_tiles": eligible - skipped,
            "full_tile_sparsity": _ratio(skipped, eligible), "valid_qk_element_sparsity": _ratio(skipped_elements, elements),
            "retained_dense_attention_mass": _ratio(mass, sum(weights)), "calls": len(items),
            "effective_lambda_min": min(values) if values else None, "effective_lambda_max": max(values) if values else None,
            "effective_lambda_mean": float(np.mean(values)) if values else None}
    return {"overall": group(calls), "local": group([x for x in calls if x.get("attention_type") == "local"]), "global": group([x for x in calls if x.get("attention_type") == "global"])}


def _sol_aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for row in rows for call in (row.get("routing_stats") or {}).get("per_call", [])]
    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = sum(int(item.get("physical_total_tiles", 0)) for item in items); skipped = sum(int(item.get("physical_skipped_tiles", 0)) for item in items)
        weights = [int(item.get("valid_rows", 0)) for item in items]; mass = sum(float(item.get("retained_dense_attention_mass", 0) or 0) * w for item, w in zip(items, weights))
        return {"eligible_tiles": total, "skipped_tiles": skipped, "retained_tiles": total-skipped,
            "full_tile_sparsity": _ratio(skipped, total), "retained_dense_attention_mass": _ratio(mass, sum(weights)), "calls": len(items)}
    return {"overall": group(calls), "local": group([x for x in calls if x.get("attention_type") == "local"]), "global": group([x for x in calls if x.get("attention_type") == "global"])}


def build(root: Path) -> dict[str, Any]:
    expected = conditions(); all_rows = {}
    for condition in expected:
        path = root / "conditions" / condition.name / "predictions.jsonl"
        if not path.exists(): raise RuntimeError(f"missing {path}")
        rows = _read(path)
        if len(rows) != 80 or len({row["id"] for row in rows}) != 80: raise RuntimeError(f"{condition.name}: expected 80 unique rows")
        if any(row.get("correct") is None for row in rows): raise RuntimeError(f"{condition.name}: LiveCodeBench has not been graded")
        all_rows[condition.name] = sorted(rows, key=lambda row: row["id"])
    dense = all_rows["dense"]; identity = [(x["id"], x["prompt_hash"], x["seed"]) for x in dense]
    if not all([(x["id"], x["prompt_hash"], x["seed"]) for x in rows] == identity for rows in all_rows.values()): raise RuntimeError("paired prompt/seed audit failed")
    summaries = []
    for condition in expected:
        rows = all_rows[condition.name]; scores = np.asarray([bool(row["correct"]) for row in rows], float)
        by_benchmark = {bench: float(np.mean([bool(row["correct"]) for row in rows if row["benchmark"] == bench])) for bench in sorted({row["benchmark"] for row in rows})}
        agreements = [_token_agreement(a["completion_tokens"], b["completion_tokens"]) for a, b in zip(dense, rows)]
        if condition.method == "dense": routing = {scope: {"full_tile_sparsity": 0.0, "retained_dense_attention_mass": 1.0, "skipped_tiles": 0, "eligible_tiles": 0} for scope in ("overall", "local", "global")}
        elif condition.method == "sol": routing = _sol_aggregate(rows)
        else: routing = _blasst_aggregate(rows)
        summaries.append({"condition": condition.name, "method": condition.method, "target_sparsity": condition.target_sparsity, "beta": condition.beta,
            "accuracy_micro": float(scores.mean()), "accuracy_macro_benchmark": float(np.mean(list(by_benchmark.values()))), "accuracy_by_benchmark": by_benchmark,
            "token_agreement": float(np.mean(agreements)), "sequence_exact": float(np.mean([a["completion_tokens"] == b["completion_tokens"] for a,b in zip(dense,rows)])), "routing": routing,
            "mean_completion_tokens": float(np.mean([len(row["completion_tokens"]) for row in rows]))})
    audit = {"passed": True, "conditions": len(all_rows), "rows_per_condition": 80, "paired_prompts_and_seeds": True, "livecodebench_graded": True,
        "sparsity_definition": "sum(skipped eligible physical tiles) / sum(eligible physical tiles)"}
    result = {"schema_version": 1, "study": "DiffusionGemma Sol-Attn vs BLASST compact multibench", "samples": 80, "benchmarks": ["ruler16k", "aime24", "longbench_v2", "livecodebench_v6"], "conditions": summaries, "audit": audit}
    (root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _write_outputs(root, summaries)
    return result


def _write_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["condition", "method", "target_sparsity", "beta", "accuracy_micro", "accuracy_macro_benchmark", "ruler16k", "aime24", "longbench_v2", "livecodebench_v6", "whole_sparsity", "local_sparsity", "global_sparsity", "retained_mass", "token_agreement", "lambda_local", "lambda_global"]
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            routing = row["routing"]
            writer.writerow({"condition": row["condition"], "method": row["method"], "target_sparsity": row["target_sparsity"], "beta": row["beta"], "accuracy_micro": row["accuracy_micro"], "accuracy_macro_benchmark": row["accuracy_macro_benchmark"], **row["accuracy_by_benchmark"], "whole_sparsity": routing["overall"]["full_tile_sparsity"], "local_sparsity": routing["local"]["full_tile_sparsity"], "global_sparsity": routing["global"]["full_tile_sparsity"], "retained_mass": routing["overall"]["retained_dense_attention_mass"], "token_agreement": row["token_agreement"], "lambda_local": routing["local"].get("effective_lambda_mean"), "lambda_global": routing["global"].get("effective_lambda_mean")})
    plots = root / "plots"; plots.mkdir(exist_ok=True)
    sparse = [row for row in rows if row["method"] != "dense"]
    for metric, label, filename in (("accuracy_micro", "Accuracy (micro)", "accuracy_vs_sparsity.png"), ("token_agreement", "Token-wise agreement", "agreement_vs_sparsity.png"),):
        fig, ax = plt.subplots(figsize=(6,4));
        for method, color in (("sol", "#d95319"), ("blasst", "#2878b5")):
            group = [r for r in sparse if r["method"] == method]; ax.plot([r["routing"]["overall"]["full_tile_sparsity"] for r in group], [r[metric] for r in group], marker="o", label=method, color=color)
        ax.set(xlabel="Measured full-tile sparsity", ylabel=label); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / filename, dpi=180); plt.close(fig)
    lines = ["# DiffusionGemma Sol-Attn vs BLASST: compact multi-benchmark", "", "| Condition | Target | β | Accuracy | Whole | Local | Global | Mass | Agreement | λ local | λ global |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        r=row["routing"]; lines.append(f"| {row['condition']} | {row['target_sparsity']:.0%} | {('—' if row['beta'] is None else f'{row['beta']:+.6f}')} | {row['accuracy_micro']:.1%} | {r['overall']['full_tile_sparsity']:.1%} | {r['local']['full_tile_sparsity']:.1%} | {r['global']['full_tile_sparsity']:.1%} | {r['overall']['retained_dense_attention_mass']:.1%} | {row['token_agreement']:.1%} | {('—' if r['local'].get('effective_lambda_mean') is None else f'{r['local']['effective_lambda_mean']:.5g}')} | {('—' if r['global'].get('effective_lambda_mean') is None else f'{r['global']['effective_lambda_mean']:.5g}')} |")
    lines += ["", "Per-benchmark accuracy is in `comparison.csv`. LongBench means LongBench-v2 short-context MCQ. BLASST λ is computed per attention call from the prior RULER calibration relation λL=αexp(γs); calibration-marked unattainable targets use λ=1. These reference masks do not claim speedup.", ""]
    (root / "report.md").write_text("\n".join(lines))
