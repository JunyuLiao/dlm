"""Build compact audit tables and plots from completed RULER routing runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.io import read_jsonl, write_json
from dllm.evaluation.ruler.official import score_predictions


def _paired_metrics(dense_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], ruler_root: str | Path) -> dict[str, Any]:
    from collections import Counter
    import hashlib
    import numpy as np

    dense_scores = {}
    candidate_scores = {}
    dense_tokens = {}
    candidate_tokens = {}
    for row in dense_rows:
        _, value = score_predictions([row], ruler_root)
        dense_scores[str(row["sample_id"])] = float(value)
        dense_tokens[str(row["sample_id"])] = row.get("completion_tokens")
    for row in candidate_rows:
        _, value = score_predictions([row], ruler_root)
        candidate_scores[str(row["sample_id"])] = float(value)
        candidate_tokens[str(row["sample_id"])] = row.get("completion_tokens")
    ids = sorted(set(dense_scores) & set(candidate_scores))
    deltas = [candidate_scores[item] - dense_scores[item] for item in ids]
    if deltas:
        # Deterministic problem-level bootstrap; the fixed seed makes report
        # regeneration byte-stable while preserving paired resampling.
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"ruler16k-paired-bootstrap").digest()[:8], "little"))
        samples = rng.integers(0, len(deltas), size=(20_000, len(deltas)))
        means = np.asarray(deltas, dtype=np.float64)[samples].mean(axis=1)
        bootstrap_ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    else:
        bootstrap_ci = [0.0, 0.0]
    dense_correct = {item for item in ids if dense_scores[item] >= 1.0}
    candidate_correct = {item for item in ids if candidate_scores[item] >= 1.0}
    discordant = Counter()
    for item in ids:
        if item in dense_correct and item in candidate_correct:
            discordant["dense_correct_candidate_correct"] += 1
        elif item in dense_correct:
            discordant["dense_correct_candidate_wrong"] += 1
        elif item in candidate_correct:
            discordant["dense_wrong_candidate_correct"] += 1
        else:
            discordant["dense_wrong_candidate_wrong"] += 1
    result = {
        "paired_examples": len(ids),
        "dense_accuracy": sum(dense_scores[item] for item in ids) / len(ids) if ids else 0.0,
        "candidate_accuracy": sum(candidate_scores[item] for item in ids) / len(ids) if ids else 0.0,
        "paired_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "paired_delta_bootstrap_95ci": bootstrap_ci,
        "dense_correct_examples": len(dense_correct),
        "candidate_accuracy_on_dense_correct": sum(candidate_scores[item] for item in dense_correct) / len(dense_correct) if dense_correct else 0.0,
        "dense_correct_broken": len(dense_correct - candidate_correct),
        "dense_incorrect_changed_prediction": sum(
            1
            for item in ids
            if item not in dense_correct
            and dense_tokens[item] != candidate_tokens[item]
        ),
        "mcnemar": dict(discordant),
    }

    dense_by_task = {}
    candidate_by_task = {}
    for row in dense_rows:
        dense_by_task.setdefault(str(row.get("task_base", row.get("task", "unknown"))), []).append(dense_scores.get(str(row["sample_id"]), 0.0))
    for row in candidate_rows:
        candidate_by_task.setdefault(str(row.get("task_base", row.get("task", "unknown"))), []).append(candidate_scores.get(str(row["sample_id"]), 0.0))
    result["per_task"] = {
        task: {
            "dense_accuracy": float(np.mean(dense_by_task.get(task, [0.0]))),
            "candidate_accuracy": float(np.mean(candidate_by_task.get(task, [0.0]))),
            "paired_delta": float(np.mean(candidate_by_task.get(task, [0.0])) - np.mean(dense_by_task.get(task, [0.0]))),
            "examples": len(candidate_by_task.get(task, [])),
        }
        for task in sorted(set(dense_by_task) | set(candidate_by_task))
    }
    return result


def _model_roots(output: Path) -> list[Path]:
    """Accept either a model directory or a combined results directory."""
    def complete_dense(path: Path) -> bool:
        predictions = path / "dense" / "predictions.jsonl"
        if not predictions.exists() or not (path / "dense" / "summary.json").exists():
            return False
        with predictions.open(encoding="utf-8") as handle:
            return any(True for _ in handle)
    if complete_dense(output):
        return [output]
    return [item for item in sorted(output.iterdir()) if item.is_dir() and complete_dense(item)]


def _plot(summary: dict[str, Any], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    plots = []
    rows = []
    for model_dir in _model_roots(output):
        dense_path = model_dir / "dense" / "predictions.jsonl"
        if not dense_path.exists():
            continue
        dense_rows = read_jsonl(dense_path)
        for condition_dir in sorted(model_dir.iterdir()):
            path = condition_dir / "summary.json"
            pred = condition_dir / "predictions.jsonl"
            if not path.exists() or not pred.exists() or condition_dir.name == "dense":
                continue
            item = json.loads(path.read_text())
            agg = item.get("routing_aggregate") or {}
            rows.append((model_dir.name, condition_dir.name, item.get("official_ruler_accuracy", 0.0), agg.get("logical_density"), agg.get("retained_dense_attention_mass"), agg.get("attention_output_relative_error")))
    if not rows:
        return plots
    for title, index, ylabel, filename in (
        ("RULER accuracy versus realized logical density", 2, "official accuracy", "accuracy_vs_density.png"),
        ("Retained dense attention mass versus realized density", 4, "retained mass", "retained_mass_vs_density.png"),
        ("Attention output error versus realized density", 5, "relative output error", "output_error_vs_density.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in sorted({row[0] for row in rows}):
            # Some logical runs do not expose retained-mass/output-error
            # metrics (and physical timing fields may be absent on dense
            # reference rows).  Do not pass ``None`` coordinates to
            # matplotlib; simply omit those points from the corresponding
            # diagnostic plot.
            selected = [
                row for row in rows
                if row[0] == model and row[3] is not None and row[index] is not None
            ]
            if not selected:
                continue
            selected.sort(key=lambda row: row[3])
            ax.plot([row[3] for row in selected], [row[index] for row in selected], marker="o", label=model)
            for row in selected:
                ax.annotate(row[1], (row[3], row[index]), fontsize=6, rotation=35)
        ax.set_title(title); ax.set_xlabel("realized logical density"); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
        path = output / "plots" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        plots.append(str(path.relative_to(output)))
    return plots


def build_ruler_report(output_dir: str | Path, ruler_root: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    models = []
    for model_dir in _model_roots(output):
        dense_rows = read_jsonl(model_dir / "dense" / "predictions.jsonl")
        dense_summary = json.loads((model_dir / "dense" / "summary.json").read_text()) if (model_dir / "dense" / "summary.json").exists() else {}
        dense_latency = (
            float(dense_summary.get("total_elapsed_seconds", 0.0))
            / int(dense_summary.get("actual_num_samples", 0))
            if int(dense_summary.get("actual_num_samples", 0)) else None
        )
        conditions = []
        for condition_dir in sorted(model_dir.iterdir()):
            if condition_dir.name == "dense" or not (condition_dir / "summary.json").exists() or not (condition_dir / "predictions.jsonl").exists():
                continue
            run = json.loads((condition_dir / "summary.json").read_text())
            predictions = read_jsonl(condition_dir / "predictions.jsonl")
            run["paired_metrics"] = _paired_metrics(dense_rows, predictions, ruler_root)
            dense_identity = {
                str(row["sample_id"]): (row.get("prompt_sha256"), row.get("inference_seed"), row.get("generator_seed"))
                for row in dense_rows
            }
            candidate_identity = {
                str(row["sample_id"]): (row.get("prompt_sha256"), row.get("inference_seed"), row.get("generator_seed"))
                for row in predictions
            }
            run["paired_input_audit"] = {
                "complete": len(predictions) == len(dense_rows),
                "same_sample_ids_prompts_and_seeds": candidate_identity == dense_identity,
            }
            candidate_count = int(run.get("actual_num_samples", 0))
            candidate_latency = (
                float(run.get("total_elapsed_seconds", 0.0)) / candidate_count
                if candidate_count else None
            )
            physical_execution = run.get("routing", {}).get("execution") == "physical"
            run["performance"] = {
                "dense_seconds_per_sample": dense_latency,
                "condition_seconds_per_sample": candidate_latency,
                "end_to_end_speedup_vs_dense": (
                    dense_latency / candidate_latency
                    if physical_execution and dense_latency is not None and candidate_latency not in (None, 0.0)
                    else None
                ),
                "speedup_reportable": physical_execution,
                "protocol": run.get("performance_protocol"),
            }
            conditions.append(run)
        models.append({"model": model_dir.name, "dense": dense_summary, "conditions": conditions})
    profile_path = output / "profiled_threshold_model.json"
    if not profile_path.exists():
        profile_path = output.parent / "profiled_threshold_model.json"
    result = {
        "schema_version": 1,
        "models": models,
        "profiled_threshold_model": json.loads(profile_path.read_text()) if profile_path.exists() else None,
        "audit": {
            "all_completed_conditions_share_prompts_and_seeds": all(
                condition.get("paired_input_audit", {}).get("same_sample_ids_prompts_and_seeds", False)
                for model in models for condition in model["conditions"]
            ),
            "theoretical_density_described_as_speedup": False,
        },
    }
    result["plots"] = _plot(result, output)
    write_json(output / "ruler_report.json", result)
    lines = ["# Fresh RULER 16K routing report", "", "This report uses the fixed tokenizer-specific 50-example manifests. Logical and physical densities are measurements of the reference mask, not speedups. Unless a row explicitly says `physical`, routing uses the exact logical dense-masked reference. Physical timing rows additionally expose prefill and denoising attention phases; these are attention-path timings, not full-generation phase timings.", ""]
    for model in models:
        lines += [f"## {model['model']}", "", "| condition | accuracy | paired delta (95% CI) | logical density | physical density | retained mass | output error | sec/sample | speedup | proxy / sparse sec | prefill proxy / sparse sec | denoising proxy / sparse sec | dense-correct subset | broken / recovered |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for run in model["conditions"]:
            condition = run.get("condition", {}).get("name", "unknown")
            agg = run.get("routing_aggregate") or {}
            pair = run.get("paired_metrics", {})
            ci = pair.get("paired_delta_bootstrap_95ci", [float("nan"), float("nan")])
            mc = pair.get("mcnemar", {})
            perf = run.get("performance", {})
            number = lambda value: float("nan") if value is None else float(value)
            lines.append(f"| {condition} | {run.get('official_ruler_accuracy', 0):.3f} | {pair.get('paired_delta', 0):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {number(agg.get('logical_density')):.3f} | {number(agg.get('physical_density')):.3f} | {number(agg.get('retained_dense_attention_mass')):.3f} | {number(agg.get('attention_output_relative_error')):.3f} | {number(perf.get('condition_seconds_per_sample')):.3f} | {number(perf.get('end_to_end_speedup_vs_dense')):.3f} | {number(agg.get('proxy_mask_seconds')):.3f} / {number(agg.get('sparse_attention_seconds')):.3f} | {number(agg.get('prefill_proxy_mask_seconds')):.3f} / {number(agg.get('prefill_sparse_attention_seconds')):.3f} | {number(agg.get('denoising_proxy_mask_seconds')):.3f} / {number(agg.get('denoising_sparse_attention_seconds')):.3f} | {pair.get('candidate_accuracy_on_dense_correct', 0):.3f} | {mc.get('dense_correct_candidate_wrong', 0)} / {mc.get('dense_wrong_candidate_correct', 0)} |")
        lines.append("")
    (output / "ruler_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
