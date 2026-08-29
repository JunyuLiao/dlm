"""Canonical audit, tables, and plots for the nine-condition bundle."""

from __future__ import annotations

import csv
import json
import ast
from pathlib import Path
from typing import Any, Mapping

from dllm.evaluation.ruler.io import read_jsonl, write_json

from .config import DEFAULT_TASKS, TARGET_SPARSITIES, condition_name
from .metrics import aggregate_routing_stats, paired_bootstrap_ci, positional_token_id_agreement, exact_sequence_match


EXPECTED_CONDITIONS = ["dense", *(condition_name("sol_gaussian", s) for s in TARGET_SPARSITIES), *(condition_name("blasst_calibrated", s) for s in TARGET_SPARSITIES)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value) if any(ch in value for ch in ".eE") else int(value)
                except ValueError:
                    if value[:1] in "[{" and value[-1:] in "]}":
                        try:
                            row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            try:
                                row[key] = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                row[key] = value
                    else:
                        row[key] = value
            rows.append(row)
    return rows


def _routing_rows(
    summary: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
    condition_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        item = prediction.get("routing_stats")
        if isinstance(item, Mapping):
            if item.get("per_call"):
                rows.extend(dict(row) for row in item["per_call"])
            else:
                rows.append(dict(item))
    # BLASST's canonical per-step export carries the local/global attention
    # labels and exact aggregate counters; prefer it when prediction rows do
    # not contain those details.
    if condition_path is not None:
        detailed = _read_csv_rows(condition_path / "attention_stats" / "per_step.csv")
        if detailed and (not rows or not any(row.get("attention_type") for row in rows)):
            return detailed
    if rows:
        return rows
    attention = summary.get("attention_sparsity")
    if isinstance(attention, Mapping):
        return [dict(attention)]
    aggregate = summary.get("routing_aggregate")
    if isinstance(aggregate, Mapping):
        return [dict(aggregate)]
    return []


def _score_rows(
    rows: list[Mapping[str, Any]], ruler_root: str | Path | None
) -> tuple[dict[str, float], float | None, float | None]:
    if not rows or ruler_root is None:
        return {}, None, None
    from dllm.evaluation.ruler.official import score_predictions
    per_task, overall = score_predictions([dict(row) for row in rows], ruler_root)
    macro = sum(float(value) for value in per_task.values()) / len(per_task) if per_task else None
    return per_task, overall, macro


def _threshold_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Merge runner routing metadata with the canonical condition record."""

    merged: dict[str, Any] = {}
    for key in ("condition", "routing", "threshold_provenance"):
        value = summary.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    # Keep the condition's explicit provenance authoritative if both layers
    # expose the same field.
    condition = summary.get("condition")
    if isinstance(condition, Mapping):
        merged.update(dict(condition))
    return merged


def _routing_coverage(
    condition: str,
    summary: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
    condition_path: Path | None,
) -> dict[str, Any]:
    """Audit local/global layer/head/step and prefix/canvas stat coverage."""

    attention_types: set[str] = set()
    layers: set[tuple[str, str]] = set()
    heads: set[tuple[str, str]] = set()
    steps: set[tuple[str, str]] = set()
    regions: set[str] = set()
    if condition.startswith("blasst_") and condition_path is not None:
        step_rows = _read_csv_rows(condition_path / "attention_stats" / "per_step.csv")
        layer_rows = _read_csv_rows(condition_path / "attention_stats" / "per_layer.csv")
        head_rows = _read_csv_rows(condition_path / "attention_stats" / "per_head.csv")
        for row in step_rows:
            attention_types.add(str(row.get("attention_type", "")))
            steps.add((str(row.get("attention_type", "")), str(row.get("denoising_step", ""))))
            parsed = row.get("region_counts")
            if isinstance(parsed, Mapping):
                regions.update(str(name) for name in parsed)
        for row in layer_rows:
            layers.add((str(row.get("attention_type", "")), str(row.get("layer", ""))))
        for row in head_rows:
            heads.add((str(row.get("attention_type", "")), str(row.get("head", ""))))
        summary_regions = summary.get("region_counts")
        if isinstance(summary_regions, Mapping):
            regions.update(str(name) for name in summary_regions)
    else:
        for prediction in predictions:
            routing = prediction.get("routing_stats")
            if not isinstance(routing, Mapping):
                continue
            attention_types.update(str(name) for name in (routing.get("by_attention_type") or {}))
            # Fresh-routing conditions retain per-call records rather than a
            # separate layer/head/step export. Prefer those records for the
            # coverage audit so local/global completeness is checked against
            # the same data used for metric aggregation.
            per_call = routing.get("per_call")
            if isinstance(per_call, list):
                for item in per_call:
                    if not isinstance(item, Mapping):
                        continue
                    attention_type = str(item.get("attention_type", ""))
                    if attention_type:
                        attention_types.add(attention_type)
                    layers.add((attention_type, str(item.get("layer", ""))))
                    head_ids = item.get("head_ids")
                    if isinstance(head_ids, (list, tuple)):
                        for head in head_ids:
                            heads.add((attention_type, str(head)))
                    else:
                        heads.add((attention_type, str(item.get("head", ""))))
                    steps.add((attention_type, str(item.get("denoising_step", ""))))
                    parsed_regions = item.get("region_counts")
                    if isinstance(parsed_regions, Mapping):
                        regions.update(str(name) for name in parsed_regions)
            for name in (routing.get("by_layer") or {}):
                attention_type = str(name).split("|", 1)[0] if "|" in str(name) else ""
                layers.add((attention_type, str(name)))
            for name in (routing.get("by_head") or {}):
                parts = str(name).split("|", 2)
                heads.add((parts[0] if parts else "", str(name)))
            for name in (routing.get("by_step") or {}):
                steps.add(("", str(name)))
            regions.update(str(name) for name in (routing.get("region_counts") or {}))
    local_global = {"local", "global"}.issubset(attention_types)
    layer_coverage = all(any(item[0] == attention_type for item in layers) for attention_type in ("local", "global"))
    head_coverage = all(any(item[0] == attention_type for item in heads) for attention_type in ("local", "global"))
    step_coverage = all(any(item[0] == attention_type for item in steps) for attention_type in ("local", "global"))
    return {
        "attention_types": sorted(attention_types),
        "local_global_present": local_global,
        "local_global_layer_present": layer_coverage,
        "local_global_head_present": head_coverage,
        "local_global_step_present": step_coverage,
        "prefix_region_present": "prefix" in regions,
        "canvas_region_present": "canvas" in regions,
        "complete": local_global and layer_coverage and head_coverage and step_coverage and {"prefix", "canvas"}.issubset(regions),
    }


def _paired(dense: list[Mapping[str, Any]], sparse: list[Mapping[str, Any]], ruler_root: str | Path | None) -> dict[str, Any]:
    dense_by_id = {str(row["sample_id"]): row for row in dense}
    sparse_by_id = {str(row["sample_id"]): row for row in sparse}
    ids = sorted(set(dense_by_id) & set(sparse_by_id))
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    if ruler_root is not None:
        from dllm.evaluation.ruler.official import score_predictions
        for sample_id in ids:
            _, dense_scores[sample_id] = score_predictions([dict(dense_by_id[sample_id])], ruler_root)
            _, sparse_scores[sample_id] = score_predictions([dict(sparse_by_id[sample_id])], ruler_root)
    deltas = [float(sparse_scores[item] - dense_scores[item]) for item in ids if item in sparse_scores and item in dense_scores]
    agreements = [positional_token_id_agreement(dense_by_id[item].get("completion_tokens", []), sparse_by_id[item].get("completion_tokens", [])) for item in ids]
    exact = [exact_sequence_match(dense_by_id[item].get("completion_tokens", []), sparse_by_id[item].get("completion_tokens", [])) for item in ids]
    return {
        "paired_examples": len(ids),
        "accuracy_delta": sum(deltas) / len(deltas) if deltas else None,
        "accuracy_delta_bootstrap_95ci": list(paired_bootstrap_ci(deltas, seed=42)) if deltas else [0.0, 0.0],
        "token_id_agreement": sum(agreements) / len(agreements) if agreements else None,
        "full_sequence_exact_match_rate": sum(exact) / len(exact) if exact else None,
        "dense_accuracy": sum(dense_scores.values()) / len(dense_scores) if dense_scores else None,
        "sparse_accuracy": sum(sparse_scores.values()) / len(sparse_scores) if sparse_scores else None,
    }


def _plots(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths: list[str] = []
    for x_field, y_field, filename, title in (
        ("full_tile_sparsity", "accuracy", "accuracy_vs_sparsity.png", "RULER accuracy versus full-tile sparsity"),
        ("full_tile_sparsity", "retained_attention_mass", "retained_mass_vs_sparsity.png", "Retained dense attention mass versus sparsity"),
        ("full_tile_sparsity", "token_agreement", "token_agreement_vs_sparsity.png", "Token agreement versus sparsity"),
    ):
        points = [row for row in rows if row.get(x_field) is not None and row.get(y_field) is not None]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot([row[x_field] for row in points], [row[y_field] for row in points], "o-")
        for row in points:
            ax.annotate(row["condition"], (row[x_field], row[y_field]), fontsize=7)
        ax.set(xlabel="measured full-tile sparsity", ylabel=y_field.replace("_", " "), title=title)
        ax.grid(alpha=0.25); fig.tight_layout()
        path = output / "plots" / filename; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)
        paths.append(str(path.relative_to(output)))
    target_points = [row for row in rows if row.get("target_sparsity") is not None]
    if target_points:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        targets = [float(row["target_sparsity"]) for row in target_points if row["condition"] != "dense"]
        for label, key, marker in (
            ("overall", "full_tile_sparsity", "o"),
            ("global", "global", "s"),
            ("local", "local", "^"),
        ):
            xs, ys = [], []
            for row in target_points:
                if row["condition"] == "dense":
                    continue
                value = row.get(key) if key in ("global", "local") else row.get(key)
                if isinstance(value, Mapping):
                    value = value.get("full_tile_sparsity")
                if value is not None:
                    xs.append(float(row["target_sparsity"])); ys.append(float(value))
            if xs:
                ax.plot(xs, ys, marker=marker, linestyle="-", label=label)
        if targets:
            lo, hi = min(targets), max(targets)
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.45, label="target")
        ax.set(xlabel="target skipped-tile sparsity", ylabel="achieved skipped-tile sparsity", title="Target versus achieved sparsity")
        ax.set_xlim(left=0.0); ax.set_ylim(bottom=0.0); ax.grid(alpha=0.25); ax.legend()
        fig.tight_layout()
        path = output / "plots" / "target_vs_achieved_sparsity.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)
        paths.append(str(path.relative_to(output)))
    return paths


def _write_comparison_tables(output: Path, table: list[dict[str, Any]]) -> dict[str, str]:
    """Write flat overall and attention-type comparison tables."""

    comparison_rows: list[dict[str, Any]] = []
    for target in TARGET_SPARSITIES:
        sol = next((row for row in table if row["condition"] == condition_name("sol_gaussian", target)), None)
        blasst = next((row for row in table if row["condition"] == condition_name("blasst_calibrated", target)), None)
        if sol is None or blasst is None:
            continue
        comparison_rows.append({
            "target_sparsity": target,
            "sol_condition": sol["condition"],
            "blasst_condition": blasst["condition"],
            "sol_actual_sparsity": sol.get("full_tile_sparsity"),
            "blasst_actual_sparsity": blasst.get("full_tile_sparsity"),
            "sol_threshold": json.dumps(sol.get("thresholds", {}), sort_keys=True),
            "blasst_threshold": json.dumps(blasst.get("thresholds", {}), sort_keys=True),
            "sol_accuracy": sol.get("accuracy"),
            "blasst_accuracy": blasst.get("accuracy"),
            "sol_retained_mass": sol.get("retained_attention_mass"),
            "blasst_retained_mass": blasst.get("retained_attention_mass"),
            "sol_token_agreement": sol.get("token_agreement"),
            "blasst_token_agreement": blasst.get("token_agreement"),
        })
    comparison_path = output / "sol_vs_blasst.csv"
    if comparison_rows:
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader(); writer.writerows(comparison_rows)

    layer_rows: list[dict[str, Any]] = []
    for row in table:
        for attention_type in ("global", "local"):
            metrics = row.get(attention_type) or {}
            layer_rows.append({
                "condition": row["condition"],
                "method": row["method"],
                "target_sparsity": row["target_sparsity"],
                "attention_type": attention_type,
                "actual_sparsity": metrics.get("full_tile_sparsity"),
                "valid_qk_element_sparsity": metrics.get("valid_qk_element_sparsity"),
                "retained_attention_mass": metrics.get("retained_dense_attention_mass"),
                "accuracy": row.get("accuracy"),
            })
    layer_path = output / "global_local.csv"
    if layer_rows:
        with layer_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
            writer.writeheader(); writer.writerows(layer_rows)
    return {
        "sol_vs_blasst": str(comparison_path.relative_to(output)) if comparison_rows else "",
        "global_local": str(layer_path.relative_to(output)) if layer_rows else "",
    }


def build_report(output_dir: str | Path, ruler_root: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    dense_predictions = read_jsonl(output / "dense" / "predictions.jsonl")
    table: list[dict[str, Any]] = []
    condition_dirs = {path.name: path for path in output.iterdir() if path.is_dir() and (path / "summary.json").exists()}
    for condition in EXPECTED_CONDITIONS:
        path = condition_dirs.get(condition)
        summary = _load_json(path / "summary.json") if path else {}
        predictions = read_jsonl(path / "predictions.jsonl") if path else []
        routing_rows = _routing_rows(summary, predictions, path)
        routing = aggregate_routing_stats(routing_rows) if routing_rows else {"overall": {}}
        # Keep the per-condition fresh-routing shard self-consistent when a
        # run was created by an older runner that only aggregated its nested
        # per-example totals.  The canonical report is generated from the
        # completed prediction/per-call shards and can safely refresh this
        # derived aggregate without replaying model inference.
        if path is not None and routing_rows and condition.startswith("sol_gaussian"):
            routing_path = path / "routing_stats.json"
            routing_payload = _load_json(routing_path)
            routing_payload["schema_version"] = int(routing_payload.get("schema_version", 1))
            routing_payload["aggregate"] = routing
            write_json(routing_path, routing_payload)
            summary_payload = dict(summary)
            summary_payload["routing_aggregate"] = routing
            write_json(path / "summary.json", summary_payload)
        overall = routing.get("overall", {})
        per_task, scored, macro = _score_rows(predictions, ruler_root)
        paired = {} if condition == "dense" else _paired(dense_predictions, predictions, ruler_root)
        accuracy = scored if scored is not None else summary.get("official_ruler_accuracy")
        summary_per_task = summary.get("per_task_accuracy", {})
        if macro is None and isinstance(summary_per_task, Mapping) and summary_per_task:
            macro = sum(float(value) for value in summary_per_task.values()) / len(summary_per_task)
        coverage = _routing_coverage(condition, summary, predictions, path)
        dense_baseline = condition == "dense"
        table.append({
            "condition": condition,
            "method": "dense" if condition == "dense" else "sol_gaussian" if condition.startswith("sol_gaussian") else "blasst_calibrated",
            "target_sparsity": 0.0 if condition == "dense" else int(condition.rsplit("s", 1)[-1]) / 100.0,
            "full_tile_sparsity": 0.0 if dense_baseline and predictions else overall.get("full_tile_sparsity"),
            "valid_qk_element_sparsity": 0.0 if dense_baseline and predictions else overall.get("valid_qk_element_sparsity"),
            "retained_attention_mass": 1.0 if dense_baseline and predictions else overall.get("retained_dense_attention_mass"),
            "accuracy": accuracy,
            "equal_task_macro_accuracy": macro if macro is not None else summary.get("equal_task_macro_accuracy"),
            "per_task_accuracy": per_task or summary.get("per_task_accuracy", {}),
            "token_agreement": 1.0 if dense_baseline and predictions else paired.get("token_id_agreement"),
            "exact_match_rate": 1.0 if dense_baseline and predictions else paired.get("full_sequence_exact_match_rate"),
            "paired_accuracy_delta": paired.get("accuracy_delta"),
            "paired_accuracy_ci95": paired.get("accuracy_delta_bootstrap_95ci"),
            "eligible_tiles": overall.get("eligible_tiles", 0),
            "skipped_tiles": overall.get("skipped_tiles", 0),
            "degenerate_row_rate": overall.get("degenerate_row_rate"),
            "empty_row_fallback_rate": overall.get("empty_row_fallback_rate"),
            "global": routing.get("global", {}),
            "local": routing.get("local", {}),
            "global_layer": routing.get("global", {}),
            "local_layer": routing.get("local", {}),
            "thresholds": _threshold_metadata(summary),
            "coverage": coverage,
            "examples": len(predictions),
        })
    rows_csv = []
    for row in table:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        rows_csv.append(flat)
    csv_path = output / "summary.csv"
    if rows_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0])); writer.writeheader(); writer.writerows(rows_csv)
    canonical_dirs = {name: condition_dirs[name] for name in EXPECTED_CONDITIONS if name in condition_dirs}
    identity_sets = {}
    for name, path in canonical_dirs.items():
        identity_sets[name] = sorted(
            (str(row.get("sample_id", "")), str(row.get("prompt_hash", row.get("prompt_sha256", ""))), int(row.get("inference_seed", row.get("seed", -1))))
            for row in read_jsonl(path / "predictions.jsonl")
        )
    dense_identity = identity_sets.get("dense", [])
    thresholds_complete = True
    coverage = {}
    for row in table:
        if row["condition"] == "dense":
            continue
        threshold = row.get("thresholds") or {}
        if row["method"] == "sol_gaussian":
            thresholds_complete &= threshold.get("beta_used") is not None or threshold.get("beta") is not None
        else:
            thresholds_complete &= threshold.get("lambda_local") is not None and threshold.get("lambda_global") is not None
        coverage[row["condition"]] = row.get("coverage", {})
    sparse_coverage = [coverage[name].get("complete", False) for name in EXPECTED_CONDITIONS if name != "dense" and name in coverage]
    audit = {
        "expected_conditions": EXPECTED_CONDITIONS,
        "present_conditions": sorted(condition_dirs),
        "all_nine_conditions_present": all(name in condition_dirs for name in EXPECTED_CONDITIONS),
        "final_examples_per_condition": {name: len(read_jsonl(path / "predictions.jsonl")) for name, path in canonical_dirs.items()},
        "all_conditions_have_50_examples": bool(canonical_dirs) and all(len(read_jsonl(path / "predictions.jsonl")) == 50 for path in canonical_dirs.values()),
        "all_conditions_share_prompts_and_seeds": bool(dense_identity) and all(items == dense_identity for items in identity_sets.values()),
        "threshold_provenance_complete": thresholds_complete,
        "routing_coverage": coverage,
        "all_sparse_conditions_have_complete_routing_coverage": bool(sparse_coverage) and all(sparse_coverage),
        "report_uses_only_completed_shards": True,
        "measured_speedup_reported": False,
    }
    write_json(output / "audit.json", audit)
    plots = _plots(output, table)
    comparison_tables = _write_comparison_tables(output, table)
    result = {"schema_version": 1, "study": "diffusion_gemma_solattn_vs_blasst_ruler16k", "conditions": table, "audit": audit, "plots": plots, "comparison_tables": comparison_tables}
    write_json(output / "summary.json", result)
    lines = ["# DiffusionGemma Sol-Attn versus BLASST on RULER16K", "", "Physical full-tile sparsity and valid-QK sparsity are reference-mask measurements; no speedup is claimed.", "", "| condition | target skip | measured skip | accuracy | task macro | retained mass | token agreement | exact match |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.3f}"
        lines.append(f"| {row['condition']} | {float(row['target_sparsity']):.0%} | {fmt(row['full_tile_sparsity'])} | {fmt(row['accuracy'])} | {fmt(row['equal_task_macro_accuracy'])} | {fmt(row['retained_attention_mass'])} | {fmt(row['token_agreement'])} | {fmt(row['exact_match_rate'])} |")
    lines += [
        "",
        "Comparison tables: `sol_vs_blasst.csv` and `global_local.csv`.",
        f"Audit: `{json.dumps(audit, sort_keys=True)}`",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


__all__ = ["EXPECTED_CONDITIONS", "build_report"]
