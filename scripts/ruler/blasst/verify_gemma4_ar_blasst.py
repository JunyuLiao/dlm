#!/usr/bin/env python3
"""Audit the completed Gemma 4 causal-BLASST RULER sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.io import read_jsonl, sha256_file, write_json
from dllm.evaluation.ruler.official import load_scorers, postprocess_prediction, score_predictions


CASES: tuple[tuple[str, float | None], ...] = (
    ("dense", None),
    ("lambda_0p001", 0.001),
    ("lambda_0p003", 0.003),
    ("lambda_0p01", 0.01),
    ("lambda_0p03", 0.03),
    ("lambda_0p1", 0.1),
)
COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
)
SAMPLE_FIELDS = (
    "sample_id",
    "task",
    "task_base",
    "target_length",
    "actual_prompt_length",
    "actual_total_length",
    "generator_seed",
    "inference_seed",
    "prompt",
    "prompt_sha256",
    "outputs",
    "tokens_to_generate",
    "official_index",
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_close(actual: float, expected: float, message: str) -> None:
    if abs(actual - expected) > 1.0e-12:
        raise AssertionError(f"{message}: {actual} != {expected}")


def _individual_scores(rows: list[dict[str, Any]], ruler_root: Path) -> dict[str, float]:
    scorers = load_scorers(ruler_root)
    values = {}
    for row in rows:
        scorer = scorers[str(row["task_base"])]
        values[str(row["sample_id"])] = float(
            scorer(
                [postprocess_prediction(str(row["prediction"]))],
                [[str(answer) for answer in row["outputs"]]],
            )
        ) / 100.0
    return values


def _read_layer_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/blasst/gemma4_ar/ruler8k_n260"),
    )
    parser.add_argument("--ruler-root", type=Path, default=Path("/tmp/NVIDIA-RULER"))
    parser.add_argument(
        "--diffusion-reference",
        type=Path,
        default=Path(
            "results/blasst/diffusion_gemma/paper_8k_n650/screen/lambda_0p003"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_path = Path(manifest["samples"]["path"])
    if sha256_file(samples_path) != manifest["samples"]["sha256"]:
        raise AssertionError("manifest sample hash changed")
    samples = read_jsonl(samples_path)
    if len(samples) != 260 or len({row["sample_id"] for row in samples}) != 260:
        raise AssertionError("manifest must contain 260 unique samples")
    task_counts = Counter(str(row["task"]) for row in samples)
    if set(task_counts.values()) != {20} or len(task_counts) != 13:
        raise AssertionError("manifest must contain 20 samples for each of 13 tasks")
    expected_by_id = {str(row["sample_id"]): row for row in samples}
    expected_ids = [str(row["sample_id"]) for row in samples]

    case_rows: dict[str, list[dict[str, Any]]] = {}
    case_scores: dict[str, dict[str, float]] = {}
    case_audits = []
    invariant_config: dict[str, Any] | None = None
    invariant_fields = (
        "algorithm_version",
        "script_sha256",
        "model",
        "revision",
        "manifest",
        "manifest_sha256",
        "samples",
        "samples_per_task",
        "context_length",
        "q_tile_size",
        "kv_tile_size",
        "prefill_backend",
        "decode_backend",
        "physical_denominator",
        "thinking",
        "max_new_tokens_override",
        "torch",
        "transformers",
        "cuda",
        "device",
    )

    for case_name, lambda_value in CASES:
        case_dir = root / "runs" / case_name
        config = json.loads((case_dir / "run_config.json").read_text(encoding="utf-8"))
        stored_fingerprint = config.pop("fingerprint")
        if _json_hash(config) != stored_fingerprint:
            raise AssertionError(f"{case_name}: run-config fingerprint mismatch")
        config["fingerprint"] = stored_fingerprint
        if config["lambda"] != lambda_value:
            raise AssertionError(f"{case_name}: lambda mismatch")
        if bool(config["apply_sparse_mask"]) != (lambda_value is not None):
            raise AssertionError(f"{case_name}: sparse-mask setting mismatch")
        selected_invariants = {field: config[field] for field in invariant_fields}
        if invariant_config is None:
            invariant_config = selected_invariants
        elif selected_invariants != invariant_config:
            raise AssertionError(f"{case_name}: experimental invariants changed")

        rows = read_jsonl(case_dir / "predictions.jsonl")
        if [str(row["sample_id"]) for row in rows] != expected_ids:
            raise AssertionError(f"{case_name}: predictions are missing, duplicated, or reordered")
        for row in rows:
            expected = expected_by_id[str(row["sample_id"])]
            for field in SAMPLE_FIELDS:
                if row[field] != expected[field]:
                    raise AssertionError(f"{case_name}/{row['sample_id']}: {field} changed")
            if not row["completion_tokens"] or not str(row["prediction"]).strip():
                raise AssertionError(f"{case_name}/{row['sample_id']}: empty completion")

        summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        recomputed_per_task, recomputed_accuracy = score_predictions(rows, args.ruler_root)
        _assert_close(
            float(summary["official_ruler_accuracy"]),
            recomputed_accuracy,
            f"{case_name}: official accuracy",
        )
        if summary["per_task_accuracy"] != recomputed_per_task:
            raise AssertionError(f"{case_name}: per-task score mismatch")

        layer_count = 0
        if lambda_value is not None:
            layer_rows = _read_layer_rows(case_dir / "per_layer.csv")
            if [int(row["layer"]) for row in layer_rows] != list(range(30)):
                raise AssertionError(f"{case_name}: expected layers 0 through 29")
            global_layers = {5, 11, 17, 23, 29}
            aggregate = Counter()
            by_type: dict[str, Counter[str]] = {"global": Counter(), "local": Counter()}
            for row in layer_rows:
                layer = int(row["layer"])
                expected_type = "global" if layer in global_layers else "local"
                if row["attention_type"] != expected_type:
                    raise AssertionError(f"{case_name}: layer {layer} type mismatch")
                values = {field: int(row[field]) for field in COUNT_FIELDS}
                if values["eligible_tiles"] != values["skipped_tiles"] + values["retained_tiles"]:
                    raise AssertionError(f"{case_name}: layer {layer} count identity failed")
                if expected_type == "local":
                    bound = int(row["decode_attention_calls"]) * int(row["query_heads"]) * 17
                    if values["eligible_tiles"] > bound:
                        raise AssertionError(f"{case_name}: layer {layer} exceeds local window")
                for field, value in values.items():
                    aggregate[field] += value
                    by_type[expected_type][field] += value
                _assert_close(
                    float(row["physical_sparsity"]),
                    values["skipped_tiles"] / values["eligible_tiles"],
                    f"{case_name}: layer {layer} sparsity",
                )
            if dict(aggregate) != {
                field: int(summary["attention_sparsity"][field]) for field in COUNT_FIELDS
            }:
                raise AssertionError(f"{case_name}: overall layer aggregation mismatch")
            summary_types = {
                row["attention_type"]: row for row in summary["sparsity_by_attention_type"]
            }
            for attention_type, values in by_type.items():
                if dict(values) != {
                    field: int(summary_types[attention_type][field]) for field in COUNT_FIELDS
                }:
                    raise AssertionError(f"{case_name}: {attention_type} aggregation mismatch")
            layer_count = len(layer_rows)

        case_rows[case_name] = rows
        case_scores[case_name] = _individual_scores(rows, args.ruler_root)
        case_audits.append(
            {
                "case": case_name,
                "lambda": lambda_value,
                "samples": len(rows),
                "tasks": len(task_counts),
                "samples_per_task": sorted(set(task_counts.values()))[0],
                "all_manifest_fields_match": True,
                "all_completions_nonempty": True,
                "official_accuracy_recomputed": recomputed_accuracy,
                "layer_count": layer_count,
                "count_identities_pass": True,
                "local_window_bound_pass": True,
            }
        )

    dense_rows = {str(row["sample_id"]): row for row in case_rows["dense"]}
    dense_scores = case_scores["dense"]
    paired = []
    for case_name, lambda_value in CASES[1:]:
        rows = case_rows[case_name]
        scores = case_scores[case_name]
        wins = sum(scores[sample_id] > dense_scores[sample_id] for sample_id in expected_ids)
        losses = sum(scores[sample_id] < dense_scores[sample_id] for sample_id in expected_ids)
        paired.append(
            {
                "case": case_name,
                "lambda": lambda_value,
                "score_wins": wins,
                "score_losses": losses,
                "score_ties": len(expected_ids) - wins - losses,
                "exact_prediction_matches": sum(
                    str(row["prediction"]) == str(dense_rows[str(row["sample_id"])]["prediction"])
                    for row in rows
                ),
                "exact_completion_token_matches": sum(
                    row["completion_tokens"]
                    == dense_rows[str(row["sample_id"])]["completion_tokens"]
                    for row in rows
                ),
            }
        )

    diffusion_rows = read_jsonl(args.diffusion_reference / "predictions.jsonl")
    ar_first_ten = {
        str(row["sample_id"]): row
        for row in samples
        if int(str(row["sample_id"]).rsplit("_", 1)[1]) < 10
    }
    if len(ar_first_ten) != 130 or len(diffusion_rows) != 130:
        raise AssertionError("matched DiffusionGemma subset must contain 130 examples")
    for row in diffusion_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in ar_first_ten:
            raise AssertionError(f"DiffusionGemma sample {sample_id} is absent from AR subset")
        if row["prompt_sha256"] != ar_first_ten[sample_id]["prompt_sha256"]:
            raise AssertionError(f"DiffusionGemma prompt {sample_id} differs from AR")

    diffusion_ids = {str(row["sample_id"]) for row in diffusion_rows}
    matched_ar_rows = [
        row for row in case_rows["lambda_0p003"] if str(row["sample_id"]) in diffusion_ids
    ]
    matched_counts: dict[str, Counter[str]] = {
        "overall": Counter(),
        "global": Counter(),
        "local": Counter(),
    }
    for row in matched_ar_rows:
        for layer in row["attention_stats"]["layers"]:
            attention_type = str(layer["attention_type"])
            for field in COUNT_FIELDS:
                matched_counts["overall"][field] += int(layer[field])
                matched_counts[attention_type][field] += int(layer[field])
    matched_ar_sparsity = {
        attention_type: values["skipped_tiles"] / values["eligible_tiles"]
        for attention_type, values in matched_counts.items()
    }
    diffusion_summary = json.loads(
        (args.diffusion_reference / "summary.json").read_text(encoding="utf-8")
    )
    diffusion_layer_rows = _read_layer_rows(
        args.diffusion_reference / "attention_stats/per_layer.csv"
    )
    diffusion_counts: dict[str, Counter[str]] = {"global": Counter(), "local": Counter()}
    for row in diffusion_layer_rows:
        attention_type = str(row["attention_type"])
        diffusion_counts[attention_type]["eligible_tiles"] += int(row["eligible_tiles"])
        diffusion_counts[attention_type]["skipped_tiles"] += int(row["skipped_tiles"])
    matched_diffusion_sparsity = {
        "overall": float(diffusion_summary["attention_sparsity"]["physical_tile_sparsity"]),
        **{
            attention_type: values["skipped_tiles"] / values["eligible_tiles"]
            for attention_type, values in diffusion_counts.items()
        },
    }

    output = {
        "schema_version": 1,
        "passed": True,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "samples": len(samples),
            "tasks": len(task_counts),
            "samples_per_task": 20,
            "task_counts": dict(sorted(task_counts.items())),
        },
        "experimental_invariants": invariant_config,
        "cases": case_audits,
        "paired_vs_dense": paired,
        "diffusion_reference_pairing": {
            "samples": len(diffusion_rows),
            "same_sample_ids": True,
            "same_prompt_hashes": True,
            "selection": "first 10 samples per task of the AR 20-sample set",
        },
        "matched_lambda_0p003_comparison": {
            "samples_each": len(diffusion_rows),
            "ar_sparsity": matched_ar_sparsity,
            "diffusion_sparsity": matched_diffusion_sparsity,
            "ar_minus_diffusion": {
                attention_type: matched_ar_sparsity[attention_type]
                - matched_diffusion_sparsity[attention_type]
                for attention_type in ("overall", "global", "local")
            },
        },
    }
    write_json(root / "validation.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
