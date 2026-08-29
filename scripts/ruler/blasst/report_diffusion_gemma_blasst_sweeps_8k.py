#!/usr/bin/env python3
"""Validate and report the 8K DiffusionGemma Q-tile and lambda sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "masked_only_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)
LAYER_TYPES = tuple(
    "global" if (layer + 1) % 6 == 0 else "local" for layer in range(30)
)
Q_CASES = ((32, 0.003), (64, 0.003), (128, 0.003), (256, 0.003))
LAMBDA_CASES = (
    (128, 0.001),
    (128, 0.003),
    (128, 0.01),
    (128, 0.03),
    (128, 0.1),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lambda_name(value: float) -> str:
    return str(value).replace(".", "p")


def case_id(q_tile: int, lambda_value: float) -> str:
    return f"q{q_tile}_lambda{lambda_name(lambda_value)}"


def case_path(root: Path, q_tile: int, lambda_value: float) -> Path:
    if math.isclose(lambda_value, 0.003, rel_tol=0.0, abs_tol=1e-15):
        return root / "runs" / "q_sweep" / f"q_{q_tile}"
    return root / "runs" / "lambda_sweep" / f"lambda_{lambda_name(lambda_value)}"


def with_ratios(counts: dict[str, int]) -> dict:
    eligible = counts["eligible_tiles"]
    votes = counts["valid_row_votes"]
    elements = counts["valid_elements"]
    return {
        **counts,
        "physical_sparsity": counts["skipped_tiles"] / eligible if eligible else 0.0,
        "row_vote_sparsity": counts["skippable_row_votes"] / votes if votes else 0.0,
        "valid_element_sparsity": (
            counts["skipped_valid_elements"] / elements if elements else 0.0
        ),
    }


def add_counts(target: dict[str, int], source: dict) -> None:
    for field in COUNT_FIELDS:
        target[field] += int(source[field])


def aggregate_layers(path: Path, q_tile: int, kv_tile: int) -> list[dict]:
    totals = {layer: defaultdict(int) for layer in range(30)}
    calls = defaultdict(int)
    kv_lengths = {layer: [] for layer in range(30)}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            q_len = int(row["query_length"])
            kv_len = int(row["sequence_length"])
            expected = 16 * math.ceil(q_len / q_tile) * math.ceil(kv_len / kv_tile)
            if int(row["eligible_tiles"]) != expected:
                raise RuntimeError(
                    f"layer {layer}: eligible {row['eligible_tiles']} != physical {expected}"
                )
            if int(row["structurally_masked_tiles"]) != 0:
                raise RuntimeError(f"layer {layer}: physical cache tile was excluded")
            add_counts(totals[layer], row)
            calls[layer] += 1
            kv_lengths[layer].append(kv_len)
    result = []
    for layer in range(30):
        if not calls[layer]:
            raise RuntimeError(f"layer {layer} has no attention calls")
        counts = with_ratios(dict(totals[layer]))
        if counts["eligible_tiles"] != counts["skipped_tiles"] + counts["retained_tiles"]:
            raise RuntimeError(f"layer {layer}: physical count identity failed")
        result.append(
            {
                "layer": layer,
                "attention_type": LAYER_TYPES[layer],
                "attention_calls": calls[layer],
                "min_kv_length": min(kv_lengths[layer]),
                "max_kv_length": max(kv_lengths[layer]),
                **counts,
            }
        )
    return result


def aggregate_group(layers: list[dict], attention_type: str | None) -> dict:
    totals = defaultdict(int)
    selected = [
        row for row in layers
        if attention_type is None or row["attention_type"] == attention_type
    ]
    for row in selected:
        add_counts(totals, row)
    return with_ratios(dict(totals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    manifest = read_json(root / "manifest" / "manifest.json")
    manifest_rows = read_jsonl(root / "manifest" / "samples.jsonl")
    if manifest["actual_num_samples"] != 100 or set(manifest["task_counts"].values()) != {20}:
        raise RuntimeError("manifest must contain exactly 20 samples per task")
    expected_samples = [
        (row["sample_id"], row["prompt_sha256"], int(row["inference_seed"]))
        for row in manifest_rows
    ]

    dense_dir = root / "runs" / "dense"
    dense = read_json(dense_dir / "summary.json")
    dense_predictions = read_jsonl(dense_dir / "predictions.jsonl")
    if dense["actual_num_samples"] != 100 or len(dense_predictions) != 100:
        raise RuntimeError("dense baseline is incomplete")
    dense_samples = [
        (row["sample_id"], row["prompt_sha256"], int(row["inference_seed"]))
        for row in dense_predictions
    ]
    if dense_samples != expected_samples:
        raise RuntimeError("dense baseline does not exactly match the manifest")

    unique_cases = sorted(set(Q_CASES + LAMBDA_CASES))
    case_summaries: dict[tuple[int, float], dict] = {}
    layer_rows: list[dict] = []
    validation = {
        "passed": True,
        "manifest_samples": 100,
        "samples_per_task": 20,
        "dense_predictions": 100,
        "dense_manifest_identity": True,
        "case_checks": {},
    }
    for q_tile, lambda_value in unique_cases:
        run = case_path(root, q_tile, lambda_value)
        summary = read_json(run / "summary.json")
        config = read_json(run / "run_config.json")
        predictions = read_jsonl(run / "predictions.jsonl")
        if summary["actual_num_samples"] != 100 or len(predictions) != 100:
            raise RuntimeError(f"{run}: incomplete sample count")
        observed_samples = [
            (row["sample_id"], row["prompt_sha256"], int(row["inference_seed"]))
            for row in predictions
        ]
        if observed_samples != expected_samples:
            raise RuntimeError(f"{run}: predictions do not exactly match the manifest")
        if not (
            config["block_size"] == 256
            and config["q_tile_size"] == q_tile
            and config["kv_tile_size"] == 64
            and math.isclose(config["blasst_lambda"], lambda_value)
            and config["include_masked_kv_tiles_in_physical_stats"]
            and config["stats_level"] == "layer"
            and summary["full_checkpoint_on_cuda"]
        ):
            raise RuntimeError(f"{run}: configuration mismatch")

        layers = aggregate_layers(run / "attention_stats" / "per_layer.csv", q_tile, 64)
        overall = aggregate_group(layers, None)
        local = aggregate_group(layers, "local")
        global_ = aggregate_group(layers, "global")
        exported = summary["attention_sparsity"]
        for field in COUNT_FIELDS:
            if int(exported[field]) != overall[field]:
                raise RuntimeError(f"{run}: layer/summary mismatch for {field}")
        identifier = case_id(q_tile, lambda_value)
        for row in layers:
            layer_rows.append(
                {
                    "case_id": identifier,
                    "q_tile_size": q_tile,
                    "kv_tile_size": 64,
                    "blasst_lambda": lambda_value,
                    **row,
                }
            )
        case_summaries[(q_tile, lambda_value)] = {
            "case_id": identifier,
            "q_tile_size": q_tile,
            "kv_tile_size": 64,
            "blasst_lambda": lambda_value,
            "dense_accuracy": dense["official_ruler_accuracy"],
            "sparse_accuracy": summary["official_ruler_accuracy"],
            "accuracy_delta": (
                summary["official_ruler_accuracy"] - dense["official_ruler_accuracy"]
            ),
            "per_task_accuracy": summary["per_task_accuracy"],
            "overall": overall,
            "local": local,
            "global": global_,
            "elapsed_seconds": summary["total_elapsed_seconds"],
        }
        validation["case_checks"][identifier] = {
            "predictions": len(predictions),
            "layers": len(layers),
            "global_count_aggregation_matches_export": True,
            "manifest_identity": True,
            "structurally_masked_tiles": overall["structurally_masked_tiles"],
        }

    summary_rows = []
    for sweep, cases in (("q_tile", Q_CASES), ("lambda", LAMBDA_CASES)):
        for q_tile, lambda_value in cases:
            case = case_summaries[(q_tile, lambda_value)]
            summary_rows.append(
                {
                    "sweep": sweep,
                    "case_id": case["case_id"],
                    "q_tile_size": q_tile,
                    "kv_tile_size": 64,
                    "blasst_lambda": lambda_value,
                    "dense_accuracy": case["dense_accuracy"],
                    "sparse_accuracy": case["sparse_accuracy"],
                    "accuracy_delta": case["accuracy_delta"],
                    "overall_physical_sparsity": case["overall"]["physical_sparsity"],
                    "local_physical_sparsity": case["local"]["physical_sparsity"],
                    "global_physical_sparsity": case["global"]["physical_sparsity"],
                    "eligible_tiles": case["overall"]["eligible_tiles"],
                    "skipped_tiles": case["overall"]["skipped_tiles"],
                    "masked_only_tiles": case["overall"]["masked_only_tiles"],
                    "elapsed_seconds": case["elapsed_seconds"],
                }
            )
    write_csv(root / "summary.csv", summary_rows)
    write_csv(root / "per_layer.csv", layer_rows)
    (root / "report.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "model": "google/diffusiongemma-26B-A4B-it",
                    "context_length": 8192,
                    "canvas_size": 256,
                    "kv_tile_size": 64,
                    "samples": 100,
                    "samples_per_task": 20,
                    "tasks": manifest["tasks"],
                    "physical_aggregation": "sum(skipped) / sum(eligible) across all layers",
                    "masked_empty_tiles": "included because eager QK and V use the physical cache tensor",
                },
                "dense": dense,
                "cases": list(case_summaries.values()),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# DiffusionGemma BLASST sweeps at 8K",
        "",
        "- Canvas: `256`; KV tile: `64`; 20 samples per each of five RULER tasks.",
        "- Physical sparsity: globally summed `skipped_tiles / eligible_tiles` across all layers and calls.",
        "- Masked-only physical cache tiles are included because the eager attention path performs QK and V work over that tensor.",
        "",
        "## Q-tile sweep (lambda 0.003)",
        "",
        "| Q tile | Dense acc. | Sparse acc. | Delta | Overall | Local | Global |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q_tile, lambda_value in Q_CASES:
        case = case_summaries[(q_tile, lambda_value)]
        lines.append(
            f"| {q_tile} | {case['dense_accuracy']:.2%} | {case['sparse_accuracy']:.2%} | "
            f"{case['accuracy_delta']:+.2%} | {case['overall']['physical_sparsity']:.2%} | "
            f"{case['local']['physical_sparsity']:.2%} | {case['global']['physical_sparsity']:.2%} |"
        )
    lines.extend([
        "",
        "## Lambda sweep (Q tile 128)",
        "",
        "| Lambda | Dense acc. | Sparse acc. | Delta | Overall | Local | Global |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for q_tile, lambda_value in LAMBDA_CASES:
        case = case_summaries[(q_tile, lambda_value)]
        lines.append(
            f"| {lambda_value:g} | {case['dense_accuracy']:.2%} | {case['sparse_accuracy']:.2%} | "
            f"{case['accuracy_delta']:+.2%} | {case['overall']['physical_sparsity']:.2%} | "
            f"{case['local']['physical_sparsity']:.2%} | {case['global']['physical_sparsity']:.2%} |"
        )
    lines.extend([
        "",
        "Full numerator/denominator counts and all 30 per-layer statistics for every unique case are in `report.json` and `per_layer.csv`.",
        "",
    ])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
