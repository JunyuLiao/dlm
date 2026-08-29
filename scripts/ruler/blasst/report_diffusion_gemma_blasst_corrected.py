"""Validate and report the corrected DiffusionGemma BLASST RULER sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


CONTEXTS = (1024, 2048, 4096, 8192, 16384)
TILE_SIZES = (32, 64, 128, 256)
LAYER_TYPES = tuple(
    "global" if (layer + 1) % 6 == 0 else "sliding" for layer in range(30)
)
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
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
ATTENTION_HEADS = 16


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_layers(path: Path, tile_size: int) -> list[dict]:
    totals = {layer: defaultdict(int) for layer in range(30)}
    sequence_lengths = {layer: [] for layer in range(30)}
    calls = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            if layer not in totals:
                raise RuntimeError(f"unexpected layer {layer}")
            for field in COUNT_FIELDS:
                totals[layer][field] += int(row[field])
            calls[layer] += 1
            q_len = int(row["query_length"])
            kv_len = int(row["sequence_length"])
            sequence_lengths[layer].append(kv_len)
            expected = (
                ATTENTION_HEADS
                * math.ceil(q_len / tile_size)
                * math.ceil(kv_len / tile_size)
            )
            if int(row["eligible_tiles"]) != expected:
                raise RuntimeError(
                    f"layer {layer}: eligible={row['eligible_tiles']}, expected={expected} "
                    f"for Q={q_len}, KV={kv_len}, tile={tile_size}"
                )
            if int(row["structurally_masked_tiles"]) != 0:
                raise RuntimeError(f"layer {layer}: physical in-scope tiles became structural")

    result = []
    for layer in range(30):
        count = totals[layer]
        eligible = count["eligible_tiles"]
        if not eligible or eligible != count["skipped_tiles"] + count["retained_tiles"]:
            raise RuntimeError(f"layer {layer}: inconsistent physical counts")
        lengths = sequence_lengths[layer]
        result.append(
            {
                "layer": layer,
                "attention_type": LAYER_TYPES[layer],
                "attention_calls": calls[layer],
                "min_kv_length": min(lengths),
                "max_kv_length": max(lengths),
                **{field: count[field] for field in COUNT_FIELDS},
                "physical_sparsity": count["skipped_tiles"] / eligible,
                "row_vote_sparsity": (
                    count["skippable_row_votes"] / count["valid_row_votes"]
                    if count["valid_row_votes"]
                    else 0.0
                ),
            }
        )
    return result


def group_ratio(rows: list[dict], attention_type: str) -> float:
    selected = [row for row in rows if row["attention_type"] == attention_type]
    return sum(row["skipped_tiles"] for row in selected) / sum(
        row["eligible_tiles"] for row in selected
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    summary_rows: list[dict] = []
    layer_rows: list[dict] = []
    dense_rows: list[dict] = []

    for context in CONTEXTS:
        manifest = read_json(root / "manifests" / str(context) / "manifest.json")
        if manifest["requested_num_samples"] != 20 or manifest["actual_num_samples"] != 20:
            raise RuntimeError(f"context {context}: manifest is not exact-count 20")
        dense_dir = root / "runs" / f"context_{context}" / "dense"
        dense = read_json(dense_dir / "summary.json")
        dense_predictions = read_jsonl(dense_dir / "predictions.jsonl")
        if dense["actual_num_samples"] != 20 or len(dense_predictions) != 20:
            raise RuntimeError(f"context {context}: dense count mismatch")
        dense_rows.append(
            {
                "context_length": context,
                "num_samples": 20,
                "dense_accuracy": dense["official_ruler_accuracy"],
                "dense_elapsed_seconds": dense["total_elapsed_seconds"],
            }
        )

        for tile_size in TILE_SIZES:
            run = root / "runs" / f"context_{context}" / f"tile_{tile_size}"
            summary = read_json(run / "summary.json")
            config = read_json(run / "run_config.json")
            predictions = read_jsonl(run / "predictions.jsonl")
            if summary["actual_num_samples"] != 20 or len(predictions) != 20:
                raise RuntimeError(f"context {context}, tile {tile_size}: count mismatch")
            if not (
                config["q_tile_size"] == config["kv_tile_size"] == tile_size
                and config["block_size"] == 256
                and config["blasst_lambda"] == 0.003
                and config["include_masked_kv_tiles_in_physical_stats"]
                and config["stats_level"] == "layer"
                and config["resolved_model_revision"] == MODEL_REVISION
                and summary["full_checkpoint_on_cuda"]
            ):
                raise RuntimeError(f"context {context}, tile {tile_size}: configuration mismatch")

            layers = aggregate_layers(run / "attention_stats" / "per_layer.csv", tile_size)
            local = group_ratio(layers, "sliding")
            global_ = group_ratio(layers, "global")
            weighted = (5.0 / 6.0) * local + (1.0 / 6.0) * global_
            layer_mean = sum(row["physical_sparsity"] for row in layers) / 30.0
            if not math.isclose(weighted, layer_mean, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("group-weighted and equal-layer averages disagree")

            for row in layers:
                layer_rows.append(
                    {"context_length": context, "tile_size": tile_size, **row}
                )
            stats = summary["attention_sparsity"]
            summary_rows.append(
                {
                    "context_length": context,
                    "tile_size": tile_size,
                    "canvas_size": 256,
                    "num_samples": 20,
                    "dense_accuracy": dense["official_ruler_accuracy"],
                    "sparse_accuracy": summary["official_ruler_accuracy"],
                    "accuracy_delta": (
                        summary["official_ruler_accuracy"] - dense["official_ruler_accuracy"]
                    ),
                    "weighted_physical_sparsity": weighted,
                    "sliding_physical_sparsity": local,
                    "global_physical_sparsity": global_,
                    "compute_weighted_physical_sparsity": stats["physical_tile_sparsity"],
                    "eligible_tiles": stats["eligible_tiles"],
                    "skipped_tiles": stats["skipped_tiles"],
                    "masked_only_tiles": stats["masked_only_tiles"],
                    "elapsed_seconds": summary["total_elapsed_seconds"],
                }
            )

    write_csv(root / "dense_baselines.csv", dense_rows)
    write_csv(root / "summary.csv", summary_rows)
    write_csv(root / "per_layer.csv", layer_rows)

    tile_averages = []
    for tile_size in TILE_SIZES:
        selected = [row for row in summary_rows if row["tile_size"] == tile_size]
        tile_averages.append(
            {
                "tile_size": tile_size,
                "contexts": len(selected),
                "samples_per_context": 20,
                "dense_accuracy": sum(row["dense_accuracy"] for row in selected) / len(selected),
                "sparse_accuracy": sum(row["sparse_accuracy"] for row in selected) / len(selected),
                "accuracy_delta": sum(row["accuracy_delta"] for row in selected) / len(selected),
                "weighted_physical_sparsity": sum(
                    row["weighted_physical_sparsity"] for row in selected
                ) / len(selected),
                "sliding_physical_sparsity": sum(
                    row["sliding_physical_sparsity"] for row in selected
                ) / len(selected),
                "global_physical_sparsity": sum(
                    row["global_physical_sparsity"] for row in selected
                ) / len(selected),
            }
        )
    write_csv(root / "tile_averages.csv", tile_averages)
    (root / "report.json").write_text(
        json.dumps(
            {
                "model": "google/diffusiongemma-26B-A4B-it",
                "canvas_size": 256,
                "tile_sizes": list(TILE_SIZES),
                "contexts": list(CONTEXTS),
                "samples_per_context_case": 20,
                "blasst_lambda": 0.003,
                "physical_sparsity_formula": "5/6 * sliding + 1/6 * global",
                "summary": summary_rows,
                "tile_averages": tile_averages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Corrected DiffusionGemma BLASST RULER sweep",
        "",
        "- Canvas: `256`; Q and KV tile sizes swept together.",
        "- BLASST lambda: `0.003`; 20 exact-count paired samples per context/case.",
        "- Reported physical sparsity: `5/6 × sliding + 1/6 × global`.",
        "- Sliding denominator: native local cache tensor plus canvas (1279 KV positions once full).",
        "- Global denominator: the complete cache tensor plus canvas.",
        "- Padding and empty-cache tiles inside either tensor are included.",
        "",
        "## Per-case results",
        "",
        "| Context | Tile | Dense acc. | Sparse acc. | Delta | Weighted physical | Sliding | Global |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['context_length']} | {row['tile_size']} | "
            f"{row['dense_accuracy']:.2%} | {row['sparse_accuracy']:.2%} | "
            f"{row['accuracy_delta']:+.2%} | {row['weighted_physical_sparsity']:.2%} | "
            f"{row['sliding_physical_sparsity']:.2%} | {row['global_physical_sparsity']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Average across contexts",
            "",
            "| Tile | Dense acc. | Sparse acc. | Delta | Weighted physical | Sliding | Global |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tile_averages:
        lines.append(
            f"| {row['tile_size']} | {row['dense_accuracy']:.2%} | "
            f"{row['sparse_accuracy']:.2%} | {row['accuracy_delta']:+.2%} | "
            f"{row['weighted_physical_sparsity']:.2%} | "
            f"{row['sliding_physical_sparsity']:.2%} | "
            f"{row['global_physical_sparsity']:.2%} |"
        )
    lines.extend(
        [
            "",
            "All 30-layer statistics for every context/tile case are in `per_layer.csv`.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
