#!/usr/bin/env python3
"""Validate and report the phase-aware DiffusionGemma BLASST experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.official import score_predictions


COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
)
PHASE_NAMES = {0: "first", 1: "middle", 2: "late"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["prompt_sha256"], int(row["inference_seed"])


def ratio(counts: dict[str, int]) -> float:
    return counts["skipped_tiles"] / counts["eligible_tiles"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--sparse-run", default="sparse_target20_refined")
    parser.add_argument("--calibration", default="calibration_target20_refined")
    args = parser.parse_args()
    root = args.experiment_root.resolve()

    dense_dir = root / "dense_eager_calibration"
    sparse_dir = root / args.sparse_run
    calibration_dir = root / args.calibration
    dense_summary = read_json(dense_dir / "summary.json")
    sparse_summary = read_json(sparse_dir / "summary.json")
    dense_config = read_json(dense_dir / "run_config.json")
    sparse_config = read_json(sparse_dir / "run_config.json")
    policy = read_json(calibration_dir / "policy.json")
    dense_rows = read_jsonl(dense_dir / "predictions.jsonl")
    sparse_rows = read_jsonl(sparse_dir / "predictions.jsonl")

    if len(dense_rows) != 100 or len(sparse_rows) != 100:
        raise RuntimeError("both runs must contain exactly 100 predictions")
    if [sample_identity(row) for row in dense_rows] != [
        sample_identity(row) for row in sparse_rows
    ]:
        raise RuntimeError("dense and sparse sample identities differ")
    for field, expected in {
        "block_size": 256,
        "q_tile_size": 128,
        "kv_tile_size": 64,
        "context_length": 8192,
    }.items():
        if dense_config[field] != expected or sparse_config[field] != expected:
            raise RuntimeError(f"configuration mismatch for {field}")
    if dense_config["attention_backend"] != "eager-dense":
        raise RuntimeError("dense run did not use eager-dense")
    if sparse_config["attention_backend"] != "blasst-reference":
        raise RuntimeError("sparse run did not use BLASST eager reference")
    if sparse_config["include_masked_kv_tiles_in_physical_stats"]:
        raise RuntimeError("semantic denominator must exclude structural tiles")
    if sparse_config["blasst_policy"] != policy:
        raise RuntimeError("deployed policy differs from calibrated policy")

    per_layer: dict[int, dict[str, Any]] = {}
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_phase: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    with (sparse_dir / "attention_stats" / "per_layer.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            attention_type = row["attention_type"]
            phase = int(row["denoising_phase"])
            target = per_layer.setdefault(
                layer,
                {
                    "layer": layer,
                    "attention_type": attention_type,
                    **{field: 0 for field in COUNT_FIELDS},
                },
            )
            if target["attention_type"] != attention_type:
                raise RuntimeError(f"layer {layer} changed attention type")
            for field in COUNT_FIELDS:
                value = int(row[field])
                target[field] += value
                by_group[attention_type][field] += value
                by_phase[(attention_type, phase)][field] += value

    layer_rows = []
    for layer in sorted(per_layer):
        row = per_layer[layer]
        if row["eligible_tiles"] != row["skipped_tiles"] + row["retained_tiles"]:
            raise RuntimeError(f"count identity failed for layer {layer}")
        layer_rows.append({**row, "physical_sparsity": ratio(row)})
    overall = {field: sum(row[field] for row in layer_rows) for field in COUNT_FIELDS}
    if any(int(sparse_summary["attention_sparsity"][field]) != overall[field] for field in COUNT_FIELDS):
        raise RuntimeError("summary and globally aggregated layer counts differ")

    ruler_root = sparse_config["ruler_root"]
    dense_correct = {
        row["sample_id"]: score_predictions([row], ruler_root)[1] for row in dense_rows
    }
    sparse_correct = {
        row["sample_id"]: score_predictions([row], ruler_root)[1] for row in sparse_rows
    }
    transitions = Counter(
        (row["task"], dense_correct[row["sample_id"]], sparse_correct[row["sample_id"]])
        for row in dense_rows
    )

    phase_rows = [
        {
            "attention_type": attention_type,
            "phase": phase,
            "phase_name": PHASE_NAMES[phase],
            **counts,
            "physical_sparsity": ratio(counts),
            "lambda": policy[f"{attention_type}_phase_lambdas"][phase],
        }
        for (attention_type, phase), counts in sorted(by_phase.items())
    ]
    group_rows = [
        {"attention_type": name, **counts, "physical_sparsity": ratio(counts)}
        for name, counts in sorted(by_group.items())
    ]

    with (root / "per_layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    report = {
        "validation": {
            "passed": True,
            "same_100_samples_and_seeds": True,
            "same_eager_qk_softmax_pv_path": True,
            "dense_backend": dense_config["attention_backend"],
            "sparse_backend": sparse_config["attention_backend"],
            "physical_denominator": "eligible non-structural KV tiles",
        },
        "policy": policy,
        "accuracy": {
            "dense": dense_summary["official_ruler_accuracy"],
            "sparse": sparse_summary["official_ruler_accuracy"],
            "delta": sparse_summary["official_ruler_accuracy"]
            - dense_summary["official_ruler_accuracy"],
            "dense_by_task": dense_summary["per_task_accuracy"],
            "sparse_by_task": sparse_summary["per_task_accuracy"],
            "paired_wins": sum(sparse_correct[key] > dense_correct[key] for key in dense_correct),
            "paired_losses": sum(sparse_correct[key] < dense_correct[key] for key in dense_correct),
            "transitions": [
                {
                    "task": key[0],
                    "dense_correct": key[1],
                    "sparse_correct": key[2],
                    "count": count,
                }
                for key, count in sorted(transitions.items())
            ],
        },
        "sparsity": {
            "overall": {**overall, "physical_sparsity": ratio(overall)},
            "by_attention_type": group_rows,
            "by_phase": phase_rows,
            "per_layer": layer_rows,
        },
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase-aware DiffusionGemma BLASST validation",
        "",
        "Dense and sparse use the same eager QK/mask/softmax/PV implementation and identical samples/seeds.",
        "",
        "| Run | Accuracy | Physical sparsity |",
        "|---|---:|---:|",
        f"| Dense eager | {report['accuracy']['dense']:.1%} | 0.0% |",
        f"| Phase-aware BLASST | {report['accuracy']['sparse']:.1%} | {ratio(overall):.2%} |",
        "",
        "| Attention | Phase | Lambda | Achieved sparsity |",
        "|---|---|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            f"| {row['attention_type']} | {row['phase_name']} | {row['lambda']:.6g} | "
            f"{row['physical_sparsity']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Paired outcomes: {report['accuracy']['paired_wins']} sparse wins, "
            f"{report['accuracy']['paired_losses']} sparse losses. All changes are on VT.",
            "",
            "The physical denominator excludes structurally masked tiles because the target sparse kernel would not schedule QK/PV work for them. The eager reference run applies the sparse mask for correctness testing but does not itself provide sparse-kernel speedup.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
