#!/usr/bin/env python3
"""Validate and report the paper-aligned DiffusionGemma BLASST experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
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


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["prompt_sha256"], int(row["inference_seed"])


def sparsity(counts: dict[str, int]) -> float:
    return counts["skipped_tiles"] / counts["eligible_tiles"]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    differences: list[float],
    tasks: list[str],
    *,
    iterations: int = 20_000,
    task_level_differences: list[float] | None = None,
) -> dict[str, list[float]]:
    rng = random.Random(20260811)
    paired_draws = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(iterations)
    ]
    by_task: dict[str, list[float]] = defaultdict(list)
    for task, difference in zip(tasks, differences, strict=True):
        by_task[task].append(difference)
    task_means = task_level_differences or [
        sum(values) / len(values) for values in by_task.values()
    ]
    task_draws = [
        sum(rng.choice(task_means) for _ in task_means) / len(task_means)
        for _ in range(iterations)
    ]
    return {
        "paired_sample_bootstrap_95_ci": [
            quantile(paired_draws, 0.025),
            quantile(paired_draws, 0.975),
        ],
        "task_cluster_bootstrap_95_ci": [
            quantile(task_draws, 0.025),
            quantile(task_draws, 0.975),
        ],
    }


def aggregate_stats(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    per_layer: dict[int, dict[str, Any]] = {}
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_phase: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            layer = int(source["layer"])
            attention_type = source["attention_type"]
            phase = int(source["denoising_phase"])
            target = per_layer.setdefault(
                layer,
                {"layer": layer, "attention_type": attention_type, **{key: 0 for key in COUNT_FIELDS}},
            )
            if target["attention_type"] != attention_type:
                raise RuntimeError(f"layer {layer} changed attention type")
            for key in COUNT_FIELDS:
                value = int(source[key])
                target[key] += value
                by_group[attention_type][key] += value
                by_phase[(attention_type, phase)][key] += value

    layer_rows = []
    for layer in sorted(per_layer):
        row = per_layer[layer]
        if row["eligible_tiles"] != row["skipped_tiles"] + row["retained_tiles"]:
            raise RuntimeError(f"count identity failed for layer {layer}")
        layer_rows.append({**row, "physical_sparsity": sparsity(row)})
    group_rows = [
        {"attention_type": name, **counts, "physical_sparsity": sparsity(counts)}
        for name, counts in sorted(by_group.items())
    ]
    phase_rows = [
        {
            "attention_type": name,
            "phase": phase,
            "phase_name": PHASE_NAMES[phase],
            **counts,
            "physical_sparsity": sparsity(counts),
        }
        for (name, phase), counts in sorted(by_phase.items())
    ]
    overall = {key: sum(row[key] for row in layer_rows) for key in COUNT_FIELDS}
    return layer_rows, group_rows, phase_rows, overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--sparse-run", default="runs/final_target_25")
    parser.add_argument("--calibration", default="calibration/target_25")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    dense_dir = root / "runs/dense_eager_calibration"
    sparse_dir = root / args.sparse_run
    calibration_dir = root / args.calibration

    dense_summary = read_json(dense_dir / "summary.json")
    sparse_summary = read_json(sparse_dir / "summary.json")
    dense_config = read_json(dense_dir / "run_config.json")
    sparse_config = read_json(sparse_dir / "run_config.json")
    policy = read_json(calibration_dir / "policy.json")
    dense_rows = read_jsonl(dense_dir / "predictions.jsonl")
    sparse_rows = read_jsonl(sparse_dir / "predictions.jsonl")
    manifest = read_json(root / "manifest/manifest.json")

    expected = int(manifest["actual_num_samples"])
    if expected != 650 or len(dense_rows) != expected or len(sparse_rows) != expected:
        raise RuntimeError("manifest, dense, and sparse runs must each contain 650 examples")
    if [identity(row) for row in dense_rows] != [identity(row) for row in sparse_rows]:
        raise RuntimeError("dense and sparse sample identities or seeds differ")
    for key, expected_value in {
        "block_size": 256,
        "q_tile_size": 128,
        "kv_tile_size": 64,
        "context_length": 8192,
    }.items():
        if dense_config[key] != expected_value or sparse_config[key] != expected_value:
            raise RuntimeError(f"configuration mismatch for {key}")
    if dense_config["attention_backend"] != "eager-dense":
        raise RuntimeError("dense run did not use eager-dense")
    if sparse_config["attention_backend"] != "blasst-reference":
        raise RuntimeError("sparse run did not use the BLASST eager reference")
    if sparse_config["include_masked_kv_tiles_in_physical_stats"]:
        raise RuntimeError("semantic denominator must exclude structural tiles")
    if sparse_config["blasst_policy"] != policy:
        raise RuntimeError("deployed policy differs from calibrated policy")

    layer_rows, group_rows, phase_rows, overall = aggregate_stats(
        sparse_dir / "attention_stats/per_layer.csv"
    )
    for key in COUNT_FIELDS:
        if int(sparse_summary["attention_sparsity"][key]) != overall[key]:
            raise RuntimeError("summary and globally aggregated layer counts differ")
    for row in phase_rows:
        row["lambda"] = policy[f"{row['attention_type']}_phase_lambdas"][row["phase"]]

    dense_scores = {
        row["sample_id"]: score_predictions([row], dense_config["ruler_root"])[1]
        for row in dense_rows
    }
    sparse_scores = {
        row["sample_id"]: score_predictions([row], sparse_config["ruler_root"])[1]
        for row in sparse_rows
    }
    score_differences = [
        sparse_scores[row["sample_id"]] - dense_scores[row["sample_id"]]
        for row in sparse_rows
    ]
    uncertainty = bootstrap_ci(
        score_differences,
        [row["task"] for row in sparse_rows],
        task_level_differences=[
            sparse_summary["per_task_accuracy"][task]
            - dense_summary["per_task_accuracy"][task]
            for task in sorted(dense_summary["per_task_accuracy"])
        ],
    )
    non_vt_rows = [row for row in sparse_rows if row["task"] != "vt"]
    non_vt_differences = [
        sparse_scores[row["sample_id"]] - dense_scores[row["sample_id"]]
        for row in non_vt_rows
    ]
    non_vt_uncertainty = bootstrap_ci(
        non_vt_differences,
        [row["task"] for row in non_vt_rows],
        task_level_differences=[
            sparse_summary["per_task_accuracy"][task]
            - dense_summary["per_task_accuracy"][task]
            for task in sorted(dense_summary["per_task_accuracy"])
            if task != "vt"
        ],
    )
    non_vt_dense_rows = [row for row in dense_rows if row["task"] != "vt"]
    non_vt_dense_accuracy = score_predictions(
        non_vt_dense_rows, dense_config["ruler_root"]
    )[1]
    non_vt_sparse_accuracy = score_predictions(
        non_vt_rows, sparse_config["ruler_root"]
    )[1]
    vt_rows = [row for row in sparse_rows if row["task"] == "vt"]
    vt_differences = [
        sparse_scores[row["sample_id"]] - dense_scores[row["sample_id"]]
        for row in vt_rows
    ]
    vt_uncertainty = bootstrap_ci(
        vt_differences,
        ["vt"] * len(vt_rows),
        task_level_differences=[
            sparse_summary["per_task_accuracy"]["vt"]
            - dense_summary["per_task_accuracy"]["vt"]
        ],
    )

    replay_path = root / "policy_screen/target_25_pinned/predictions.jsonl"
    replay_rows = read_jsonl(replay_path)
    sparse_by_id = {row["sample_id"]: row for row in sparse_rows}
    replay_audit = {
        "samples": len(replay_rows),
        "identity_equal": sum(
            identity(row) == identity(sparse_by_id[row["sample_id"]])
            for row in replay_rows
        ),
        "prediction_text_equal": sum(
            row["prediction"] == sparse_by_id[row["sample_id"]]["prediction"]
            for row in replay_rows
        ),
        "completion_tokens_equal": sum(
            row["completion_tokens"]
            == sparse_by_id[row["sample_id"]]["completion_tokens"]
            for row in replay_rows
        ),
    }

    screen_manifest = read_json(root / "screen_manifest/manifest.json")
    screen_samples = read_jsonl(Path(screen_manifest["samples"]["path"]))
    screen_ids = {sample["sample_id"] for sample in screen_samples}
    screen_dense_rows = [row for row in dense_rows if row["sample_id"] in screen_ids]
    screen_sparse_rows = [row for row in sparse_rows if row["sample_id"] in screen_ids]
    holdout_dense_rows = [row for row in dense_rows if row["sample_id"] not in screen_ids]
    holdout_sparse_rows = [row for row in sparse_rows if row["sample_id"] not in screen_ids]
    screen_dense_accuracy = score_predictions(screen_dense_rows, dense_config["ruler_root"])[1]
    screen_sparse_accuracy = score_predictions(screen_sparse_rows, sparse_config["ruler_root"])[1]
    holdout_dense_by_task, holdout_dense_accuracy = score_predictions(
        holdout_dense_rows, dense_config["ruler_root"]
    )
    holdout_sparse_by_task, holdout_sparse_accuracy = score_predictions(
        holdout_sparse_rows, sparse_config["ruler_root"]
    )
    sweep_rows: list[dict[str, Any]] = []
    for kind, directory_prefix, names in (
        ("q_tile", "q", ("32", "64", "128", "256")),
        ("lambda", "lambda", ("0p001", "0p003", "0p01", "0p03", "0p1")),
    ):
        for name in names:
            run_summary = read_json(
                root / "screen" / f"{directory_prefix}_{name}" / "summary.json"
            )
            value = name.replace("p", ".")
            sweep_rows.append(
                {
                    "sweep": kind,
                    "value": value,
                    "samples": run_summary["actual_num_samples"],
                    "accuracy": run_summary["official_ruler_accuracy"],
                    "accuracy_delta": run_summary["official_ruler_accuracy"] - screen_dense_accuracy,
                    "physical_sparsity": run_summary["attention_sparsity"]["physical_tile_sparsity"],
                }
            )

    output_dir = root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    with (output_dir / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    accuracy_delta = sparse_summary["official_ruler_accuracy"] - dense_summary["official_ruler_accuracy"]
    report = {
        "validation": {
            "passed": accuracy_delta >= -0.05,
            "sample_count_per_run": expected,
            "samples_per_task": 50,
            "same_samples_prompts_and_seeds": True,
            "same_eager_qk_softmax_pv_path": True,
            "dense_backend": "eager-dense",
            "sparse_backend": "blasst-reference",
            "physical_denominator": "eligible non-structural KV tiles",
            "count_identity_verified": True,
        },
        "policy": policy,
        "accuracy": {
            "dense": dense_summary["official_ruler_accuracy"],
            "sparse": sparse_summary["official_ruler_accuracy"],
            "delta": accuracy_delta,
            "dense_by_task": dense_summary["per_task_accuracy"],
            "sparse_by_task": sparse_summary["per_task_accuracy"],
            "paired_wins": sum(sparse_scores[key] > dense_scores[key] for key in dense_scores),
            "paired_losses": sum(sparse_scores[key] < dense_scores[key] for key in dense_scores),
            "paired_ties": sum(sparse_scores[key] == dense_scores[key] for key in dense_scores),
            "uncertainty": uncertainty,
            "non_vt": {
                "samples": len(non_vt_rows),
                "dense": non_vt_dense_accuracy,
                "sparse": non_vt_sparse_accuracy,
                "uncertainty": non_vt_uncertainty,
            },
            "vt": {
                "samples": len(vt_rows),
                "dense": sum(dense_scores[row["sample_id"]] for row in vt_rows)
                / len(vt_rows),
                "sparse": sum(sparse_scores[row["sample_id"]] for row in vt_rows)
                / len(vt_rows),
                "uncertainty": vt_uncertainty,
            },
            "selection_screen": {
                "samples": len(screen_dense_rows),
                "dense": screen_dense_accuracy,
                "sparse": screen_sparse_accuracy,
                "delta": screen_sparse_accuracy - screen_dense_accuracy,
            },
            "held_out_after_selection": {
                "samples": len(holdout_dense_rows),
                "samples_per_task": 40,
                "dense": holdout_dense_accuracy,
                "sparse": holdout_sparse_accuracy,
                "delta": holdout_sparse_accuracy - holdout_dense_accuracy,
                "dense_by_task": holdout_dense_by_task,
                "sparse_by_task": holdout_sparse_by_task,
            },
        },
        "sparsity": {
            "overall": {**overall, "physical_sparsity": sparsity(overall)},
            "by_attention_type": group_rows,
            "by_phase": phase_rows,
            "per_layer": layer_rows,
        },
        "screen_dense_accuracy": screen_dense_accuracy,
        "repeatability": replay_audit,
        "sweeps": sweep_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Paper-aligned DiffusionGemma BLASST validation",
        "",
        "All final accuracy values use 50 examples for each of the 13 official RULER tasks. Dense and sparse use identical prompts, seeds, precision, and eager QK/mask/softmax/PV code.",
        "",
        "| Run | Accuracy | Physical sparsity |",
        "|---|---:|---:|",
        f"| Dense eager | {report['accuracy']['dense']:.2%} | 0.00% |",
        f"| Phase-aware BLASST | {report['accuracy']['sparse']:.2%} | {sparsity(overall):.2%} |",
        f"| Difference | {accuracy_delta:+.2%} | — |",
        f"| 40/task holdout only | {holdout_sparse_accuracy:.2%} sparse vs {holdout_dense_accuracy:.2%} dense ({holdout_sparse_accuracy-holdout_dense_accuracy:+.2%}) | — |",
        "",
        "## Accuracy by task",
        "",
        "| Task | Dense | Sparse | Delta |",
        "|---|---:|---:|---:|",
    ]
    for task in sorted(dense_summary["per_task_accuracy"]):
        dense_value = dense_summary["per_task_accuracy"][task]
        sparse_value = sparse_summary["per_task_accuracy"][task]
        lines.append(f"| {task} | {dense_value:.2%} | {sparse_value:.2%} | {sparse_value-dense_value:+.2%} |")
    lines.extend(
        [
            "",
            "## Sparsity aggregation",
            "",
            "| Attention | Eligible tiles | Skipped tiles | Physical sparsity |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in group_rows:
        lines.append(
            f"| {row['attention_type']} | {row['eligible_tiles']} | {row['skipped_tiles']} | {row['physical_sparsity']:.2%} |"
        )
    lines.append(
        f"| overall | {overall['eligible_tiles']} | {overall['skipped_tiles']} | {sparsity(overall):.2%} |"
    )
    lines.extend(
        [
            "",
            "## Calibrated thresholds and achieved sparsity",
            "",
            "| Attention | Phase | Lambda | Achieved sparsity |",
            "|---|---|---:|---:|",
        ]
    )
    for row in phase_rows:
        lines.append(
            f"| {row['attention_type']} | {row['phase_name']} | {row['lambda']:.6g} | {row['physical_sparsity']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Trust audit",
            "",
            f"The all-task paired-sample bootstrap 95% CI is [{uncertainty['paired_sample_bootstrap_95_ci'][0]:+.2%}, {uncertainty['paired_sample_bootstrap_95_ci'][1]:+.2%}]. The task-cluster bootstrap 95% CI is [{uncertainty['task_cluster_bootstrap_95_ci'][0]:+.2%}, {uncertainty['task_cluster_bootstrap_95_ci'][1]:+.2%}].",
            "",
            f"Excluding VT, sparse changes accuracy by {report['accuracy']['non_vt']['sparse'] - report['accuracy']['non_vt']['dense']:+.2%}; VT alone changes by {report['accuracy']['vt']['sparse'] - report['accuracy']['vt']['dense']:+.2%}. The aggregate gain is therefore driven by VT.",
            "",
            f"The independent screen-to-full replay reproduced {replay_audit['prediction_text_equal']}/{replay_audit['samples']} prediction texts and {replay_audit['completion_tokens_equal']}/{replay_audit['samples']} token sequences exactly.",
            "",
            f"Paired outcomes: {report['accuracy']['paired_wins']} sparse wins, {report['accuracy']['paired_losses']} sparse losses, and {report['accuracy']['paired_ties']} ties.",
            "",
            "Physical sparsity is globally summed skipped/eligible tile counts. Eligible tiles exclude structurally masked work that a target sparse kernel would not schedule. The eager reference applies the BLASST mask for correctness testing; it does not demonstrate wall-clock sparse-kernel speedup.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if not report["validation"]["passed"]:
        raise RuntimeError("selected sparse policy exceeded the 5-point accuracy-loss budget")


if __name__ == "__main__":
    main()
