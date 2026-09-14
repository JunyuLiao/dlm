"""Analyse rich pre-QK supervisor candidates against exact current-QK labels."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .offline_analysis import (
    OUTPUT,
    average_precision,
    binary_auc,
    mass_support,
    neighbor_mean,
    top_budget_mask,
    _finite_spearman,
)


SOURCE = OUTPUT / "rich_same_state"


SIGNAL_METADATA = {
    "sol_proxy": {"pre_current_qk": True, "incremental_cost": "one pooled-Q/K dot per tile plus reductions"},
    "kv_neighbor_proxy": {"pre_current_qk": True, "incremental_cost": "two-neighbor scalar stencil"},
    "previous_step_mass": {"pre_current_qk": True, "incremental_cost": "one stored scalar per tile"},
    "previous_step_proxy": {"pre_current_qk": True, "incremental_cost": "one stored scalar per tile"},
    "previous_step_contribution": {"pre_current_qk": True, "incremental_cost": "one stored scalar per tile; prior exact PV-derived label"},
    "previous_step_drop_error": {"pre_current_qk": True, "incremental_cost": "one stored scalar per tile; prior diagnostic label"},
    "prior_mass_mean": {"pre_current_qk": True, "incremental_cost": "running scalar and count per tile"},
    "value_rms": {"pre_current_qk": True, "incremental_cost": "V-norm reduction over tokens/dimension; reusable across query tiles"},
    "value_max_norm": {"pre_current_qk": True, "incremental_cost": "V-norm reduction and tile maximum; reusable across query tiles"},
    "absolute_recency": {"pre_current_qk": True, "incremental_cost": "closed-form position"},
    "boundary_proximity": {"pre_current_qk": True, "incremental_cost": "closed-form position"},
    "canvas_indicator": {"pre_current_qk": True, "incremental_cost": "one position comparison"},
    "current_exact_max": {"pre_current_qk": False, "incremental_cost": "requires dense current QK"},
    "current_exact_q95": {"pre_current_qk": False, "incremental_cost": "requires dense current QK and quantile"},
    "current_dense_mass": {"pre_current_qk": False, "incremental_cost": "requires current QK and softmax"},
    "current_contribution": {"pre_current_qk": False, "incremental_cost": "requires current QK, softmax, and PV"},
    "current_drop_error": {"pre_current_qk": False, "incremental_cost": "drop-one-tile diagnostic oracle"},
}


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / max(len(values) - 1, 1)


def load_rows(source: Path = SOURCE) -> list[dict]:
    output: list[dict] = []
    for shard in sorted((source / "shards").iterdir()):
        records = json.loads((shard / "records.json").read_text())
        local = []
        with np.load(shard / "rich_snapshots.npz") as arrays:
            for record in records:
                key = record["snapshot_key"]
                eligible = arrays[key + "_eligible"].astype(bool)
                tile_count = len(eligible)
                starts = np.arange(tile_count) * 64
                prefix_length = record["kv_length"] - record["query_length"]
                centers = starts + 31.5
                item = {
                    "example_id": record["example_id"],
                    "split": record["split"],
                    "attention_type": record["attention_type"],
                    "layer": record["layer"],
                    "call": record["call"],
                    "head": record["head"],
                    "query_start": record["query_start"],
                    "kv_length": record["kv_length"],
                    "eligible": eligible,
                    "sol_proxy": arrays[key + "_proxy"].astype(float),
                    "kv_neighbor_proxy": neighbor_mean(arrays[key + "_proxy"].astype(float)),
                    "value_rms": arrays[key + "_tile_value_rms"].astype(float),
                    "value_max_norm": arrays[key + "_tile_value_max"].astype(float),
                    "absolute_recency": centers / max(record["kv_length"], 64),
                    "boundary_proximity": -np.abs(centers - prefix_length) / max(record["kv_length"], 64),
                    "canvas_indicator": (starts >= prefix_length).astype(float),
                    "current_exact_max": arrays[key + "_tile_max"].astype(float),
                    "current_exact_q95": arrays[key + "_tile_q95"].astype(float),
                    "current_dense_mass": arrays[key + "_tile_mass"].astype(float),
                    "current_contribution": arrays[key + "_tile_contribution_l2"].astype(float),
                    "current_drop_error": arrays[key + "_tile_drop_output_l2"].astype(float),
                    "token_entropy": float(np.mean(arrays[key + "_normalized_token_entropy"])),
                    "tile_entropy": float(np.mean(arrays[key + "_normalized_tile_entropy"])),
                }
                local.append(item)
        histories: dict[tuple, list[dict]] = defaultdict(list)
        for item in local:
            histories[(item["layer"], item["head"], item["query_start"], item["kv_length"])].append(item)
        for sequence in histories.values():
            sequence.sort(key=lambda item: item["call"])
            prior_mass = []
            for index, item in enumerate(sequence):
                if index:
                    previous = sequence[index - 1]
                    item["previous_step_mass"] = previous["current_dense_mass"]
                    item["previous_step_proxy"] = previous["sol_proxy"]
                    item["previous_step_contribution"] = previous["current_contribution"]
                    item["previous_step_drop_error"] = previous["current_drop_error"]
                    item["prior_mass_mean"] = np.mean(prior_mass, axis=0)
                prior_mass.append(item["current_dense_mass"])
        output.extend(local)
    return output


def _positive_label(item: dict, label: str, values: np.ndarray) -> np.ndarray:
    if label == "current_dense_mass":
        return mass_support(values, 0.90)
    threshold = np.quantile(values, 0.90)
    return values >= threshold


def analyse(rows: list[dict], split: str = "final") -> dict:
    result: dict[str, dict] = {}
    labels = ("current_exact_max", "current_exact_q95", "current_dense_mass", "current_contribution", "current_drop_error")
    signals = tuple(SIGNAL_METADATA)
    for scope in ("overall", "local", "global"):
        selected = [row for row in rows if row["split"] == split and (scope == "overall" or row["attention_type"] == scope)]
        scope_result = {"records": len(selected), "token_entropy_mean": float(np.mean([r["token_entropy"] for r in selected])), "tile_entropy_mean": float(np.mean([r["tile_entropy"] for r in selected])), "signals": {}}
        for signal in signals:
            entries = [row for row in selected if signal in row]
            if not entries:
                continue
            signal_result = {"records": len(entries), "labels": {}, "budgets": {}}
            for label in labels:
                correlations = []
                aucs = []
                aps = []
                for item in entries:
                    eligible = item["eligible"]
                    score = item[signal][eligible]
                    values = item[label][eligible]
                    correlations.append(_finite_spearman(score, values))
                    positive = _positive_label(item, label, values)
                    aucs.append(binary_auc(positive, score))
                    aps.append(average_precision(positive, score))
                signal_result["labels"][label] = {
                    "mean_spearman": float(np.mean([x for x in correlations if x is not None])),
                    "mean_roc_auc": float(np.mean([x for x in aucs if x is not None])),
                    "mean_average_precision": float(np.mean([x for x in aps if x is not None])),
                }
            for target in (0.25, 0.50, 0.75, 0.90):
                totals = defaultdict(float)
                kept_tiles = eligible_tiles = max_mass_hits = 0
                for item in entries:
                    eligible = item["eligible"]
                    score = item[signal][eligible]
                    keep = top_budget_mask(score, target)
                    kept_tiles += int(keep.sum())
                    eligible_tiles += len(keep)
                    max_mass_hits += int(keep[np.argmax(item["current_dense_mass"][eligible])])
                    for label in labels[2:]:
                        values = item[label][eligible]
                        totals[label + "_kept"] += float(values[keep].sum())
                        totals[label + "_total"] += float(values.sum())
                signal_result["budgets"][f"s{int(target*100)}"] = {
                    "actual_sparsity": 1 - kept_tiles / eligible_tiles,
                    "retained_mass": totals["current_dense_mass_kept"] / totals["current_dense_mass_total"],
                    "retained_contribution_norm_sum": totals["current_contribution_kept"] / totals["current_contribution_total"],
                    "retained_drop_error_sum": totals["current_drop_error_kept"] / totals["current_drop_error_total"],
                    "highest_mass_tile_recall": max_mass_hits / len(entries),
                }
            scope_result["signals"][signal] = signal_result
        result[scope] = scope_result
    return result


def plots(payload: dict, output: Path = OUTPUT) -> list[str]:
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures = []
    selected = ["sol_proxy", "previous_step_mass", "value_rms", "current_exact_max", "current_dense_mass"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for signal in selected:
        metrics = payload["final"]["overall"]["signals"].get(signal)
        if not metrics:
            continue
        x = [metrics["budgets"][f"s{target}"]["actual_sparsity"] for target in (25, 50, 75, 90)]
        axes[0].plot(x, [metrics["budgets"][f"s{target}"]["retained_mass"] for target in (25, 50, 75, 90)], marker="o", label=signal.replace("_", " "))
        axes[1].plot(x, [metrics["budgets"][f"s{target}"]["retained_drop_error_sum"] for target in (25, 50, 75, 90)], marker="o", label=signal.replace("_", " "))
    axes[0].set_ylabel("Retained exact dense mass")
    axes[1].set_ylabel("Retained drop-error importance sum")
    for axis in axes:
        axis.set_xlabel("Actual physical-tile sparsity")
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = directory / "07_rich_supervisor_budget_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path.relative_to(output)))

    labels = ("current_exact_max", "current_exact_q95", "current_dense_mass", "current_contribution", "current_drop_error")
    signals = ("sol_proxy", "kv_neighbor_proxy", "previous_step_mass", "value_rms", "value_max_norm")
    matrix = []
    for signal in signals:
        metrics = payload["final"]["overall"]["signals"].get(signal)
        matrix.append([metrics["labels"][label]["mean_spearman"] if metrics else np.nan for label in labels])
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    image = ax.imshow(matrix, vmin=-0.1, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)), [label.replace("current_", "").replace("_", "\n") for label in labels])
    ax.set_yticks(range(len(signals)), [signal.replace("_", " ") for signal in signals])
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Mean per-snapshot Spearman")
    fig.tight_layout()
    path = directory / "08_supervisor_label_alignment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path.relative_to(output)))
    return figures


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    audit = json.loads((source / "audit.json").read_text())
    if not audit.get("complete"):
        raise RuntimeError("rich source audit is incomplete")
    rows = load_rows(source)
    payload = {
        "protocol": {
            "source": str(source),
            "records": len(rows),
            "samples": len({row["example_id"] for row in rows}),
            "split_policy": "calibration summaries are exploratory; final eight prompts are held out",
            "fixed_budget": "top-k eligible physical 64x64 tiles per sampled query block",
            "labels": {
                "current_exact_max": "maximum valid token-level QK score in physical tile",
                "current_exact_q95": "95th percentile valid token-level QK score in physical tile",
                "current_dense_mass": "dense probability mass summed over sampled valid query rows",
                "current_contribution": "sum-of-squares norm of unnormalized per-tile PV contributions",
                "current_drop_error": "sum-of-squares norm of exact drop-one-tile, renormalized output delta",
            },
            "signal_metadata": SIGNAL_METADATA,
            "limitations": [
                "one head/query block per prompt, all heads represented only across prompts",
                "drop-one-tile effects are not additive multi-tile output error",
                "diagnostic reductions are not deployment overhead measurements",
            ],
        },
        "calibration": analyse(rows, "calibration"),
        "final": analyse(rows, "final"),
    }
    payload["figures"] = plots(payload, output)
    (output / "rich_supervisor_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({"records": result["protocol"]["records"], "figures": result["figures"]}, indent=2))


if __name__ == "__main__":
    main()
