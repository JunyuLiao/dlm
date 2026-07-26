#!/usr/bin/env python3
"""Create cross-layer/head/step predictability heatmaps and head clusters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tracing.trace_schema import load_trace_directory
from .proxy_relationships import relationship_metrics


def _coordinate(row: np.void) -> tuple[str, int, int]:
    return str(row["sample_id"]), int(row["query_tile"]), int(row["kv_tile"])


def pair_matrix(records: np.ndarray, axis: str) -> tuple[np.ndarray, list[int]]:
    values = sorted({int(row[axis]) for row in records})
    matrix = np.full((len(values), len(values)), np.nan)
    value_to_index = {value: index for index, value in enumerate(values)}
    fixed_fields = {
        "layer": ("sample_id", "denoising_iteration", "head", "query_tile", "kv_tile"),
        "head": ("sample_id", "denoising_iteration", "layer", "query_tile", "kv_tile"),
        "denoising_iteration": ("sample_id", "layer", "head", "query_tile", "kv_tile"),
    }[axis]
    groups: dict[tuple[object, ...], dict[int, np.void]] = {}
    for row in records:
        key = tuple(str(row[field]) if field == "sample_id" else int(row[field]) for field in fixed_fields)
        groups.setdefault(key, {})[int(row[axis])] = row
    collected: dict[tuple[int, int], list[tuple[np.void, np.void, str]]] = {}
    for rows in groups.values():
        for source_value, source in rows.items():
            for target_value, target in rows.items():
                if source_value < target_value:
                    collected.setdefault((source_value, target_value), []).append((source, target, str(source_value)))
    for (source, target), pairs in collected.items():
        matrix[value_to_index[source], value_to_index[target]] = float(
            relationship_metrics(pairs).get("skip_precision", np.nan)
        )
    return matrix, values


def save_matrix(path: Path, matrix: np.ndarray, labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source\\target", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def head_clusters(records: np.ndarray, threshold: float = 0.8) -> list[dict[str, object]]:
    results = []
    for layer in sorted(set(records["layer"].tolist())):
        for bucket in sorted(set(records["noise_bucket"].tolist())):
            selected = records[(records["layer"] == layer) & (records["noise_bucket"] == bucket)]
            masks: dict[int, dict[tuple[str, int, int, int], bool]] = {}
            for row in selected:
                coordinate = (*_coordinate(row), int(row["denoising_iteration"]))
                masks.setdefault(int(row["head"]), {})[coordinate] = bool(row["exact_skip"])
            clusters: list[list[int]] = []
            for head in sorted(masks):
                placed = False
                for cluster in clusters:
                    leader = cluster[0]
                    common = masks[head].keys() & masks[leader].keys()
                    if not common:
                        continue
                    left = {key for key in common if masks[head][key]}
                    right = {key for key in common if masks[leader][key]}
                    union = left | right
                    jaccard = len(left & right) / len(union) if union else 1.0
                    if jaccard >= threshold:
                        cluster.append(head)
                        placed = True
                        break
                if not placed:
                    clusters.append([head])
            results.append({"layer": int(layer), "noise_bucket": str(bucket), "clusters": clusters})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--output-dir", default="artifacts/proxy_analysis")
    args = parser.parse_args()
    records = load_trace_directory(args.trace_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for name, axis in (("cross_layer", "layer"), ("cross_head", "head"), ("cross_step", "denoising_iteration")):
        matrix, labels = pair_matrix(records, axis)
        save_matrix(output / f"{name}_precision.csv", matrix, labels)
        matrices[name] = (matrix, labels)
    clusters = head_clusters(records)
    (output / "head_clusters.json").write_text(json.dumps(clusters, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for name, (matrix, labels) in matrices.items():
        figure, axes = plt.subplots(figsize=(6, 5))
        image = axes.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axes.set_xticks(range(len(labels)), labels)
        axes.set_yticks(range(len(labels)), labels)
        axes.set_xlabel("target")
        axes.set_ylabel("legally earlier source")
        axes.set_title(f"{name.replace('_', ' ')} PRE_SKIP precision")
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        figure.savefig(output / f"{name}_precision.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
