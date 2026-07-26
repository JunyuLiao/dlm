"""Leakage-free joins and dependency-light relationship metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from tracing.trace_schema import load_trace_directory

IDENTITY = ("sample_id", "denoising_iteration", "layer", "head", "query_tile", "kv_tile")


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _key(row: np.void, *, step: int | None = None, layer: int | None = None, head: int | None = None) -> tuple[object, ...]:
    return (
        str(row["sample_id"]),
        int(row["denoising_iteration"] if step is None else step),
        int(row["layer"] if layer is None else layer),
        int(row["head"] if head is None else head),
        int(row["query_tile"]),
        int(row["kv_tile"]),
    )


def joined_pairs(records: np.ndarray, relation: str) -> list[tuple[np.void, np.void, str]]:
    """Return (source, target, source label) using coordinate-only joins."""
    index = {_key(row): row for row in records}
    pairs = []
    for target in records:
        if relation == "previous_step":
            source = index.get(_key(target, step=int(target["denoising_iteration"]) - 1))
            if source is not None:
                pairs.append((source, target, "previous_step"))
        elif relation == "previous_layer":
            source = index.get(_key(target, layer=int(target["layer"]) - 1))
            if source is not None:
                pairs.append((source, target, "previous_layer"))
        elif relation == "earlier_heads":
            for source_head in range(int(target["head"])):
                source = index.get(_key(target, head=source_head))
                if source is not None:
                    pairs.append((source, target, f"head:{source_head}"))
        else:
            raise ValueError(f"unknown relation: {relation}")
    return pairs


def split_is_calibration(sample_id: str, fraction: float = 0.5) -> bool:
    """Stable prompt/sample-level split; individual tiles never cross it."""
    value = int.from_bytes(hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), "little")
    return value / 2**64 < fraction


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels.astype(bool)
    count_pos, count_neg = int(positives.sum()), int((~positives).sum())
    if not count_pos or not count_neg:
        return float("nan")
    ranks = _rank(scores)
    return float((ranks[positives].sum() - count_pos * (count_pos - 1) / 2) / (count_pos * count_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    result = 0.0
    for a in (0, 1):
        for b in (0, 1):
            joint = float(((left == a) & (right == b)).mean())
            if joint:
                result += joint * math.log(joint / (float((left == a).mean()) * float((right == b).mean())))
    return result


def relationship_metrics(pairs: list[tuple[np.void, np.void, str]]) -> dict[str, float | int]:
    if not pairs:
        return {"tiles": 0}
    source_skip = np.array([int(source["exact_skip"]) for source, _, _ in pairs], dtype=np.uint8)
    target_skip = np.array([int(target["exact_skip"]) for _, target, _ in pairs], dtype=np.uint8)
    predicted = source_skip.astype(bool)
    actual = target_skip.astype(bool)
    intersection = int((predicted & actual).sum())
    union = int((predicted | actual).sum())
    false = int((predicted & ~actual).sum())
    finite_pairs = [
        (float(source["log_score"]), float(target["log_score"]))
        for source, target, _ in pairs
        if math.isfinite(float(source["log_score"])) and math.isfinite(float(target["log_score"]))
    ]
    if len(finite_pairs) >= 2:
        source_log, target_log = np.asarray(finite_pairs, dtype=np.float64).T
        pearson = float(np.corrcoef(source_log, target_log)[0, 1])
        spearman = float(np.corrcoef(_rank(source_log), _rank(target_log))[0, 1])
    else:
        pearson = spearman = float("nan")
    source_score = np.nan_to_num(
        np.array([-float(source["log_score"]) for source, _, _ in pairs]),
        neginf=-1e6,
        posinf=1e6,
    )
    safe_ratio = lambda a, b: a / b if b else float("nan")
    return {
        "tiles": len(pairs),
        "source_skip_tiles": int(predicted.sum()),
        "target_skip_tiles": int(actual.sum()),
        "p_target_skip_given_source_skip": safe_ratio(intersection, int(predicted.sum())),
        "p_target_keep_given_source_skip": safe_ratio(false, int(predicted.sum())),
        "skip_mask_agreement": float((predicted == actual).mean()),
        "jaccard": safe_ratio(intersection, union),
        "skip_precision": safe_ratio(intersection, int(predicted.sum())),
        "skip_recall": safe_ratio(intersection, int(actual.sum())),
        "false_skip_rate": safe_ratio(false, int(predicted.sum())),
        "false_skip_new_max": int(
            sum(bool(source["exact_skip"]) and not bool(target["exact_skip"]) and bool(target["introduced_new_max"])
                for source, target, _ in pairs)
        ),
        "pearson_log_score": pearson,
        "spearman_log_score": spearman,
        "auroc_target_skip": _auroc(target_skip, source_score),
        "average_precision_target_skip": _average_precision(target_skip, source_score),
        "mutual_information_skip_nats": _mutual_information(source_skip, target_skip),
    }


def conditional_score_bins(pairs: list[tuple[np.void, np.void, str]], bins: int = 10) -> list[dict[str, float | int]]:
    finite = [(float(a["log_score"]), float(b["log_score"]), int(b["exact_skip"])) for a, b, _ in pairs
              if math.isfinite(float(a["log_score"])) and math.isfinite(float(b["log_score"]))]
    if not finite:
        return []
    values = np.asarray(finite)
    edges = np.unique(np.quantile(values[:, 0], np.linspace(0, 1, bins + 1)))
    result = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (values[:, 0] >= low) & ((values[:, 0] <= high) if index == len(edges) - 2 else (values[:, 0] < high))
        if selected.any():
            result.append({
                "bin": index,
                "source_log_low": float(low),
                "source_log_high": float(high),
                "tiles": int(selected.sum()),
                "target_log_mean": float(values[selected, 1].mean()),
                "target_log_q95": float(np.quantile(values[selected, 1], 0.95)),
                "target_skip_rate": float(values[selected, 2].mean()),
            })
    return result


def analyze(trace_dir: str | Path, relation: str, output_dir: str | Path) -> dict[str, object]:
    records = load_trace_directory(trace_dir)
    pairs = joined_pairs(records, relation)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []

    def add(group: str, value: str, selected: list[tuple[np.void, np.void, str]]) -> None:
        summaries.append({"group": group, "value": value, **relationship_metrics(selected)})

    add("global", "all", pairs)
    for field in ("noise_bucket", "layer", "head", "kv_group", "diffusion_block", "sequence_length"):
        for value in sorted({str(target[field]) for _, target, _ in pairs}):
            add(field, value, [pair for pair in pairs if str(pair[1][field]) == value])
    for region in ("is_local", "is_distant", "is_diagonal", "is_sink"):
        add("region", region, [pair for pair in pairs if bool(pair[1][region])])
    if relation == "earlier_heads":
        for label in sorted({label for _, _, label in pairs}):
            add("source_head", label, [pair for pair in pairs if pair[2] == label])

    columns = sorted({key for row in summaries for key in row})
    with (directory / f"{relation}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "relation": relation,
        "records": int(len(records)),
        "joined_pairs": len(pairs),
        "sample_split": "stable hash by sample_id (never individual tiles)",
        "summaries": summaries,
        "conditional_source_score_bins": conditional_score_bins(pairs),
    }
    (directory / f"{relation}_metrics.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload
