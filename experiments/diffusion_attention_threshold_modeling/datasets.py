"""Deterministic corpus selection and calibration/validation splitting."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


MATH500_SUBSET_SEED = 20260816


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                value = {"prompt": value}
            if not isinstance(value, dict):
                raise ValueError(f"row {index} is not a JSON object or string")
            prompt = value.get("prompt", value.get("input"))
            if not isinstance(prompt, str):
                raise ValueError(f"row {index} has no string prompt/input")
            row = dict(value)
            row.update(prompt=prompt, request_id=str(value.get("request_id", value.get("sample_id", value.get("id", index)))))
            rows.append(row)
    if not rows:
        raise ValueError(f"no prompts found in {path}")
    return rows


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def balanced_ruler_selection(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin tasks after deterministic within-task shuffling."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("task", "unclassified")), []).append(dict(row))
    for task, group in groups.items():
        group.sort(key=lambda row: _rank(seed, task, row["request_id"]))
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for task in sorted(groups):
            if depth < len(groups[task]):
                selected.append(groups[task][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(f"requested {count} balanced RULER prompts, only {len(selected)} available")
    return selected


def stratified_split(rows: list[dict[str, Any]], seed: int, validation_fraction: float = 0.2) -> list[dict[str, Any]]:
    """Assign deterministic 80/20 splits within corpus/task strata."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("corpus", "unknown")), str(row.get("task", "unclassified")))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: _rank(seed, *key, row["request_id"]))
        validation_count = max(1, int(round(len(ordered) * validation_fraction))) if len(ordered) >= 2 else 0
        validation_ids = {id(row) for row in ordered[:validation_count]}
        for row in group:
            item = dict(row)
            item["split"] = "validation" if id(row) in validation_ids else "calibration"
            output.append(item)
    return output


def load_ruler(path: str | Path, count: int, seed: int, corpus: str = "ruler8k") -> list[dict[str, Any]]:
    rows = balanced_ruler_selection(read_jsonl(path), count, seed)
    for row in rows:
        row["corpus"] = corpus
    return stratified_split(rows, seed)


def _math_prompt_template(root: Path) -> str:
    path = root / "benchmarks/prompts/generic/math.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("user: |-") + 1
    except ValueError as error:
        raise RuntimeError("unexpected NeMo-Gym generic/math prompt format") from error
    content = []
    for line in lines[start:]:
        if line and not line.startswith("  "):
            break
        content.append(line[2:] if line.startswith("  ") else "")
    template = "\n".join(content).rstrip()
    if "{question}" not in template or "\\boxed{{}}" not in template:
        raise RuntimeError("NeMo-Gym math prompt contract changed")
    return template


def load_math500(root: str | Path, count: int, split_seed: int) -> list[dict[str, Any]]:
    root = Path(root)
    source = root / "benchmarks/math-500/data/math-500_benchmark.jsonl"
    problems = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                problems.append(json.loads(line))
    if len(problems) != 500:
        raise RuntimeError(f"expected 500 Math500 rows, found {len(problems)}")
    if count > 100:
        raise ValueError("distribution profiling is capped at 100 Math500 prompts")
    fixed = sorted(random.Random(MATH500_SUBSET_SEED).sample(range(500), 50))
    remaining = [index for index in range(500) if index not in set(fixed)]
    extension = random.Random(split_seed).sample(remaining, max(0, count - 50))
    # The first 50 always remain the canonical accuracy subset; extension rows
    # are distribution-only and never enter eval-math500.
    indices = fixed[: min(count, 50)] + sorted(extension)
    template = _math_prompt_template(root)
    rows = []
    for subset_index, dataset_index in enumerate(indices):
        problem = dict(problems[dataset_index])
        rows.append({
            **problem,
            "request_id": f"math500-{dataset_index:03d}",
            "prompt": template.format(question=problem["problem"]),
            "corpus": "math500",
            "task": str(problem.get("subject", "math")),
            "subset_index": subset_index,
            "dataset_index": dataset_index,
        })
    return stratified_split(rows, split_seed)
