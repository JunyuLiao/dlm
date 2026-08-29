#!/usr/bin/env python3
"""Create a deterministic per-task subset of an existing RULER manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dllm.evaluation.ruler.io import sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="task to omit; may be repeated",
    )
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("samples-per-task must be positive")

    source_path = args.manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_samples = Path(source["samples"]["path"])
    rows = [
        json.loads(line)
        for line in source_samples.read_text(encoding="utf-8").splitlines()
    ]
    excluded = set(args.exclude_task)
    unknown = excluded - set(source["tasks"])
    if unknown:
        raise ValueError("unknown excluded task(s): " + ", ".join(sorted(unknown)))
    tasks = [task for task in source["tasks"] if task not in excluded]
    if not tasks:
        raise ValueError("excluding tasks left an empty manifest")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    for task in tasks:
        if len(grouped[task]) < args.samples_per_task:
            raise RuntimeError(
                f"task {task} has {len(grouped[task])} samples, requested "
                f"{args.samples_per_task}"
            )

    selected = []
    for index in range(args.samples_per_task):
        selected.extend(grouped[task][index] for task in tasks)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "samples.jsonl"
    write_jsonl(samples_path, selected)
    manifest = {
        **source,
        "schema_version": max(3, int(source.get("schema_version", 1))),
        "requested_num_samples": len(selected),
        "actual_num_samples": len(selected),
        "tasks": tasks,
        "task_counts": {task: args.samples_per_task for task in tasks},
        "parent_manifest": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "selection": (
                f"first {args.samples_per_task} samples per task; "
                f"excluded={sorted(excluded)}"
            ),
        },
        "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
