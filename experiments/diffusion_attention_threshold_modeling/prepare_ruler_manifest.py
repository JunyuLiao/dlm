"""Build a fixed, tokenizer-specific 50-example RULER 16K manifest.

The pinned RULER generator is not rerun here: the repository's cached raw
validation shards are immutable inputs, and this script records their hashes
and deterministic row selection in the resulting manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.io import sha256_file, write_json, write_jsonl
from dllm.evaluation.ruler.official import (
    DEFAULT_TASKS,
    RULER_COMMIT,
    load_configuration,
    verify_checkout,
)
from dllm.models import create_adapter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_manifest(
    *,
    raw_root: str | Path,
    ruler_root: str | Path,
    model_path: str,
    adapter_name: str,
    revision: str | None,
    output_dir: str | Path,
    num_samples: int = 50,
    context_length: int = 16384,
    seed: int = 42,
) -> Path:
    if num_samples not in (2, 50):
        raise ValueError("the canonical study supports only 2-example smokes or 50 samples")
    tasks = tuple(DEFAULT_TASKS) if num_samples == 50 else ("niah_multikey_1", "niah_multivalue")
    if num_samples % len(tasks):
        raise ValueError("the canonical task allocation must be balanced")
    per_task = num_samples // len(tasks)
    raw_root = Path(raw_root).resolve()
    ruler_root = Path(ruler_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = verify_checkout(ruler_root)
    if provenance["commit"] != RULER_COMMIT:
        raise ValueError("unexpected RULER checkout")
    customized, base = load_configuration(ruler_root)
    adapter = create_adapter(
        adapter_name,
        model_path,
        device="cpu",
        precision="float32",
        revision=revision,
    ).load_tokenizer()

    records: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}
    for task_index, task in enumerate(tasks):
        source = raw_root / task / "validation.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        source_files[task] = sha256_file(source)
        rows = sorted(_read_jsonl(source), key=lambda row: (int(row["index"]), str(row["input"])))
        if len(rows) < per_task:
            raise ValueError(f"{task} has {len(rows)} rows, need {per_task}")
        task_base = str(customized[task]["task"])
        tokens_to_generate = int(base[task_base]["tokens_to_generate"])
        for selection_index, raw in enumerate(rows[:per_task]):
            prompt = str(raw["input"]) + str(raw["answer_prefix"])
            actual_prompt_length = len(adapter.encode_prompt(prompt, {}))
            ordinal = len(records)
            sample_id = f"ruler_{context_length}_{task}_{selection_index:03d}"
            inference_seed = seed * 1_000_000 + ordinal
            records.append({
                "actual_prompt_length": actual_prompt_length,
                "answer_prefix": str(raw["answer_prefix"]),
                "generator_seed": seed * 100_000 + context_length + task_index * 1_000 + int(raw["index"]),
                "inference_seed": inference_seed,
                "official_index": int(raw["index"]),
                "outputs": list(raw["outputs"]),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "sample_id": sample_id,
                "target_length": context_length,
                "task": task,
                "task_base": task_base,
                "tokens_to_generate": tokens_to_generate,
            })
    if len(records) != num_samples:
        raise AssertionError(len(records))
    samples_path = output_dir / "samples.jsonl"
    write_jsonl(samples_path, records)
    manifest = {
        "schema_version": 1,
        "ruler": provenance,
        "model_adapter": adapter_name,
        "tokenizer_path": model_path,
        "tokenizer_revision": revision,
        "prompt_configuration": adapter.prompt_configuration({}),
        "context_length": context_length,
        "requested_num_samples": num_samples,
        "actual_num_samples": len(records),
        "seed": seed,
        "tasks": list(tasks),
        "task_counts": {task: per_task for task in tasks},
        "selection": {
            "source_raw_root": str(raw_root),
            "source_sha256": source_files,
            "rule": "sort each pinned raw validation shard by (index, input), select first 10",
        },
        "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", choices=("diffusion_gemma", "fast_dllm_v2"), required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()
    print(build_manifest(
        raw_root=args.raw_root,
        ruler_root=args.ruler_root,
        model_path=args.model_path,
        adapter_name=args.adapter,
        revision=args.revision,
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
    ))


if __name__ == "__main__":
    main()
