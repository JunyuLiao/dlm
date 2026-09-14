"""Deterministic 50-problem MATH500 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.diffusion_attention_threshold_modeling.datasets import load_math500


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prepare_manifest(nemo_gym_root: Path, output_dir: Path, split_seed: int) -> list[dict[str, Any]]:
    rows = load_math500(nemo_gym_root, 50, split_seed)
    manifest = []
    for row in rows:
        manifest.append({
            **row,
            "prompt_hash": prompt_hash(str(row["prompt"])),
            "inference_seed": 42 + int(row["dataset_index"]),
            "generation_budget": 2048,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError("existing manifest differs from the deterministic MATH500 selection")
    path.write_text(payload, encoding="utf-8")
    audit = {
        "schema_version": 1,
        "num_samples": len(manifest),
        "unique_request_ids": len({row["request_id"] for row in manifest}) == 50,
        "unique_prompt_hashes": len({row["prompt_hash"] for row in manifest}) == 50,
        "split_seed": split_seed,
        "dataset_indices": [row["dataset_index"] for row in manifest],
    }
    (output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50:
        raise RuntimeError(f"expected 50 manifest rows, found {len(rows)}")
    if len({row["request_id"] for row in rows}) != 50:
        raise RuntimeError("manifest request IDs are not unique")
    if any(prompt_hash(str(row["prompt"])) != row["prompt_hash"] for row in rows):
        raise RuntimeError("manifest prompt hash audit failed")
    return rows
