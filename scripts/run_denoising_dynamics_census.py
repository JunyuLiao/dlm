"""Run the frozen dense native trajectory census.

This command is deliberately separate from clean timing and sparse-selector
evaluation.  It writes only sanitized numeric traces; prompts, answers, token
IDs, and completions stay in the local private manifest or in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from dllm.models import GenerationRequest, create_adapter
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import (
    MODEL,
    MODEL_REVISION,
    OUT,
    atomic_write,
    sha_file,
)
from experiments.diffusion_gemma_sparse_attention_research.native_observer import NativeTrajectoryObserver


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("private frozen manifest must contain exactly 30 rows")
    return rows


def _request(row: dict[str, Any], max_new_tokens: int | None = None) -> GenerationRequest:
    return GenerationRequest(
        prompt=str(row["prompt"]),
        max_new_tokens=int(max_new_tokens or row["generation_budget"]),
        block_size=256,
        temperature=0.0,
        seed=int(row["seed"]),
        extra={"thinking": bool(row.get("thinking", True))},
    )


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _output_record(row: dict[str, Any], generated: Any, observer: NativeTrajectoryObserver, *, smoke: bool) -> dict[str, Any]:
    text = str(generated.text)
    return {
        "schema": "denoising_dynamics_record_v1",
        "id": row["id"],
        "source_id": row["source_id"],
        "benchmark": row["benchmark"],
        "prompt_hash": row["prompt_hash"],
        "seed": int(row["seed"]),
        "thinking": bool(row.get("thinking", True)),
        "smoke": bool(smoke),
        "output_token_count": len(generated.completion_tokens),
        "output_hash": hashlib.sha256(text.encode()).hexdigest(),
        "termination_reason": generated.termination_reason,
        "elapsed_seconds": float(generated.elapsed_seconds),
        "model_evaluations": generated.model_evaluations,
        "generation_metadata": generated.metadata,
        "trajectory": observer.records,
        "trajectory_summary": observer.summary(),
    }


def run(
    *,
    root: Path = OUT,
    model_path: str = MODEL,
    revision: str = MODEL_REVISION,
    item_ids: list[str] | None = None,
    max_items: int | None = None,
    max_new_tokens: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    private_manifest = root / "private" / "manifest.json"
    protocol_path = root / "protocol.json"
    if not private_manifest.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("run dynamics_screen.py first to freeze the private manifest and protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("arms") != ["dense_native_hf"]:
        raise RuntimeError("this runner accepts only the frozen dense native D0 arm")
    rows = _rows(private_manifest)
    if item_ids:
        wanted = set(item_ids)
        rows = [row for row in rows if row["id"] in wanted]
        if len(rows) != len(wanted):
            raise ValueError(f"unknown item IDs: {sorted(wanted - {r['id'] for r in rows})}")
    if max_items is not None:
        rows = rows[: int(max_items)]
    if not rows:
        raise ValueError("no rows selected")

    output_dir = root / ("smoke" if smoke else "native_dense")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "denoising_dynamics_run_config_v1",
        "model_path": model_path,
        "revision": revision,
        "protocol_sha256": sha_file(protocol_path),
        "private_manifest_sha256": sha_file(private_manifest),
        "diagnostic_only": True,
        "native_sampler_unchanged": True,
        "extra_forwards": 0,
        "extra_rng_draws": 0,
        "smoke": bool(smoke),
        "max_new_tokens_override": max_new_tokens,
        "selected_ids": [row["id"] for row in rows],
    }
    config_path = output_dir / "run_config.json"
    atomic_write(config_path, config)
    done_path = output_dir / "records.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if done_path.exists():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("run_config_sha256") != sha_file(config_path):
                    raise RuntimeError("existing records belong to another frozen run config")
                done[item["id"]] = item

    adapter = create_adapter("diffusion_gemma", model_path, device="cuda", precision="bfloat16", revision=revision).load()
    observer = NativeTrajectoryObserver()
    try:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            observer.begin_request(row["id"])
            started = time.time()
            with observer.installed(adapter.model):
                generated = adapter.generate(_request(row, max_new_tokens=max_new_tokens))
            if generated.prompt_tokens != adapter.encode_prompt(row["prompt"], {"thinking": row.get("thinking", True)}):
                raise RuntimeError(f"prompt token replay mismatch for {row['id']}")
            record = _output_record(row, generated, observer, smoke=smoke)
            record["run_config_sha256"] = sha_file(config_path)
            record["wall_seconds"] = float(time.time() - started)
            _write_jsonl(done_path, record)
            done[row["id"]] = record
            atomic_write(output_dir / "progress.json", {
                "completed": len(done),
                "expected": len(rows),
                "latest_id": row["id"],
                "updated_unix": time.time(),
            })
            print(json.dumps({"completed": len(done), "expected": len(rows), "id": row["id"], "calls": len(observer.records)}), flush=True)
    finally:
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "schema": "denoising_dynamics_census_summary_v1",
        "run_config_sha256": sha_file(config_path),
        "completed": len(done),
        "expected": len(rows),
        "complete": len(done) == len(rows),
        "smoke": bool(smoke),
        "records": sorted(done),
    }
    atomic_write(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--id", action="append", dest="item_ids")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(root=args.root, model_path=args.model_path, revision=args.revision,
                         item_ids=args.item_ids, max_items=args.max_items,
                         max_new_tokens=args.max_new_tokens, smoke=args.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
