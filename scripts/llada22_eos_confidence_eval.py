"""Paired HumanEval validation of LLaDA EOS-confidence masking.

Uses the pinned original numerical worker and keeps all raw task data private.
This is a new decoding variant; it never alters the original benchmark arms.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

import torch


BASE_ARM = "value_v2_gaussian32_reuse_r4"
V2_THRESHOLD = 0.8944271909999159
VARIANTS = (
    ("dense_native_eos_conf0", "dense_native"),
    ("value_v2_gaussian32_reuse_r4_eos_conf0", BASE_ARM),
)
PIN = "224f8f11173bd16e5cf10054a963c74233758cf2"
MODEL_REVISION = "e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_eos_confidence(confidence, predicted_tokens, eos_id):
    if confidence.shape != predicted_tokens.shape:
        raise ValueError("confidence and predicted token tensors must have identical shape")
    return confidence.masked_fill(predicted_tokens == eos_id, 0.)


def pinned_worker(base):
    snapshot = base / "private/execution" / PIN
    source = snapshot / "experiments/numerical_qk_reuse"
    if not (snapshot / "experiments/__init__.py").exists():
        package = types.ModuleType("experiments")
        package.__path__ = [str(snapshot / "experiments")]
        package.__package__ = "experiments"
        sys.modules["experiments"] = package
    sys.path[:0] = [str(snapshot), str(snapshot / "scripts")]
    attention = importlib.import_module("experiments.numerical_qk_reuse.multi_model_attention")
    runtime = importlib.import_module("experiments.numerical_qk_reuse.multi_model_runtime")
    worker = importlib.import_module("multi_model_worker")
    if Path(attention.__file__).resolve() != (source / "multi_model_attention.py").resolve():
        raise RuntimeError("attention module escaped the pinned snapshot")
    if Path(runtime.__file__).resolve() != (source / "multi_model_runtime.py").resolve():
        raise RuntimeError("runtime module escaped the pinned snapshot")
    return snapshot, attention, runtime, worker


def run(base, out):
    if not (out / "protocol.json").exists() or (out / "generation_complete.json").exists():
        raise RuntimeError("expected a fresh frozen protocol directory")
    snapshot, attention, runtime, worker = pinned_worker(base)
    expected = json.loads((out / "protocol.json").read_text())
    for field, path in (("worker_sha256", snapshot / "scripts/multi_model_worker.py"),
                        ("runtime_sha256", Path(runtime.__file__)),
                        ("attention_sha256", Path(attention.__file__))):
        if sha(path) != expected[field]:
            raise RuntimeError("source binding mismatch: " + field)
    if (expected["model_revision"] != MODEL_REVISION or
            expected["arms"]["value_v2_gaussian32_reuse_r4_eos_conf0"]["threshold"] != V2_THRESHOLD):
        raise RuntimeError("frozen model or V2 threshold differs")
    data_path = base / "private/data/llada22/humaneval.json"
    dataset = json.loads(data_path.read_text())
    if len(dataset) != 164 or any(row["seeds"] != [42] for row in dataset):
        raise RuntimeError("unexpected HumanEval cohort")
    identity_path = base / "llada22_model_identity.json"
    identity = json.loads(identity_path.read_text())
    if identity["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision differs from original evaluation")

    out_private = out / "private"
    out_private.mkdir(exist_ok=True)
    if sha(Path(__file__)) != expected["runner_sha256"]:
        raise RuntimeError("runner differs from frozen protocol")
    data_link = out_private / "data"
    if not data_link.exists():
        data_link.symlink_to(base / "private/data", target_is_directory=True)
    os_env = __import__("os").environ
    os_env.update(HF_MODULES_CACHE=str(base / "private/hf_modules"),
                  TRITON_CACHE_DIR=str(base / "private/triton_cache"))
    harness = worker.ModelHarness(base, "llada22").load()
    if harness.tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer EOS ID is missing")
    eos_id = harness.tokenizer.eos_token_id
    original_sample = harness.model._diffusion_sample
    mask_counts = {name: 0 for name, _ in VARIANTS}

    def eos_masked_sample(logits, temperature=1.0, top_k=None, top_p=None):
        tokens, confidence = original_sample(logits, temperature, top_k, top_p)
        mask_counts[current_variant[0]] += int((tokens == eos_id).sum().item())
        return tokens, zero_eos_confidence(confidence, tokens, eos_id)

    # This changes confidence-based selection only. EOS remains in the logits
    # and can still be emitted by native forced progress and stopping behavior.
    harness.model._diffusion_sample = eos_masked_sample
    current_variant = [None]
    try:
        for arm, selector_arm in VARIANTS:
            current_variant[0] = arm
            mask_counts[arm] = 0
            threshold = V2_THRESHOLD
            root = out / "runs/llada22/humaneval" / arm
            cells = [(row, 42) for row in dataset]
            started_total = time.time()
            for shard in range(1, 5):
                leaf = root / f"shard{shard:03}/attempt001"
                leaf.mkdir(parents=True, exist_ok=False)
                shard_cells = [(row, seed) for index, (row, seed) in enumerate(cells)
                               if index % 4 == shard - 1]
                harness.runtime.arm = selector_arm
                harness.runtime.threshold = threshold
                binding = {
                    "model": "llada22", "suite": "humaneval", "arm": arm,
                    "selector_arm": selector_arm, "shard": shard, "shards": 4,
                    "cells": len(shard_cells), "threshold": threshold,
                    "eos_confidence_policy": "set predicted EOS confidence to zero for confidence-based token selection; retain logits, native forced progress, and EOS stop",
                    "eos_token_id": eos_id,
                    "manifest_sha256": sha(data_path), "source_sha256": sha(Path(worker.__file__)),
                    "protocol_sha256": sha(out / "protocol.json"),
                    "attention_source_sha256": sha(Path(attention.__file__)),
                    "runtime_sha256": sha(Path(runtime.__file__)),
                    "model_revision": MODEL_REVISION,
                    "host_gpu_uuid": __import__("subprocess").check_output(
                        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip(),
                    "sampler": {"block_length": 32, "steps": 32, "threshold": .5,
                                "editing_threshold": 0., "max_post_steps": 16,
                                "temperature": 0., "eos_early_stop": True, "generation_cap": 1024},
                    "query_tile_rows": 128, "kv_tile_tokens": 64,
                    "reuse_interval_calls": 4 if "reuse" in selector_arm else None,
                }
                worker.immutable(leaf / "config.json", binding)
                shard_started = time.time()
                for index, (row, seed) in enumerate(shard_cells, 1):
                    count_before = mask_counts[arm]
                    public, private = worker.run_cell(
                        harness, row, worker.CAPS["llada22"]["humaneval"], seed)
                    public.update(model="llada22", suite="humaneval", arm=arm,
                                  threshold=threshold, manifest_sha256=binding["manifest_sha256"],
                                  source_sha256=binding["source_sha256"],
                                  protocol_sha256=binding["protocol_sha256"],
                                  runtime_sha256=binding["runtime_sha256"],
                                  model_revision=MODEL_REVISION,
                                  eos_confidences_zeroed=mask_counts[arm] - count_before)
                    worker.append(leaf / "private/generation.private.jsonl",
                                  dict(private, model="llada22", suite="humaneval", arm=arm))
                    worker.append(leaf / "records.jsonl", public)
                    print(json.dumps({"event": "cell_complete", "arm": arm,
                                      "shard": shard, "index": index,
                                      "cells": len(shard_cells),
                                      "output_tokens": public["output_tokens"],
                                      "N": public["counts"]["denoising_forwards"],
                                      "C": public["counts"]["canvases"]}), flush=True)
                receipt = dict(binding, gpu_worker_seconds=time.time() - shard_started,
                               eos_confidences_zeroed=sum(
                                   r["eos_confidences_zeroed"]
                                   for r in map(json.loads, (leaf / "records.jsonl").read_text().splitlines())),
                               records_sha256=sha(leaf / "records.jsonl"))
                worker.immutable(leaf / "receipt.json", receipt)
                worker.immutable(leaf / "final_complete.json",
                                 {"status": "complete", "cells": len(shard_cells)})
                print(json.dumps({"event": "shard_complete", "arm": arm,
                                  "shard": shard, "seconds": receipt["gpu_worker_seconds"]}), flush=True)
            worker.immutable(out / f"{arm}_complete.json", {
                "status": "complete", "requests": len(cells), "seed": 42,
                "selector_arm": selector_arm, "threshold": threshold,
                "eos_confidences_zeroed": mask_counts[arm],
                "allocated_worker_seconds": time.time() - started_total})
    finally:
        harness.model._diffusion_sample = original_sample
    worker.immutable(out / "generation_complete.json", {
        "status": "complete", "models": ["llada22"], "suite": "humaneval",
        "requests_per_arm": 164, "arms": [arm for arm, _ in VARIANTS],
        "eos_confidence_policy": "confidence zero only; logits and native termination retained"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.base.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
