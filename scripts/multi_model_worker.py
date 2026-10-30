"""Serial model worker for immutable calibration and benchmark shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.numerical_qk_reuse.multi_model_attention import ARMS
from experiments.numerical_qk_reuse.multi_model_runtime import ModelHarness


CAPS = {
    "llada22": {"aime26": 32768, "humaneval": 1024, "ruler128k": 128},
    "ultrallada": {"aime26": 4096, "humaneval": 1024, "ruler128k": 128},
    "trado": {"aime26": 2000, "humaneval": 2000},
}
CALIBRATION = {"aime26": "calibration_math", "humaneval": "calibration_code_contextmatched001", "ruler128k": "calibration_ruler"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_cell(harness, row, cap, seed):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = harness.generate(row["prompt"], cap, seed, answer_prefix=row.get("answer_prefix", ""))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    counts = result["counts"]
    pub = {"id": row["id"], "seed": seed, "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
           "prompt_tokens": result["prompt_tokens"], "output_tokens": len(result["tokens"]),
           "output_sha256": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest(),
           "finished": result["finished"], "generation_cap": cap,
           "counts": counts, "instrumented_seconds": elapsed,
           "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    priv = dict(pub, text=result["text"], tokens=result["tokens"])
    return pub, priv


def calibration(h, root, suite, attempt, bracket=(1e-6, 1.), manifest=None):
    leaf = root / "calibration" / h.key / suite / attempt
    leaf.mkdir(parents=True, exist_ok=False)
    data_path = Path(manifest) if manifest else root / "private/data" / h.key / (CALIBRATION[suite] + ".json")
    rows = json.loads(data_path.read_text())
    cap = CAPS[h.key][suite]
    # Frozen logarithmic bisection, sparsity measurements only. Online reuse
    # makes the curve non-monotone: retain all candidates and choose closest.
    lo, hi = bracket
    immutable(leaf / "config.json", {"model": h.key, "suite": suite,
                                     "threshold_bracket": list(bracket), "candidates": 7,
                                     "generation_cap": cap,
                                     "source_sha256": sha(__file__), "data_sha256": sha(data_path)})
    candidates = []
    for index in range(7):
        threshold = (lo * hi) ** .5
        h.runtime.arm = "value_v2_gaussian32_reuse_r4"
        h.runtime.threshold = threshold
        kept = eligible = 0
        for row in rows:
            pub, priv = run_cell(h, row, cap, 142)
            pub.update(threshold=threshold, candidate=index)
            append(leaf / "records.jsonl", pub)
            append(leaf / "private/generation.private.jsonl", dict(priv, threshold=threshold, candidate=index))
            kept += pub["counts"].get("kept_tiles", 0)
            eligible += pub["counts"].get("eligible_tiles", 0)
        sparse = 1. - kept / eligible
        candidates.append({"threshold": threshold, "sparsity": sparse, "kept_tiles": kept, "eligible_tiles": eligible})
        print(json.dumps({"event": "calibration_candidate", "model": h.key, "suite": suite, **candidates[-1]}), flush=True)
        if .48 <= sparse <= .52:
            break
        if sparse < .5:
            lo = threshold
        else:
            hi = threshold
    selected = min(candidates, key=lambda v: abs(v["sparsity"] - .5))
    # Native repeat confirms the selected operating point without scoring.
    h.runtime.threshold = selected["threshold"]
    confirmation = []
    for row in rows:
        pub, priv = run_cell(h, row, cap, 142)
        append(leaf / "confirmation.jsonl", pub)
        append(leaf / "private/confirmation.private.jsonl", priv)
        confirmation.append(pub)
    eligible = sum(r["counts"]["eligible_tiles"] for r in confirmation)
    kept = sum(r["counts"]["kept_tiles"] for r in confirmation)
    sparsity = 1. - kept / eligible
    result = {"model": h.key, "suite": suite, "candidates": candidates,
              "selected_threshold": selected["threshold"], "confirmed_sparsity": sparsity,
              "target_range": [.48, .52], "in_range": .48 <= sparsity <= .52,
              "manifest_sha256": sha(data_path), "computation_only": True,
              "native_confirmation": True, "source_sha256": sha(__file__)}
    immutable(leaf / "selected_threshold.json", result)
    immutable(leaf / "final_complete.json", {"status": "complete", "in_range": result["in_range"]})
    return result


def benchmark(h, root, suite, arm, shard, shards, attempt, threshold):
    leaf = root / "runs" / h.key / suite / arm / f"shard{shard:03}" / attempt
    leaf.mkdir(parents=True, exist_ok=False)
    data_path = root / "private/data" / h.key / f"{suite}.json"
    rows = json.loads(data_path.read_text())
    cells = [(r, seed) for r in rows for seed in r["seeds"]]
    cells = [cell for i, cell in enumerate(cells) if i % shards == shard - 1]
    h.runtime.arm, h.runtime.threshold = arm, threshold
    identity = {"model": h.key, "suite": suite, "arm": arm, "shard": shard,
                "statistics_query_blocks": h.runtime.statistics_query_blocks,
                "shards": shards, "cells": len(cells), "threshold": threshold,
                "manifest_sha256": sha(data_path), "source_sha256": sha(__file__),
                "protocol_sha256": sha(root / "protocol.json"),
                "attention_source_sha256": sha(__import__('experiments.numerical_qk_reuse.multi_model_attention', fromlist=['x']).__file__),
                "model_revision": json.loads((root / f"{h.key}_model_identity.json").read_text())["revision"],
                "runtime_sha256": sha(__import__('experiments.numerical_qk_reuse.multi_model_runtime', fromlist=['x']).__file__),
                "host_gpu_uuid": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()}
    immutable(leaf / "config.json", identity)
    started = time.time()
    for index, (row, seed) in enumerate(cells):
        pub, priv = run_cell(h, row, CAPS[h.key][suite], seed)
        pub.update(model=h.key, suite=suite, arm=arm, threshold=threshold,
                   manifest_sha256=identity["manifest_sha256"], source_sha256=identity["source_sha256"],
                   protocol_sha256=identity["protocol_sha256"], runtime_sha256=identity["runtime_sha256"],
                   model_revision=identity["model_revision"])
        append(leaf / "private/generation.private.jsonl", dict(priv, **{k: pub[k] for k in ("model", "suite", "arm")}))
        append(leaf / "records.jsonl", pub)
        print(json.dumps({"event": "cell_complete", "model": h.key, "suite": suite,
                          "arm": arm, "shard": shard, "index": index + 1,
                          "cells": len(cells), "N": pub["counts"].get("denoising_forwards"),
                          "C": pub["counts"].get("canvases"), "T": pub["output_tokens"],
                          "seconds": pub["instrumented_seconds"]}), flush=True)
    immutable(leaf / "receipt.json", dict(identity, gpu_worker_seconds=time.time() - started,
                                           records_sha256=sha(leaf / "records.jsonl")))
    immutable(leaf / "final_complete.json", {"status": "complete", "cells": len(cells)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=tuple(CAPS))
    p.add_argument("suite", choices=("aime26", "humaneval", "ruler128k"))
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--attempt", default="attempt001")
    p.add_argument("--threshold", type=float, default=.02)
    p.add_argument("--threshold-low", type=float, default=1e-6)
    p.add_argument("--threshold-high", type=float, default=1.)
    p.add_argument("--calibration-manifest", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    active = {"pid": os.getpid(), "model": a.model,
              "phase": "calibration" if a.calibrate else "benchmark", "suite": a.suite,
              "attempt": a.attempt, "started": time.time()}
    (root / "private/active_worker.json").write_text(json.dumps(active) + "\n")
    h = ModelHarness(root, a.model).load()
    if a.calibrate:
        result = calibration(h, root, a.suite, a.attempt, (a.threshold_low, a.threshold_high), a.calibration_manifest)
        print(json.dumps(result), flush=True)
    else:
        if not a.arm:
            p.error("--arm required for benchmark")
        benchmark(h, root, a.suite, a.arm, a.shard, a.shards, a.attempt, a.threshold)


if __name__ == "__main__":
    main()
