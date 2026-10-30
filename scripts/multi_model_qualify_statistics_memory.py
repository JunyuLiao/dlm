"""Bounded long-context statistics batching under simulated model residency."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

import experiments.numerical_qk_reuse.multi_model_attention as candidate_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--attempt", default="attempt007")
    a = p.parse_args()
    root = a.root.resolve()
    leaf = root / "qualification/statistics_memory" / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    started = time.time()
    path = root / "private/execution/8c0d14b031636ddcc708df75fa1d57aefa8b0ba4/experiments/numerical_qk_reuse/multi_model_attention.py"
    spec = importlib.util.spec_from_file_location("experiments.numerical_qk_reuse.memory_reference", path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    rows = []
    try:
        torch.manual_seed(18493)
        # Reserve room for model weights, retained maps and full-sequence states.
        guard_bytes = 30 * 1024 ** 3
        guard = torch.empty((guard_bytes,), dtype=torch.uint8, device="cuda")
        q = torch.randn((1, 32, 513, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, 32, 131071, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        old_statistics, new_statistics = reference_module.statistics_cuda, candidate_module.statistics_cuda
        calls = {"reference": 0, "candidate": 0}
        def old_hook(*args, **kwargs):
            calls["reference"] += 1
            return old_statistics(*args, **kwargs)
        def new_hook(*args, **kwargs):
            calls["candidate"] += 1
            return new_statistics(*args, **kwargs)
        reference_module.statistics_cuda, candidate_module.statistics_cuda = old_hook, new_hook
        torch.cuda.reset_peak_memory_stats()
        for arm in candidate_module.ARMS[2:]:
            calls.update(reference=0, candidate=0)
            old = reference_module.AttentionRuntime(arm, .0001, statistics_query_blocks=8)
            new = candidate_module.AttentionRuntime(arm, .0001, statistics_query_blocks=8,
                                                    statistics_mean_elements_cap=1073741824)
            for runtime in (old, new):
                runtime.block(131007, 131071)
                runtime.begin_forward()
            reference = old.attention(q, k, v, 3, 128 ** -.5)
            candidate = new.attention(q, k, v, 3, 128 ** -.5)
            assert torch.equal(reference, candidate), (arm, "consumer output")
            assert torch.equal(old.maps[(3, 513, 131071)][0], new.maps[(3, 513, 131071)][0]), (arm, "map")
            assert old.counts == new.counts, (arm, "counters")
            assert calls == {"reference": 5, "candidate": 2}, calls
            rows.append({"arm": arm, "q": 513, "kv": 131071, "heads": 32,
                         "head_dim": 128, "reference_statistics_batches": calls["reference"],
                         "candidate_statistics_batches": calls["candidate"],
                         "mask_mismatches": 0, "consumer_output_equal": True, "counters_equal": True})
        torch.cuda.synchronize()
        result = {"status": "pass", "cases": rows, "resident_memory_guard_bytes": guard_bytes,
                  "peak_allocated_bytes_including_guard": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes_including_guard": torch.cuda.max_memory_reserved(),
                  "reference_mean_elements_cap": 134217728, "candidate_mean_elements_cap": 1073741824,
                  "reference_attention_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "candidate_attention_sha256": hashlib.sha256(Path(candidate_module.__file__).read_bytes()).hexdigest(),
                  "gpu_worker_seconds": time.time() - started, "synthetic_attention_only": True,
                  "model_forwards": 0, "clean_latency_claim": False}
        (leaf / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    except BaseException as error:
        (leaf / "failure.json").write_text(json.dumps({"error": repr(error),
            "gpu_worker_seconds": time.time() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
