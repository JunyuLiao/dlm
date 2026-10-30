"""Same-state fresh/held support diagnostics on exposed development prompts.

Observers add attention-only work, never model forwards or model-output edits.
This is not benchmark accuracy or convergence evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.numerical_qk_reuse.multi_model_attention import AttentionRuntime
from experiments.numerical_qk_reuse.multi_model_runtime import ModelHarness
from experiments.numerical_qk_reuse import v27_fa4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=("llada22", "ultrallada", "trado"))
    p.add_argument("--attempt", default="attempt001")
    a = p.parse_args()
    root = a.root.resolve()
    leaf = root / "reuse_diagnostic" / a.model / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    selected = sorted((root / "calibration" / a.model / "humaneval").glob("attempt*/selected_threshold.json"))
    threshold = json.loads(selected[-1].read_text())["selected_threshold"]
    h = ModelHarness(root, a.model, arm="dense_native", threshold=threshold).load()
    original = h.runtime.attention
    shadows = {name: AttentionRuntime(name, threshold) for name in
               ("mass_topk_reuse_r4", "value_v2_gaussian32_reuse_r4", "value_v3a_singleton_reuse_r4")}
    held, records = {}, []
    request_id = None
    cap_per_request = 8
    sampled = [0]
    extra = {"attention_calls": 0, "discovery_qk_pairs": 0, "model_forwards": 0}

    def observe(q, k, v, layer, scale, **kwargs):
        out = original(q, k, v, layer, scale, **kwargs)
        target_layer = {"llada22": 19, "ultrallada": 31, "trado": 35}[a.model]
        if h.runtime.phase != "denoise" or layer != target_layer or sampled[0] >= cap_per_request:
            return out
        age = h.runtime.call_in_block - 1
        if age > 3:
            return out
        sampled[0] += 1
        qb = h.runtime.block_start // 128 if a.model == "ultrallada" else 0
        part = q[:, :, qb * 128:(qb + 1) * 128]
        dense = v27_fa4.dense_plain(part, k, v, scale)
        extra["attention_calls"] += 1
        for name, shadow in shadows.items():
            shadow.reset()
            shadow.block(h.runtime.block_start, h.runtime.block_end)
            shadow.begin_forward()
            fresh = shadow.attention(part, k, v, layer, scale)
            keep, lists = shadow.maps[(layer, part.shape[2], k.shape[2])]
            key = (request_id, h.runtime.block_start, name)
            if key not in held:
                held[key] = (keep.clone(), lists)
            old_keep, old_lists = held[key]
            carried = v27_fa4.sparse_lists(part, k, v, old_lists, scale)
            extra["attention_calls"] += 2
            extra["discovery_qk_pairs"] += shadow.counts["discovery_qk_pairs"]
            denom = dense.float().norm(dim=-1).clamp_min(1e-8)
            fresh_error = (fresh.float() - dense.float()).norm(dim=-1) / denom
            held_error = (carried.float() - dense.float()).norm(dim=-1) / denom
            inter = int((keep & old_keep).sum())
            union = int((keep | old_keep).sum())
            record = {"id": request_id, "model": a.model, "selector": name,
                      "block_start": h.runtime.block_start, "age": age,
                      "query_rows": part.shape[2], "context_tokens": k.shape[2],
                      "fresh_held_iou": inter / union if union else 1.,
                      "fresh_kept_tiles": int(keep.sum()), "held_kept_tiles": int(old_keep.sum()),
                      "fresh_mean_relative_l2": float(fresh_error.mean()),
                      "held_mean_relative_l2": float(held_error.mean()),
                      "fresh_max_relative_l2": float(fresh_error.max()),
                      "held_max_relative_l2": float(held_error.max())}
            records.append(record)
            with (leaf / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return out

    h.runtime.attention = observe
    path = root / "private/data" / a.model / "calibration_code_contextmatched001.json"
    rows = json.loads(path.read_text())
    started = time.time()
    requests = []
    for row in rows:
        request_id = row["id"]
        sampled[0] = 0
        result = h.generate(row["prompt"], 128, 142)
        requests.append({"id": request_id, "counts": result["counts"],
                         "output_sha256": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest(),
                         "sampled_states": sampled[0]})
    receipt = {"status": "complete_development_diagnostic", "model": a.model,
               "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "threshold": threshold, "requests": requests, "measurements": len(records),
               "extra_observer_work": extra, "gpu_worker_seconds": time.time() - started,
               "model_output_intervention": False, "quality_scored": False}
    (leaf / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (leaf / "final_complete.json").write_text(json.dumps({"status": "complete", "scope": "development_only"}) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
