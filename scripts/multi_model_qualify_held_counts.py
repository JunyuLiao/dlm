"""Exact old/new route and counter parity for cached held-map accounting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from experiments.numerical_qk_reuse.multi_model_attention import AttentionRuntime, ARMS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--attempt", default="attempt006")
    a = p.parse_args()
    root = a.root.resolve()
    leaf = root / "qualification/held_counts" / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    path = root / "private/execution/8c0d14b031636ddcc708df75fa1d57aefa8b0ba4/experiments/numerical_qk_reuse/multi_model_attention.py"
    spec = importlib.util.spec_from_file_location("experiments.numerical_qk_reuse.multi_model_attention_before_counts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.time()
    rows = []
    try:
        torch.manual_seed(8951)
        for arm in ARMS[2:]:
            old = module.AttentionRuntime(arm, .15, statistics_query_blocks=8)
            new = AttentionRuntime(arm, .15, statistics_query_blocks=8)
            for runtime in (old, new):
                runtime.block(448, 513)
            for step in range(7):
                q = torch.randn((1, 4, 385, 128), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((1, 2, 513, 128), device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                if step == 2:
                    old.invalidate()
                    new.invalidate()
                for runtime in (old, new):
                    runtime.begin_forward()
                reference = old.attention(q, k, v, 2, 128 ** -.5)
                candidate = new.attention(q, k, v, 2, 128 ** -.5)
                assert torch.equal(reference, candidate), (arm, step, "consumer output")
                assert torch.equal(old.maps[(2, 385, 513)][0], new.maps[(2, 385, 513)][0]), (arm, step, "map")
                assert old.counts == new.counts, (arm, step, "counters")
                rows.append({"arm": arm, "step": step, "output_equal": True,
                             "map_equal": True, "counters_equal": True})
            new.reset()
            assert not new.maps and not new.map_kept_counts
            new.block(0, 4)
            assert not new.maps and not new.map_kept_counts
        from experiments.numerical_qk_reuse.multi_model_runtime import ModelHarness, import_file
        h = ModelHarness(root, "ultrallada").load(install=False)
        prompt = "Write a short Python function that adds one to its integer input."
        sampler = import_file(root / "private/source/ultra_llada_generate.py", "ultra_two_block_publisher_qualification")
        ids = torch.tensor([h.encode(prompt)], device="cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        native = sampler.generate(h.model, ids, steps=128, gen_length=128,
                                  block_length=64, temperature=0., cfg_scale=0.,
                                  remasking="low_confidence", mask_id=126336)[0, ids.shape[1]:].tolist()
        eos = h.tokenizer.eos_token_id
        native = native[:native.index(eos) + 1] if eos in native else native
        h.install()
        for arm in ARMS:
            old = module.AttentionRuntime(arm, .15, statistics_query_blocks=8)
            new = AttentionRuntime(arm, .15, statistics_query_blocks=8)
            h.runtime = old
            reference = h.generate(prompt, 128, 42)
            h.runtime = new
            candidate = h.generate(prompt, 128, 42)
            assert reference["tokens"] == candidate["tokens"], (arm, "model output")
            assert reference["counts"] == candidate["counts"], (arm, "model counters")
            if arm == "dense_native" and candidate["tokens"] != native:
                raise AssertionError("mandatory native publisher two-block output mismatch")
            rows.append({"arm": arm, "model": "ultrallada", "generation_cap": 128,
                         "blocks": candidate["counts"]["canvases"], "output_equal": True,
                         "publisher_two_block_output_equal": True if arm == "dense_native" else None,
                         "counters_equal": True,
                         "output_sha256": hashlib.sha256(json.dumps(candidate["tokens"]).encode()).hexdigest()})
        result = {"status": "pass", "cases": rows, "reference_attention_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "candidate_attention_sha256": hashlib.sha256(Path(__import__("experiments.numerical_qk_reuse.multi_model_attention", fromlist=["x"]).__file__).read_bytes()).hexdigest(),
                  "gpu_worker_seconds": time.time() - started,
                  "synthetic_prompts_only": True, "selector_and_sampler_changed": False}
        (leaf / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    except BaseException as error:
        (leaf / "failure.json").write_text(json.dumps({"error": repr(error),
            "mandatory_native_publisher_mismatch": "mandatory native publisher" in str(error),
            "gpu_worker_seconds": time.time() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
