"""Independent synthetic CUDA qualification; never runs benchmark prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.numerical_qk_reuse.multi_model_attention import (
    AttentionRuntime, ARMS, dense_block,
)
from experiments.numerical_qk_reuse.v31_value_selectors import (
    statistics_reference, online_reference, singleton_reference,
)
from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda, online_cuda


def kernels():
    torch.manual_seed(8291)
    records = []
    for nq, nk, h, hk in [(4, 197, 8, 2), (32, 193, 4, 1), (129, 259, 4, 4)]:
        q = torch.randn((1, h, nq, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, hk, nk, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(h // hk, 1),
                                             v.repeat_interleave(h // hk, 1), is_causal=False)
        for arm in ARMS:
            r = AttentionRuntime(arm, .02)
            r.block(max(0, nk - 32), nk)
            r.begin_forward()
            out = r.attention(q, k, v, 0, 128 ** -.5).transpose(1, 2)
            if arm in ARMS[:2]:
                error = float((out - ref).abs().max())
                assert error <= .04, (arm, nq, nk, error)
            else:
                keep, _ = r.maps[(0, nq, nk)]
                bias = keep.repeat_interleave(128, 2).repeat_interleave(64, 3)[:, :, :nq, :nk]
                expected = F.scaled_dot_product_attention(q, k.repeat_interleave(h // hk, 1),
                                                          v.repeat_interleave(h // hk, 1), attn_mask=bias)
                error = float((out - expected).abs().max())
                assert error <= .04, (arm, nq, nk, error)
            records.append({"arm": arm, "q": nq, "kv": nk, "heads": h,
                            "kv_heads": hk, "max_abs_error": error,
                            "counts": dict(r.counts)})
    # Test true block-causal prefix visibility, with offset and partial tile.
    q = torch.randn((1, 4, 67, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 131, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for block in (4, 32):
        bias = torch.arange(131, device="cuda")[None] // block <= (torch.arange(67, device="cuda")[:, None] + 64) // block
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=bias)
        out = dense_block(q, k, v, 64, block, 128 ** -.5).transpose(1, 2)
        error = float((out - ref).abs().max())
        assert error <= .04, ("block", block, error)
        records.append({"block_size": block, "max_abs_error": error})
    torch.cuda.synchronize()
    # Independent score/mask oracle, beyond consumer-output parity.
    from experiments.numerical_qk_reuse.multi_model_attention import budget_map, singleton_scores
    from experiments.numerical_qk_reuse.v31_value_selectors import full_support, deletion_scores
    q = torch.randn((1, 4, 37, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 511, 128), device="cuda", dtype=torch.bfloat16)
    z = torch.randn((1, 2, 511, 32), device="cuda", dtype=torch.float32)
    nu = torch.ones((2,), device="cuda")
    scores = torch.matmul(q[0].float(), k[0].float().repeat_interleave(2, 0).transpose(1, 2)) * 128 ** -.5
    cpu = statistics_reference(scores.cpu(), z[0].cpu(), nu.cpu(), 8)
    gpu = statistics_cuda(q, k, z, nu, 128 ** -.5, 8)
    mandatory = torch.zeros((4, 8), dtype=torch.bool, device="cuda")
    mandatory[:, -1] = True
    for threshold in (.01, .1, .5):
        expected = online_reference(cpu, threshold, preserve_mass=True, mandatory=mandatory.cpu())[0]
        actual = online_cuda(gpu, threshold, True, mandatory, torch.zeros_like(mandatory), 0.)[0]
        assert torch.equal(actual.cpu(), expected), ("V2 oracle", threshold)
    alpha, contribution, output, g = full_support(cpu)
    expected_score = deletion_scores(cpu, alpha, g, alpha.sum(1), g.sum(1))
    expected = budget_map(expected_score, mandatory.cpu(), cpu.valid.any(-1))
    actual = budget_map(singleton_scores(gpu), mandatory, gpu.valid.any(-1))
    assert torch.equal(actual.cpu(), expected), "V3a singleton oracle mismatch"
    records.append({"selector_oracles": "V2 ordered recurrence and V3a singleton",
                    "thresholds": [.01, .1, .5], "mask_mismatches": 0})
    q = torch.randn((1, 4, 385, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 513, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for arm in ARMS[2:]:
        reference = AttentionRuntime(arm, .15, statistics_query_blocks=1)
        candidate = AttentionRuntime(arm, .15, statistics_query_blocks=8)
        for runtime in (reference, candidate):
            runtime.block(448, 513)
            runtime.begin_forward()
        a = reference.attention(q, k, v, 2, 128 ** -.5)
        b = candidate.attention(q, k, v, 2, 128 ** -.5)
        assert torch.equal(reference.maps[(2, 385, 513)][0], candidate.maps[(2, 385, 513)][0]), (arm, "batch masks")
        assert torch.equal(a, b), (arm, "batch output")
        records.append({"statistics_batch_selector": arm, "query_rows": 385,
                        "query_blocks_reference": 1, "query_blocks_candidate": 8,
                        "mask_mismatches": 0, "output_equal": True})
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--model", choices=("llada22", "ultrallada", "trado"))
    p.add_argument("--attempt", default="attempt001")
    p.add_argument("--long-prompt", action="store_true")
    a = p.parse_args()
    leaf = a.root / "qualification" / (a.model or "kernels") / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        if a.model:
            from experiments.numerical_qk_reuse.multi_model_runtime import ModelHarness
            h = ModelHarness(a.root, a.model).load(install=False)
            prompt = ("A synthetic context follows. " + "The blue square is beside the green circle. " * 36
                      + " What is 2 plus 3? Answer briefly.") if a.long_prompt else "What is 2 plus 3? Answer briefly."
            native = None
            native_calls = [0]
            if a.model == "llada22":
                ids = torch.tensor([h.encode(prompt)], device="cuda")
                original_forward = h.model.forward
                def counted_forward(*args, **kwargs):
                    native_calls[0] += 1
                    return original_forward(*args, **kwargs)
                h.model.forward = counted_forward
                torch.manual_seed(42)
                native = h.model.generate(inputs=ids, gen_length=32, block_length=32,
                                          steps=32, threshold=.5, editing_threshold=0.,
                                          max_post_steps=16, temperature=0., eos_early_stop=True)[0].tolist()
                h.model.forward = original_forward
            elif a.model == "ultrallada":
                from experiments.numerical_qk_reuse.multi_model_runtime import import_file
                sampler = import_file(a.root / "private/source/ultra_llada_generate.py", "ultra_reference_qualification")
                ids = torch.tensor([h.encode(prompt)], device="cuda")
                torch.manual_seed(42)
                native = sampler.generate(h.model, ids, steps=32, gen_length=32,
                                          block_length=32, temperature=0., cfg_scale=0.,
                                          remasking="low_confidence", mask_id=126336)[0, ids.shape[1]:].tolist()
                native_calls[0] = 32
            elif a.model == "trado":
                from experiments.numerical_qk_reuse.multi_model_runtime import import_file
                sampler = import_file(a.root / "private/source/SDAR_generate.py", "trado_reference_qualification")
                ids = torch.tensor([h.encode(prompt)], device="cuda")
                torch.manual_seed(42)
                native = sampler.block_diffusion_generate(
                    h.model, {"input_ids": ids}, mask_id=151669, gen_length=16,
                    block_length=4, denoising_steps=4, temperature=1., top_k=0, top_p=1.,
                    confidence_threshold=.9, stopping_criteria_idx=[h.tokenizer.eos_token_id])[0, ids.shape[1]:ids.shape[1]+16].tolist()
            h.install()
            rows = []
            for arm in ARMS:
                h.runtime.arm = arm
                result = h.generate(prompt, 32 if a.model != "trado" else 16, 42)
                rows.append({"arm": arm, "output_tokens": len(result["tokens"]),
                             "finished": result["finished"], "counts": result["counts"]})
                if native is not None and arm == "dense_native":
                    eos = h.tokenizer.eos_token_id
                    native_trim = native[:native.index(eos) + 1] if eos in native else native
                    rows[-1]["publisher_output_equal"] = result["tokens"] == native_trim
                    rows[-1]["publisher_forward_count"] = native_calls[0]
                    if not rows[-1]["publisher_output_equal"]:
                        raise AssertionError("cached native dense differs from publisher sampler")
                print(json.dumps(rows[-1]), flush=True)
            result = {"status": "synthetic_model_smoke_only", "rows": rows,
                      "source_changes": h.source_changes}
        else:
            result = {"status": "pass", "cases": kernels()}
        result["gpu_worker_seconds"] = time.time() - started
        (leaf / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException as e:
        (leaf / "failure.json").write_text(json.dumps({"status": "failed", "error": repr(e),
                                                      "gpu_worker_seconds": time.time() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
