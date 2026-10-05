"""v27: FA4 dense vs the same kernel through its block-sparse interface with every tile kept ("all-kept"), at the
model's real key counts (not multiples of 64, so the last KV tile is partial) and the model's [1, heads, len, D]
views -- the exact v27_fa4 entry points. Correctness vs an FP32 reference, then timing (CUDA-event median of 20).
usage: python v27_fa4_allkept_check.py OUT.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import torch


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return round(statistics.median(out), 4)


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    scale = 1.0   # the model's GLOBAL layers use scaling 1.0 (q/k RMS-normed)
    rows = []
    for keys in (17284, 32389, 60151, 60407):
        q = (torch.randn(1, NQ, H, D, device='cuda', dtype=torch.bfloat16, generator=g) * D ** -.25).transpose(1, 2)
        k = (torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g) * D ** -.25).transpose(1, 2)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
        rep = H // HK
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1),
            scale=scale).transpose(1, 2)
        kt = math.ceil(keys / 64)
        allkept = v27_fa4.block_sparse_tensors(torch.ones(1, H, 2, kt, dtype=torch.bool, device='cuda'))
        dense_out = v27_fa4.dense(q, k, v, scale)
        sparse_out = v27_fa4.sparse_lists(q, k, v, allkept, scale)
        row = dict(keys=keys, partial_last_tile=keys % 64,
                   dense_err=round(float((dense_out.float() - ref).abs().max()), 5),
                   allkept_err=round(float((sparse_out.float() - ref).abs().max()), 5),
                   dense_vs_allkept_max_abs=round(float((dense_out.float() - sparse_out.float()).abs().max()), 6),
                   dense_ms=timed(lambda: v27_fa4.dense(q, k, v, scale)),
                   allkept_ms=timed(lambda: v27_fa4.sparse_lists(q, k, v, allkept, scale)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_fa4_allkept_check_v1', torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
