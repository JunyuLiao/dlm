"""v27: the official SOTA dense kernel for DiffusionGemma's 512-dim GLOBAL layers -- FlashAttention-4
(CuTe-DSL) from vLLM's flash-attention fork (SM90 accepts head_dim <= 512; the DiffusionGemma technical
report serves the canvas with FA4) -- and its official block-sparse interface, at our decode geometry.

Per key extent: FA4 dense vs our dense64; FA4 block-sparse with a Q128 x KV64 keep map (the M1/M2/M3
tile geometry) at several keep fractions, each checked against a masked FP32 reference. Runs in the
pinned environment with the vendored overlay (V27_FA4_OVERLAY). usage: python -m scripts.v27_fa4_bench OUT.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch


def timed(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)


def keep_map(h, qb, kt, keep, g):
    """Random Q128 x KV64 keep map with the first tile always kept (first support)."""
    kept = torch.rand((1, h, qb, kt), device='cuda', generator=g) < keep
    kept[..., 0] = True
    return kept


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse import v27_fa4 as fa4
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64, dense64
    fwd = fa4.load()
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    scale = D ** -.5
    rows = []
    for keys in [int(x) for x in os.environ.get('BENCH_KEYS', '16640,32768,65536').split(',')]:
        q = torch.randn(1, H, NQ, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))      # [1, S, heads, D]
        kt, qb = math.ceil(keys / 64), math.ceil(NQ / 128)
        flops = 4.0 * H * NQ * keys * D
        row = dict(keys=keys, gflop=round(flops / 1e9, 1), impl={})
        kr, vr = k[:, :1].float().repeat_interleave(8, 1), v[:, :1].float().repeat_interleave(8, 1)

        def reference(kept=None):   # heads 0..7 (KV head 0), FP32
            s = torch.einsum('bhqd,bhkd->bhqk', q[:, :8].float(), kr) * scale
            if kept is not None:
                m = kept[:, :8].repeat_interleave(128, 2).repeat_interleave(64, 3)[:, :, :NQ, :keys]
                s = s.masked_fill(~m, float('-inf'))
            return torch.einsum('bhqk,bhkd->bhqd', s.softmax(-1), vr).transpose(1, 2)

        def record(name, fn, want, keep=None):
            try:
                out = fn()
                err = (out[:, :, :8].float() - want).abs().max().item()
                ms = timed(fn)
                row['impl'][name] = dict(ms=round(ms, 4), tflops_dense_equiv=round(flops / ms / 1e9, 1),
                                         max_abs_err=round(err, 5), keep=keep)
            except Exception as exc:
                row['impl'][name] = dict(error=f'{type(exc).__name__}: {exc}'[:300], keep=keep)
            torch.cuda.empty_cache()

        dense_ref = reference()
        record('fa4_dense_contiguous', lambda: fwd(qs, ks, vs, softmax_scale=scale, causal=False)[0], dense_ref)
        record('fa4_dense_model_views', lambda: fa4.dense(q, k, v, scale), dense_ref)
        record('dense64_s2', lambda: dense64(q, k, v, scale, splits=2), dense_ref)
        for keep in (1.0, 0.5, 0.25, 0.1):
            kept = keep_map(H, qb, kt, keep, g)
            want = reference(kept)
            tensors = fa4.block_sparse_tensors(kept)
            record(f'fa4_block_sparse_keep{keep}',
                   lambda tensors=tensors: fwd(qs, ks, vs, softmax_scale=scale, causal=False,
                                               block_sparse_tensors=tensors)[0], want, keep)
            skipped = ~kept
            record(f'fa4_sparse_model_views_keep{keep}',
                   lambda skipped=skipped: fa4.sparse(q, k, v, skipped, torch.ones_like(skipped), scale), want, keep)
            record(f'consume64_s2_keep{keep}',
                   lambda skipped=skipped: consume64(q, k, v, skipped, torch.ones_like(skipped), scale, splits=2),
                   want, keep)
        rows.append(row)
        print(json.dumps(dict(keys=keys, **{n: (x.get('ms'), x.get('max_abs_err')) if 'ms' in x else x['error'][:120]
                                            for n, x in row['impl'].items()})), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_fa4_bench_v1', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       shape=dict(H=H, HK=HK, D=D, NQ=NQ, bidirectional=True, sparse_block=(128, 64)), rows=rows), f,
                  indent=1)


if __name__ == '__main__':
    main()
