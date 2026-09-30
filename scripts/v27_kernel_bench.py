"""v27: GLOBAL-attention kernel latency at DiffusionGemma's decode geometry, the way sparse-attention papers report it
(kernel latency and speedup over dense FlashAttention across context lengths and sparsity levels).

Geometry of one GLOBAL decoder layer call: batch 1, 16 query heads over the 256 canvas queries, 2 KV heads,
head_dim 512, bf16, keys = context + 256 canvas keys, bidirectional (no mask). Per context length:
  * dense references: FA4 plain dense (official FlashAttention-4), FA4 block-sparse API with every tile kept (our
    dense reference, bitwise equal to FA4 plain), PyTorch SDPA with enable_gqa (the HF model's default path);
  * FA4 block-sparse at kept fractions of the context tiles (Q128 x KV64 tiles; the canvas tiles are always kept),
    uniform random per (head, query block) as in kernel benchmarks of FlexAttention / SpargeAttn-style papers. The
    FA4 block lists are built once per map (a held map is reused across calls) and timed separately.
Timing: CUDA events, warm-up, median of repeated launches, no L2 flush (the same for every kernel).
usage: python -m scripts.v27_kernel_bench [--lengths 8192,16384,...] [--keep 0.05,0.1,...] [--reps 50]
"""
from __future__ import annotations

import argparse
import json
import statistics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', default='8192,16384,32768,65536,131072')
    p.add_argument('--keep', default='0.05,0.1,0.2,0.3,0.5,1.0')
    p.add_argument('--reps', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    import torch
    from experiments.numerical_qk_reuse import v27_fa4
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda')
    heads, kv_heads, dim, nq = 16, 2, 512, 256
    scale = dim ** -0.5

    def timed(fn):
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(a.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            out.append(s.elapsed_time(e))
        return round(statistics.median(out), 4)
    print(json.dumps(dict(schema='v27_kernel_bench_v1', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                          geometry=dict(q_heads=heads, kv_heads=kv_heads, head_dim=dim, queries=nq))), flush=True)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    for length in [int(x) for x in a.lengths.split(',')]:
        nk = length + nq
        q = torch.randn((1, heads, nq, dim), device=dev, dtype=torch.bfloat16, generator=gen)
        k = torch.randn((1, kv_heads, nk, dim), device=dev, dtype=torch.bfloat16, generator=gen)
        v = torch.randn((1, kv_heads, nk, dim), device=dev, dtype=torch.bfloat16, generator=gen)
        row = dict(context=length, keys=nk)
        reference = v27_fa4.dense_plain(q, k, v, scale)
        row['fa4_dense_plain_ms'] = timed(lambda: v27_fa4.dense_plain(q, k, v, scale))
        row['fa4_allkept_ms'] = timed(lambda: v27_fa4.dense(q, k, v, scale))
        row['fa4_allkept_equals_plain'] = bool(torch.equal(v27_fa4.dense(q, k, v, scale), reference))
        try:
            sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
            row['sdpa_ms'] = timed(sdpa)
        except Exception as exc:                  # noqa: BLE001 -- report, do not hide
            row['sdpa_error'] = type(exc).__name__
        qb, kt = -(-nq // 128), -(-nk // 64)
        canvas_tiles = -(-nq // 64)
        context_tiles = kt - canvas_tiles
        sparse = {}
        for keep in [float(x) for x in a.keep.split(',')]:
            kept = torch.zeros((1, heads, qb, kt), device=dev, dtype=torch.bool)
            kept[..., context_tiles:] = True
            n = max(0, min(context_tiles, round(keep * context_tiles)))
            if n:
                order = torch.rand((1, heads, qb, context_tiles), device=dev, generator=gen).argsort(-1)[..., :n]
                kept[..., :context_tiles].scatter_(-1, order, True)
            build_ms = timed(lambda: v27_fa4.block_sparse_tensors(kept))
            lists = v27_fa4.block_sparse_tensors(kept)
            ms = timed(lambda: v27_fa4.sparse_lists(q, k, v, lists, scale))
            sparse[str(keep)] = dict(kept_tiles=round(float(kept.float().mean()), 4), ms=ms,
                                     speedup_vs_fa4_allkept=round(row['fa4_allkept_ms'] / ms, 3),
                                     speedup_vs_fa4_plain=round(row['fa4_dense_plain_ms'] / ms, 3),
                                     list_build_ms=build_ms)
            if keep >= 1.0:
                sparse[str(keep)]['equals_plain'] = bool(torch.equal(v27_fa4.sparse_lists(q, k, v, lists, scale),
                                                                     reference))
        row['fa4_block_sparse'] = sparse
        print(json.dumps(row), flush=True)
        del q, k, v, reference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
