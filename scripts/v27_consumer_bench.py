"""v27 consumer microbenchmark at the GLOBAL geometry (synthetic tensors, one layer).

Compares dense SDPA (native enable_gqa and repeated-KV), the production pre-QK
consumer and the experimental 64-row / split-KV consumer at several kept fractions
and key extents, and checks the experimental output against a masked dense
reference. usage: python -m scripts.v27_consumer_bench OUT.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import torch


def timed(fn, reps=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)


def reference(q, k, v, skipped, scale):
    h, hk = q.shape[1], k.shape[1]
    kr, vr = k.repeat_interleave(h // hk, 1).float(), v.repeat_interleave(h // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * scale
    keep = ~skipped.repeat_interleave(128, 2).repeat_interleave(64, 3)[:, :, :q.shape[2], :k.shape[2]]
    s = s.masked_fill(~keep, float('-inf'))
    return torch.einsum('bhqk,bhkd->bhqd', s.softmax(-1), vr).transpose(1, 2)


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device='cuda').manual_seed(0)
    rows = []
    for keys in (8192, 17536):
        h, hk, d, nq = 16, 2, 512, 256
        q = torch.randn(1, h, nq, d, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
        scale = d ** -.5
        rep = lambda x: x.repeat_interleave(h // hk, 1)
        row = dict(keys=keys)
        row['sdpa_enable_gqa'] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True))
        row['sdpa_repeat_kv'] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q, rep(k), rep(v), scale=scale))
        qb, kt = math.ceil(nq / 128), math.ceil(keys / 64)
        elig = torch.ones((1, h, qb, kt), dtype=torch.bool, device='cuda')
        for frac in (1.0, 0.5, 0.3, 0.1):
            skipped = torch.rand((1, h, qb, kt), device='cuda', generator=g) >= frac
            if frac >= 1:
                skipped.zero_()
            skipped[..., 0] = False
            row[f'prod_k{frac}'] = timed(lambda: preqk_attention(
                q, k, v, skipped, elig, scale=scale, is_causal=False, window=None, variant='generic',
                output_score_precision='fp32_scores_bf16_pv', output_layout='model_major'))
            for bm, sp, nw in ((64, 1, 8), (64, 2, 8), (64, 4, 8), (64, 8, 8), (32, 4, 4), (32, 8, 4)):
                name = f'c{bm}_s{sp}_w{nw}_k{frac}'
                try:
                    row[name] = timed(lambda: consume64(q, k, v, skipped, elig, scale, block_m=bm, splits=sp, num_warps=nw))
                except Exception as exc:  # resource limits for a config are recorded, not fatal
                    row[name] = f'error: {type(exc).__name__}: {str(exc)[:120]}'
            if frac == 0.3:
                ref = reference(q, k, v, skipped, scale)
                got = consume64(q, k, v, skipped, elig, scale, block_m=64, splits=4).float()
                row['c64_s4_max_abs_err_vs_ref'] = float((got - ref).abs().max())
        rows.append(row)
        print({k2: (round(v2, 3) if isinstance(v2, float) else v2) for k2, v2 in row.items()}, flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_consumer_bench_v1', gpu=torch.cuda.get_device_name(0), rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
