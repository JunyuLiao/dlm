"""v27 LOCAL-layer kernel feasibility probe (synthetic tensors; no model).

Question: at the actual LOCAL geometry (16 query heads, 8 KV heads, head_dim 256,
sliding window 1024, canvas 256 queries) can our pre-QK sparse consumer beat native
SDPA at all? Timed: native SDPA (repeat_kv, as the transformers sdpa path does, and
enable_gqa), our consumer with every tile kept, and with a random tile bitmap at
several kept fractions. The GLOBAL geometry (2 KV heads, head_dim 512) at a long key
extent is timed the same way as a reference. Kernel time depends on shapes and the
bitmap density, not on values; this is a microbenchmark, not a forward price.
usage: python -m scripts.v27_local_kernel_probe OUT.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import torch


def timed(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def probe(h, hk, d, keys, q_len=256, kept=(1.0, .7, .5, .3, .1), seed=0):
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(1, h, q_len, d, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
    scale = d ** -.5
    rep = lambda x: x.repeat_interleave(h // hk, dim=1)
    out = dict(heads=h, kv_heads=hk, head_dim=d, keys=keys, queries=q_len)
    out['native_sdpa_repeat_kv_ms'] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, rep(k), rep(v), scale=scale))
    out['native_sdpa_enable_gqa_ms'] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, enable_gqa=True))
    qb, kt = math.ceil(q_len / 128), math.ceil(keys / 64)
    eligible = torch.ones((1, h, qb, kt), dtype=torch.bool, device='cuda')
    for frac in kept:
        skipped = torch.rand((1, h, qb, kt), device='cuda', generator=g) >= frac
        if frac >= 1.0:
            skipped.zero_()
        out[f'consumer_kept_{frac:.1f}_ms'] = timed(lambda: preqk_attention(
            q, k, v, skipped, eligible, scale=scale, is_causal=False, window=None, variant='generic',
            output_score_precision='fp32_scores_bf16_pv', output_layout='model_major'))
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [probe(16, 8, 256, keys) for keys in (1024, 1280)]
    rows += [probe(16, 2, 512, keys) for keys in (4096, 17536)]
    report = dict(schema='v27_local_kernel_probe_v1', gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, rows=rows,
                  note='synthetic microbenchmark; LOCAL rows = 16q/8kv/d256 at window-sized key extents; '
                       'GLOBAL rows = 16q/2kv/d512 reference')
    with open(argv[0], 'x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == '__main__':
    main()
