"""v27: cost of the fused observation call (dense output + M1 prefix summaries + observation tail) at the GLOBAL
decode geometry, by split-KV count and with/without the rank-32 mu (M2c pooled_compact has none), against FA4
dense (all-kept path) and the 64-row dense kernel. CUDA-event median of 10 after 2 warm-ups, ms.
usage: python -m scripts.v27_observe_bench OUT.json   (V27_FA4_OVERLAY set)
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import torch


def timed(fn, n=10):
    for _ in range(2):
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
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64, fused_observe
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    rows = []
    for keys in (32389, 60151):
        q = (torch.randn(1, NQ, H, D, device='cuda', dtype=torch.bfloat16, generator=g) * D ** -.25).transpose(1, 2)
        k = (torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g) * D ** -.25)
        v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        sketch = torch.randn(1, HK, keys, 32, device='cuda', dtype=torch.float32, generator=g)
        prefix_tiles = (keys - NQ) // 64
        row = dict(keys=keys, prefix_tiles=prefix_tiles,
                   fa4_dense_allkept=timed(lambda: v27_fa4.dense(q, k, v, 1.0)),
                   dense64_s2=timed(lambda: dense64(q, k, v, 1.0, splits=2)))
        for mu in (True, False):
            summary = allocate_summary(1, H, math.ceil(NQ / 128), math.ceil(keys / 64), prefix_tiles,
                                       32 if mu else 0, q.device, None)
            for splits in (1, 2, 4, 8):
                try:
                    row[f'observe_mu{int(mu)}_s{splits}'] = timed(lambda: fused_observe(
                        q, k, v, sketch, 1.0, prefix_tiles, summary, splits=splits, mu=mu, mu_precision='bf16'))
                except Exception as exc:
                    row[f'observe_mu{int(mu)}_s{splits}'] = f'{type(exc).__name__}: {str(exc)[:160]}'
            del summary
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_observe_bench_v1', torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
