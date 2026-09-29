"""v27: host (enqueue) cost vs GPU cost per attention call, FA4 (CuTe DSL) vs our Triton kernels, at the
GLOBAL decode geometry. In a model forward, host cost is hidden only while the CPU runs ahead of the GPU;
after any synchronization it adds directly. Reports, per implementation:
  gpu_ms       -- CUDA-event time of one call with the GPU otherwise idle
  enqueue_ms   -- CPU time to issue one call (perf_counter, no sync), median over back-to-back calls
  stream_ms    -- per-call time of 50 back-to-back calls (throughput), then synchronize
usage: python -m scripts.v27_launch_overhead_bench OUT.json   (needs V27_FA4_OVERLAY)
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

import torch


def measure(fn, n=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gpu = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        gpu.append(a.elapsed_time(b))
    enq = []
    torch.cuda.synchronize()
    for _ in range(n):
        t = time.perf_counter(); fn(); enq.append((time.perf_counter() - t) * 1e3)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    stream = (time.perf_counter() - t) * 1e3 / n
    return dict(gpu_ms=round(statistics.median(gpu), 4), enqueue_ms=round(statistics.median(enq), 4),
                stream_ms=round(stream, 4))


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse import v27_fa4 as fa4
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64, dense64
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ, scale = 16, 2, 512, 256, 512 ** -.5
    rows = []
    for keys in [int(x) for x in os.environ.get('BENCH_KEYS', '32768,65536').split(',')]:
        q = torch.randn(1, NQ, H, D, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
        k = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        kt = math.ceil(keys / 64)
        row = dict(keys=keys)
        row['fa4_dense'] = measure(lambda: fa4.dense(q, k, v, scale))
        row['dense64'] = measure(lambda: dense64(q, k, v, scale, splits=2))
        for keep in (0.5, 0.1):
            kept = torch.rand(1, H, 2, kt, device='cuda', generator=g) < keep
            kept[..., 0] = True
            lists = fa4.block_sparse_tensors(kept)
            skipped = ~kept
            eligible = torch.ones_like(kept)
            row[f'fa4_sparse_cached_keep{keep}'] = measure(lambda lists=lists: fa4.sparse_lists(q, k, v, lists, scale))
            row[f'consume64_keep{keep}'] = measure(lambda skipped=skipped: consume64(q, k, v, skipped, eligible, scale,
                                                                                      splits=2))
            row[f'fa4_list_build_keep{keep}'] = measure(lambda kept=kept: fa4.block_sparse_tensors(kept))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_launch_overhead_v1', gpu=torch.cuda.get_device_name(0), rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
