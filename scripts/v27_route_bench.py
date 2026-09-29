"""v27 decision-call selector microbenchmark at the GLOBAL geometry (one layer, synthetic tensors).

Per key extent: dense64 (reference cost of the layer's attention), the generic summary-LOAD
route (the D-step selector today) and the pipelined LOAD route over stages/warps, for the
exact-mu summary, the compact M2 pool and the fused-observation tail. Also checks that the
pipelined decisions equal the generic ones on every configuration it times.
usage: python -m scripts.v27_route_bench OUT.json   (BENCH_KEYS=17536,32768,65536)
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


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only, tile_pool
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64, fused_observe
    g = torch.Generator(device='cuda').manual_seed(0)
    rows = []
    for keys in [int(x) for x in os.environ.get('BENCH_KEYS', '17536,32768,65536').split(',')]:
        h, hk, d, nq = 16, 2, 512, 256
        q = torch.randn(1, h, nq, d, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, hk, keys, d, device='cuda', dtype=torch.bfloat16, generator=g)
        z = torch.randn(1, hk, keys, 32, device='cuda', generator=g)
        ref = torch.rand(1, hk, device='cuda', generator=g) + .5
        sens = torch.rand(1, nq, device='cuda', generator=g) + .5
        scale = d ** -.5
        pt, qb, kt = (keys - nq) // 64, 2, math.ceil(keys / 64)
        row = dict(keys=keys, dense64=timed(lambda: dense64(q, k, v, scale, splits=2)))
        # exact-mu summary via the fused observation, then its tail
        summ = allocate_summary(1, h, qb, kt, pt, 32, 'cuda', ('e',))
        _, tail = fused_observe(q, k, v, z, scale, pt, summ, splits=2, mu=True)
        pooled, count = tile_pool(z, torch.ones(1, hk, keys, dtype=torch.bool, device='cuda'))
        csumm = allocate_summary(1, h, qb, kt, pt, 0, 'cuda', ('c',))
        _, ctail = fused_observe(q, k, v, z, scale, pt, csumm, splits=2, mu=False)
        cases = dict(exact=(tail, summ, {}),
                     compact=(ctail, csumm, dict(pool='compact', pooled=pooled, pool_count=count)))
        for name, (t, s, kw) in cases.items():
            base = dict(sensitivity=sens, log_threshold=-.5, summary=s, variant='generic', key_offset=pt * 64, **kw)
            want = route_only(t, z, ref, **base)
            row[f'{name}_generic_s1'] = timed(lambda: route_only(t, z, ref, **base))
            row[f'{name}_generic_s3'] = timed(lambda: route_only(t, z, ref, num_stages=3, **base))
            row[f'{name}_skip_fraction'] = want.skipped[want.eligible].float().mean().item()
            for stages in (2, 3, 4):
                for warps in (4, 8):
                    got = route_only(t, z, ref, pipelined=True, num_stages=stages, num_warps=warps, **base)
                    if not (torch.equal(got.skipped, want.skipped) and torch.equal(got.eligible, want.eligible)):
                        raise AssertionError(f'pipelined decisions differ: {keys} {name} s{stages} w{warps}')
                    row[f'{name}_pipelined_s{stages}_w{warps}'] = timed(
                        lambda stages=stages, warps=warps: route_only(t, z, ref, pipelined=True, num_stages=stages,
                                                                    num_warps=warps, **base))
        rows.append(row)
        print({k2: round(v2, 3) if isinstance(v2, float) else v2 for k2, v2 in row.items()}, flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_route_bench_v1', gpu=torch.cuda.get_device_name(0), rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
