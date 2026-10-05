"""v27 observation-cost microbenchmark at the GLOBAL geometry (one layer, synthetic tensors).

Per key extent: dense64 alone; fused observation with/without mu (+ the route LOAD it
enables); and the unfused path (dense64 + grouped score producer + route STORE with
summaries). usage: python -m scripts.v27_fused_bench OUT.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch


def timed(fn, reps=10, warm=3):
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
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
    from experiments.numerical_qk_reuse.integration import Attention
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
        row = dict(keys=keys)
        row['dense64'] = timed(lambda: dense64(q, k, v, scale, splits=2))

        def fused(mu):
            summ = allocate_summary(1, h, qb, kt, pt, 32 if mu else 0, 'cuda', ('f',))
            out, tail = fused_observe(q, k, v, z, scale, pt, summ, splits=2, mu=mu)
            return summ, tail
        row['fused_mu'] = timed(lambda: fused(True))
        row['fused_nomu'] = timed(lambda: fused(False))
        summ, tail = fused(True)
        row['route_load_tail'] = timed(lambda: route_only(tail, z, ref, sensitivity=sens, log_threshold=-.5,
                                                          summary=summ, variant='generic', key_offset=pt * 64))
        for prec in ('tf32', 'bf16'):
            def fp(prec=prec):
                s3 = allocate_summary(1, h, qb, kt, pt, 32, 'cuda', ('p',))
                return fused_observe(q, k, v, z, scale, pt, s3, splits=2, mu=True, mu_precision=prec)
            row[f'fused_mu_{prec}'] = timed(fp)
        for ns in (2, 3):
            for nw in (4, 8):
                row[f'route_load_s{ns}_w{nw}'] = timed(lambda ns=ns, nw=nw: route_only(
                    tail, z, ref, sensitivity=sens, log_threshold=-.5, summary=summ, variant='generic',
                    key_offset=pt * 64, num_stages=ns, num_warps=nw))

        def unfused():
            dense64(q, k, v, scale, splits=2)
            scores = Attention.observe_scores_grouped(q, k, None, scale, False, None, 0)
            s2 = allocate_summary(1, h, qb, kt, pt, 32, 'cuda', ('u',))
            route_only(scores, z, ref, sensitivity=sens, log_threshold=-.5, summary=s2, store_summary=True,
                       variant='generic')
        row['unfused_dense_producer_store'] = timed(unfused)
        row['fused_mu_plus_route'] = row['fused_mu'] + row['route_load_tail']
        rows.append(row)
        print({k2: round(v2, 3) if isinstance(v2, float) else v2 for k2, v2 in row.items()}, flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_fused_bench_v1', gpu=torch.cuda.get_device_name(0), rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
