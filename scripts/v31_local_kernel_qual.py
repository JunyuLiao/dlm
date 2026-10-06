"""Clean H100 qualification for the compact V31 LOCAL consumer.

This is a held-map consumer test.  It uses the native paged/interleaved cache,
the V31 (1023, 1023) mask, Q256, H16/Hkv8/D256, and the 512-token LOCAL
budget.  Timing is synchronized and alternates dense FA4, the old direct FA4
Q128 consumer, FA4 with the expanded Q64 map, and the compact Triton consumer.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def timed(torch, fn, samples, launches=20):
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    values = []
    for _ in range(samples):
        begin, end = event
        begin.record()
        for _ in range(launches):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) / launches)
    return dict(median_ms=statistics.median(values),
                p10_ms=statistics.quantiles(values, n=10)[0],
                p90_ms=statistics.quantiles(values, n=10)[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()

    import torch
    from scripts.v31_vllm_paired_bench import apply_fa4_local_fix
    apply_fa4_local_fix()
    from experiments.numerical_qk_reuse import v27_fa4, v31_local_kernel
    from experiments.numerical_qk_reuse.v31_local_sparse import alias64, geometry

    fwd = v27_fa4.load()
    rows = []
    for prefix in (187, 3500, 8192):
        n, h, hk, d, page, budget = 256, 16, 8, 256, 64, 512
        nk = prefix + n
        generator = torch.Generator(device='cuda').manual_seed(3106 + prefix)
        q = torch.randn((1, n, h, d), device='cuda', dtype=torch.bfloat16, generator=generator)
        pages = math.ceil(nk / page)
        cache = torch.randn((pages + 7, hk, page, 2 * d), device='cuda', dtype=torch.bfloat16,
                            generator=generator)
        key, value = cache.transpose(1, 2).split(d, -1)
        block_table = torch.randperm(pages + 7, device='cuda', generator=generator)[:pages].to(torch.int32)
        key, value, table = alias64(key, value, block_table, nk)
        used = torch.tensor([nk], device='cuda', dtype=torch.int32)
        native = dict(softmax_scale=d ** -0.5, causal=False, window_size_left=1023,
                      window_size_right=1023, page_table=table, seqused_k=used, num_splits=1)
        eligible, mandatory, kt = geometry(prefix, n)
        kept128 = torch.zeros((1, h, len(eligible), kt), device='cuda', dtype=torch.bool)
        for block, tiles in enumerate(eligible):
            prefix_tiles = [tile for tile in tiles if tile < prefix // 64]
            kept128[0, :, block, prefix_tiles[::2][:budget // 64] + mandatory[block]] = True
        kept64 = kept128.repeat_interleave(2, dim=2)[..., : (n + 63) // 64, :]
        compact_idx, compact_cnt = v31_local_kernel.compact_map(kept128, n)
        old_lists = v27_fa4.block_sparse_tensors(kept128, q_block=128)
        q64_lists = v27_fa4.block_sparse_tensors(kept64, q_block=64)

        dense = lambda: fwd(q, key, value, **native)[0]
        old_fa4 = lambda: fwd(q, key, value, **native, block_sparse_tensors=old_lists,
                              tile_mn=(128, 64), num_threads=384, pack_gqa=False)[0]
        q64_fa4 = lambda: fwd(q, key, value, **native, block_sparse_tensors=q64_lists,
                              tile_mn=(64, 64), num_threads=384, pack_gqa=False)[0]
        compact = lambda: v31_local_kernel.forward(q.transpose(1, 2), key, value, table,
                                                   compact_idx, compact_cnt, nk)
        ref = old_fa4()
        got = compact()
        eligible_tiles = h * sum(map(len, eligible)) * 2
        kept_tiles = int(kept64.sum())
        timings = {name: timed(torch, fn, args.samples)
                   for name, fn in (('dense_fa4', dense), ('old_fa4_q128', old_fa4),
                                    ('fa4_q64', q64_fa4), ('compact_triton', compact))}
        dense_ms = timings['dense_fa4']['median_ms']
        for name, timing in timings.items():
            timing['speedup_vs_dense'] = dense_ms / timing['median_ms']
        rows.append(dict(prefix=prefix, query_tokens=n, budget=budget,
                         eligible_tiles=eligible_tiles, kept_tiles=kept_tiles,
                         sparsity=1 - kept_tiles / eligible_tiles,
                         q64_counts=compact_cnt[0].tolist(),
                         compact_vs_old_fa4_max_abs_error=float((got - ref).float().abs().max()),
                         compact_vs_old_fa4_bitwise=torch.equal(got, ref),
                         timings=timings, samples=args.samples, launches_per_sample=20))
        print(json.dumps(rows[-1]), flush=True)

    result = dict(schema='v31_local_kernel_qual_v1', variant='local_compact_triton_q64',
                  budget=512, window=[1023, 1023], gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
