"""H100 LOCAL sparse consumer qualification and paired timing, without model text.

Use native interleaved/paged cache strides, the (1023,1023) token mask, Q256,
H16/Hkv8/D256, and the same Q128/K64 map for every kernel configuration.
This measures the held-map consumer; observation/selection remain extra costs.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    a = p.parse_args()
    import torch
    from scripts.v31_vllm_paired_bench import apply_fa4_local_fix
    apply_fa4_local_fix()
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v31_local_sparse import alias64, geometry
    fwd = v27_fa4.load()
    start = time.perf_counter()
    results = []
    # Decisions here depend only on synthetic consumer cost, never AIME answers.
    configs = [dict(tile_mn=(128, 64), num_threads=384, pack_gqa=False),
               dict(tile_mn=(64, 64), num_threads=384, pack_gqa=False),
               dict(tile_mn=(64, 64), num_threads=256, pack_gqa=False),
               dict(tile_mn=(128, 64), num_threads=384, pack_gqa=True)]
    for prefix in (187, 3500, 8192):
        n, h, hk, d, page = 256, 16, 8, 256, 64
        nk = prefix + n
        g = torch.Generator(device='cuda').manual_seed(3106 + prefix)
        q = torch.randn((1, n, h, d), device='cuda', dtype=torch.bfloat16, generator=g)
        pages = math.ceil(nk / page)
        cache = torch.randn((pages + 7, hk, page, 2*d), device='cuda', dtype=torch.bfloat16, generator=g)
        k, v = cache.transpose(1, 2).split(d, -1)
        table = torch.randperm(pages + 7, device='cuda', generator=g)[:pages].to(torch.int32)
        k, v, tab = alias64(k, v, table, nk)
        used = torch.tensor([nk], device='cuda', dtype=torch.int32)
        args = dict(softmax_scale=d**-.5, causal=False, window_size_left=1023, window_size_right=1023,
                    page_table=tab, seqused_k=used, num_splits=1)
        eligible, mandatory, kt = geometry(prefix, n)
        maps = []
        for budget in (None, 512):
            kept = torch.zeros((1, h, 2, kt), device='cuda', dtype=torch.bool)
            for b, ts in enumerate(eligible):
                pre = [t for t in ts if t < prefix // 64]
                if budget is not None:
                    pre = pre[::2][:budget // 64]
                kept[0, :, b, pre + mandatory[b]] = True
            maps.append((kept, v27_fa4.block_sparse_tensors(kept)))
        dense = lambda: fwd(q, k, v, **args)[0]
        ref = dense()
        calls, rows = [dense], [dict(kernel='dense_fa4', prefix=prefix)]
        for cfg in configs:
            row = dict(kernel='sparse_fa4', prefix=prefix, config=cfg,
                       eligible_tiles=h * sum(map(len, eligible)), kept_tiles=int(maps[1][0].sum()))
            try:
                allk = fwd(q, k, v, **args, block_sparse_tensors=maps[0][1], **cfg)[0]
                row['allkept_max_abs_error'] = float((allk-ref).float().abs().max())
                row['allkept_bitwise'] = torch.equal(allk, ref)
                row['sparsity'] = 1 - row['kept_tiles'] / row['eligible_tiles']
                # FP32 oracle with exact logical page order, token window and tile map.
                ck = k[tab[0].long()].reshape(-1,hk,d)[:nk].transpose(0,1).repeat_interleave(h//hk,0).float()
                cv = v[tab[0].long()].reshape(-1,hk,d)[:nk].transpose(0,1).repeat_interleave(h//hk,0).float()
                score = torch.einsum('qhd,hkd->hqk',q[0].float(),ck) * d**-.5
                qp = torch.arange(n,device='cuda') + prefix
                kp = torch.arange(nk,device='cuda')
                mask = maps[1][0][0].repeat_interleave(128,1).repeat_interleave(64,2)[:,:n,:nk]
                mask = mask & ((qp[:,None]-kp[None]).abs() <= 1023)[None]
                oracle = torch.einsum('hqk,hkd->qhd',score.masked_fill(~mask,float('-inf')).softmax(-1),cv)
                call = lambda cfg=cfg: fwd(q,k,v,**args,block_sparse_tensors=maps[1][1],**cfg)[0]
                got = call()
                row['sparse_vs_fp32_max_abs_error'] = float((got[0].float()-oracle).abs().max())
                if row['allkept_max_abs_error'] > .016 or row['sparse_vs_fp32_max_abs_error'] > .016:
                    raise AssertionError('LOCAL numeric oracle failed')
                calls.append(call)
                rows.append(row)
            except Exception as e:
                row['error'] = str(e)[:500]
                results.append(row)
        for call in calls:
            for _ in range(8):
                call()
        torch.cuda.synchronize()
        times = [[] for _ in calls]
        events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in calls]
        for sample in range(a.samples):
            for j in [(sample+i)%len(calls) for i in range(len(calls))]:
                begin,end = events[j]
                begin.record()
                for _ in range(10):
                    calls[j]()
                end.record()
                end.synchronize()
                times[j].append(begin.elapsed_time(end)/10)
        for row, ts in zip(rows, times):
            row.update(median_ms=statistics.median(ts), samples=a.samples, launches_per_sample=10)
            row['speedup_vs_dense'] = statistics.median(times[0]) / row['median_ms']
            results.append(row)
            print(json.dumps(row), flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                      elapsed_s=time.perf_counter()-start, rows=results),indent=2)+'\n')


if __name__ == '__main__':
    main()
