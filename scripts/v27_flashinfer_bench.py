"""v27 official-baseline completeness: FlashInfer at DiffusionGemma's GLOBAL decode geometry (256 queries, 16 Q /
2 KV heads, head_dim 512, bidirectional, bf16), dense single-request prefill API per backend, plus FlashInfer's
BSR block-sparse wrapper when the dense kernel exists. Same timing protocol as the FA4 bench (CUDA-event median
of 20 after 3 warm-ups); FA4 dense is timed in the same process when V27_FA4_OVERLAY is set.
usage: python v27_flashinfer_bench.py OUT.json   (JIT: needs nvcc + ninja on PATH, FLASHINFER_WORKSPACE_BASE in dyh)
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch


def timed(fn, n=20):
    for _ in range(3):
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
    import flashinfer
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    scale = D ** -.5
    rows = []
    for keys in (16640, 32768, 65536):
        q = torch.randn(NQ, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None].float(), k.transpose(0, 1)[None].repeat_interleave(H // HK, 1).float(),
            v.transpose(0, 1)[None].repeat_interleave(H // HK, 1).float(), scale=scale)[0].transpose(0, 1)
        row = dict(keys=keys, flashinfer=flashinfer.__version__)
        for backend in ('fa2', 'fa3', 'auto'):
            try:
                fn = lambda backend=backend: flashinfer.single_prefill_with_kv_cache(
                    q, k, v, causal=False, sm_scale=scale, backend=backend)
                err = float((fn().float() - ref).abs().max())
                row[f'dense_{backend}'] = dict(ms=timed(fn), max_abs_err_vs_fp32=round(err, 5))
            except Exception as exc:
                row[f'dense_{backend}'] = dict(error=f'{type(exc).__name__}: {str(exc)[:300]}')
        # batch ragged API (one request): FlashInfer's scheduler may split KV across CTAs, unlike single_prefill
        try:
            ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device='cuda')
            w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout='NHD', backend='fa2')
            qo = torch.tensor([0, NQ], dtype=torch.int32, device='cuda')
            kv = torch.tensor([0, keys], dtype=torch.int32, device='cuda')
            w.plan(qo, kv, H, HK, D, causal=False, sm_scale=scale, q_data_type=torch.bfloat16)
            fn = lambda: w.run(q, k, v)
            err = float((fn().float() - ref).abs().max())
            row['batch_ragged_fa2'] = dict(ms=timed(fn), max_abs_err_vs_fp32=round(err, 5))
        except Exception as exc:
            row['batch_ragged_fa2'] = dict(error=f'{type(exc).__name__}: {str(exc)[:300]}')
        if os.environ.get('V27_FA4_OVERLAY'):
            from experiments.numerical_qk_reuse import v27_fa4
            qm, km, vm = q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
            row['fa4_dense_ms'] = timed(lambda: v27_fa4.dense(qm, km, vm, scale))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_flashinfer_bench_v1', gpu=torch.cuda.get_device_name(0),
                       torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
