"""v27 official dense SOTA check at DiffusionGemma's GLOBAL decode geometry (256 queries, 16 Q / 2 KV heads,
head_dim 512, bidirectional, bf16), all in ONE process so kernels share the same torch/driver:
  - FA4 (vLLM flash-attention fork, CuTe DSL) dense: num_splits in {1, 2, 3, 4, 6, 8, heuristic(0)} x pack_gqa
  - FA4 block-sparse interface (Q128 x KV64 full-block lists, random keep maps, first tile kept) with the same
    num_splits choices, keep in {1.0, 0.5, 0.25, 0.1}
  - FlashInfer BatchPrefillWithRaggedKVCacheWrapper, backend fa2 (its scheduler splits KV); single_prefill fa2
Every configuration is checked against an FP32 reference (max abs error reported) before timing.
Timing: CUDA-event median of 20 after 3 warm-ups, milliseconds.
usage: python v27_dense_sota_bench.py OUT.json    (V27_FA4_OVERLAY must point at the FA4 overlay)
"""
from __future__ import annotations

import json
import math
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


def run(fn, ref):
    try:
        err = float((fn().float() - ref).abs().max())
        return dict(ms=timed(fn), err=round(err, 5))
    except Exception as exc:
        return dict(error=f'{type(exc).__name__}: {str(exc)[:240]}')


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse import v27_fa4
    fwd = v27_fa4.load()
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    scale = D ** -.5
    rows = []
    for keys in (16640, 32768, 65536):
        q = torch.randn(1, NQ, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        rep = H // HK
        qf, kf, vf = (x.transpose(1, 2).float() for x in (q, k, v))
        ref = torch.nn.functional.scaled_dot_product_attention(
            qf, kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1), scale=scale).transpose(1, 2)
        row = dict(keys=keys)
        for splits in (1, 2, 3, 4, 6, 8, 0):
            for pack in (None, True, False):
                row[f'fa4_dense_s{splits}_pack{pack}'] = run(
                    lambda splits=splits, pack=pack: fwd(q, k, v, softmax_scale=scale, causal=False,
                                                         num_splits=splits, pack_gqa=pack)[0], ref)
        kt = math.ceil(keys / 64)
        for keep in (1.0, 0.5, 0.25, 0.1):
            kept = torch.rand(1, H, 2, kt, device='cuda', generator=g) < keep
            kept[..., 0] = True
            lists = v27_fa4.block_sparse_tensors(kept)
            mask = kept.repeat_interleave(128, 2)[:, :, :NQ].repeat_interleave(64, 3)[..., :keys]
            sref = torch.nn.functional.scaled_dot_product_attention(
                qf, kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1), attn_mask=mask,
                scale=scale).transpose(1, 2)
            for splits in (1, 2, 4, 0):
                row[f'fa4_sparse_keep{keep}_s{splits}'] = run(
                    lambda splits=splits, lists=lists: fwd(q, k, v, softmax_scale=scale, causal=False,
                                                           num_splits=splits, block_sparse_tensors=lists)[0], sref)
            del mask, sref
        try:
            import flashinfer
            row['flashinfer'] = flashinfer.__version__
            ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device='cuda')
            w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout='NHD', backend='fa2')
            qo = torch.tensor([0, NQ], dtype=torch.int32, device='cuda')
            kvp = torch.tensor([0, keys], dtype=torch.int32, device='cuda')
            w.plan(qo, kvp, H, HK, D, causal=False, sm_scale=scale, q_data_type=torch.bfloat16)
            row['flashinfer_batch_ragged_fa2'] = run(lambda: w.run(q[0], k[0], v[0])[None], ref)
            row['flashinfer_single_fa2'] = run(lambda: flashinfer.single_prefill_with_kv_cache(
                q[0], k[0], v[0], causal=False, sm_scale=scale, backend='fa2')[None], ref)
        except Exception as exc:
            row['flashinfer_error'] = f'{type(exc).__name__}: {str(exc)[:240]}'
        rows.append(row)
        best = min(((k_, v_['ms']) for k_, v_ in row.items() if isinstance(v_, dict) and 'ms' in v_
                    and k_.startswith(('fa4_dense', 'flashinfer'))), key=lambda t: t[1])
        print(json.dumps(dict(keys=keys, fastest_dense=best)), flush=True)
        print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_dense_sota_bench_v1', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
