"""v27 dense-baseline fairness: is D_c64 (our 64-row kernel, all tiles kept) a strong dense attention
for DiffusionGemma's GLOBAL geometry (16 Q heads, 2 KV heads, head_dim 512, bidirectional)?

For the canvas decode shape (256 queries) at several key extents, times every dense implementation that
runs here and reports achieved TFLOPS (4*H*Q*K*D per call) and the max error vs an FP32 reference:
  dense64 (splits 1/2/4); torch SDPA with each backend (flash / efficient / cudnn / math), both with
  enable_gqa and with K/V repeated to 16 heads; FlexAttention (compiled, GQA); FlashInfer if importable.
Unsupported combinations are recorded with their error, never silently dropped.
usage: python -m scripts.v27_dense_kernel_bench OUT.json   (BENCH_KEYS=16640,32768,65536)
"""
from __future__ import annotations

import json
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
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device='cuda').manual_seed(0)
    H, HK, D, NQ = 16, 2, 512, 256
    scale = D ** -.5
    rows = []
    flex = None
    try:
        from torch.nn.attention.flex_attention import flex_attention
        flex = torch.compile(flex_attention, dynamic=False)
    except Exception as exc:  # recorded below
        flex_error = f'{type(exc).__name__}: {exc}'[:300]
    for keys in [int(x) for x in os.environ.get('BENCH_KEYS', '16640,32768,65536').split(',')]:
        q = torch.randn(1, H, NQ, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        kr, vr = k.repeat_interleave(H // HK, 1), v.repeat_interleave(H // HK, 1)
        flops = 4.0 * H * NQ * keys * D
        # FP32 reference on a slice of heads (full-size fp32 would not fit next to everything at 64K)
        ref = torch.nn.functional.scaled_dot_product_attention(q[:, :2].float(), kr[:, :2].float(), vr[:, :2].float(),
                                                               scale=scale).transpose(1, 2)
        row = dict(keys=keys, gflop=round(flops / 1e9, 1), impl={})

        def record(name, fn, layout_model_major):
            try:
                out = fn()
                got = (out if layout_model_major else out.transpose(1, 2))[:, :, :2].float()
                err = (got - ref).abs().max().item()
                ms = timed(fn)
                row['impl'][name] = dict(ms=round(ms, 4), tflops=round(flops / ms / 1e9, 1), max_abs_err=round(err, 5))
            except Exception as exc:
                row['impl'][name] = dict(error=f'{type(exc).__name__}: {exc}'[:240])
            torch.cuda.empty_cache()

        for s in (1, 2, 4):
            record(f'dense64_s{s}', lambda s=s: dense64(q, k, v, scale, splits=s), True)
        backends = dict(flash=SDPBackend.FLASH_ATTENTION, efficient=SDPBackend.EFFICIENT_ATTENTION,
                        cudnn=SDPBackend.CUDNN_ATTENTION, math=SDPBackend.MATH)
        for bname, backend in backends.items():
            if bname == 'math' and keys > 40000:
                row['impl'][f'sdpa_{bname}_gqa'] = dict(skipped='math backend materializes scores; skipped above 40K')
                continue

            def gqa(backend=backend):
                with sdpa_kernel([backend]):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)

            def rep(backend=backend):
                with sdpa_kernel([backend]):
                    return torch.nn.functional.scaled_dot_product_attention(q, kr, vr, scale=scale)
            record(f'sdpa_{bname}_gqa', gqa, False)
            record(f'sdpa_{bname}_repeat_kv', rep, False)
        if flex is not None:
            record('flex_attention_gqa', lambda: flex(q, k, v, scale=scale, enable_gqa=True), False)
            for bm, bn, st in ((64, 32, 1), (64, 64, 1), (32, 64, 1), (64, 32, 2)):
                opts = dict(BLOCK_M=bm, BLOCK_N=bn, num_stages=st)
                record(f'flex_attention_gqa_m{bm}_n{bn}_s{st}',
                       lambda opts=opts: flex(q, k, v, scale=scale, enable_gqa=True, kernel_options=opts), False)
        else:
            row['impl']['flex_attention_gqa'] = dict(error=flex_error)
        try:
            import flashinfer
            qn, kn, vn = (x[0].transpose(0, 1).contiguous() for x in (q, k, v))   # NHD layout
            record(f'flashinfer_{flashinfer.__version__}_single_prefill',
                   lambda: flashinfer.single_prefill_with_kv_cache(qn, kn, vn, causal=False, sm_scale=scale)[None],
                   True)
        except Exception as exc:
            row['impl']['flashinfer'] = dict(error=f'{type(exc).__name__}: {exc}'[:240])
        try:   # vLLM's Triton unified attention (paged KV), the backend vLLM uses for head sizes FA lacks
            import math
            from vllm.attention.ops.triton_unified_attention import unified_attention
            bs = 64
            nb = math.ceil(keys / bs)
            kc = torch.zeros(nb * bs, HK, D, device='cuda', dtype=torch.bfloat16)
            vc = torch.zeros_like(kc)
            kc[:keys], vc[:keys] = k[0].transpose(0, 1), v[0].transpose(0, 1)
            kc, vc = kc.view(nb, bs, HK, D), vc.view(nb, bs, HK, D)
            qn = q[0].transpose(0, 1).contiguous()
            out = torch.empty_like(qn)
            cu_q = torch.tensor([0, NQ], device='cuda', dtype=torch.int32)
            used = torch.tensor([keys], device='cuda', dtype=torch.int32)
            table = torch.arange(nb, device='cuda', dtype=torch.int32)[None]

            def vllm_fn():
                unified_attention(q=qn, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=NQ, seqused_k=used,
                                  max_seqlen_k=keys, softmax_scale=scale, causal=False, window_size=(-1, -1),
                                  block_table=table, softcap=0, q_descale=None, k_descale=None, v_descale=None)
                return out[None]
            import vllm
            record(f'vllm_{vllm.__version__}_triton_unified', vllm_fn, True)
        except Exception as exc:
            row['impl']['vllm_triton_unified'] = dict(error=f'{type(exc).__name__}: {exc}'[:240])
        rows.append(row)
        print(json.dumps(dict(keys=keys, **{n: (x.get('ms'), x.get('tflops')) if 'ms' in x else x.get('error', x.get('skipped', x.get('note')))[:90]
                                            for n, x in row['impl'].items()})), flush=True)
    peak = 989.4  # H100 SXM dense BF16 TFLOPS (no sparsity), for the roofline fraction
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_dense_kernel_bench_v1', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       h100_bf16_dense_peak_tflops=peak, shape=dict(H=H, HK=HK, D=D, NQ=NQ, bidirectional=True),
                       rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
