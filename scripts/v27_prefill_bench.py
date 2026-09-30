"""v27: prompt prefill and canvas-append attention, current shared path vs official FA4 (all arms share these; not a
method component). GLOBAL: 16 Q / 2 KV heads, head_dim 512, causal. LOCAL: 16 Q / 8 KV heads, head_dim 256,
sliding window 1024, causal. Prefill = all prompt tokens as queries; append = 256 new canvas tokens attending to the
cached prefix + themselves (causal, bottom-right aligned). CUDA-event median of 5 after 1 warm-up, ms.
usage: python -m scripts.v27_prefill_bench OUT.json   (V27_FA4_OVERLAY set)
"""
from __future__ import annotations

import json
import statistics
import sys

import torch


def timed(fn, n=5):
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
    return round(statistics.median(out), 3)


def main(argv=None):
    argv = argv or sys.argv[1:]
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64
    fwd = v27_fa4.load()
    g = torch.Generator(device='cuda').manual_seed(0)
    rows = []
    for n in (32768, 65536):
        for kind, h, hk, d, window in (('global', 16, 2, 512, 0), ('local', 16, 8, 256, 1024)):
            q = torch.randn(1, h, n, d, device='cuda', dtype=torch.bfloat16, generator=g) * d ** -.25
            k = torch.randn(1, hk, n, d, device='cuda', dtype=torch.bfloat16, generator=g) * d ** -.25
            v = torch.randn(1, hk, n, d, device='cuda', dtype=torch.bfloat16, generator=g)
            row = dict(tokens=n, kind=kind)
            row['prefill_dense64_s1'] = timed(lambda: dense64(q, k, v, 1.0, splits=1, causal=True, window=window))
            win = dict(window_size_left=window - 1, window_size_right=0) if window else {}
            try:
                row['prefill_fa4'] = timed(lambda: fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                        softmax_scale=1.0, causal=True, **win)[0])
                a = dense64(q, k, v, 1.0, splits=1, causal=True, window=window).float()
                b = fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=1.0, causal=True,
                        **win)[0].float()
                row['prefill_max_abs_fa4_vs_dense64'] = round(float((a - b.reshape(a.shape) if a.shape == b.shape
                                                                     else a - b.transpose(1, 2)).abs().max()), 5)
            except Exception as exc:
                row['prefill_fa4'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            if kind == 'global':
                qa = q[:, :, -256:]
                rep = h // hk
                row['append_sdpa_mem_eff'] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
                    qa, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=False,
                    attn_mask=torch.ones(256, n, dtype=torch.bool, device='cuda').tril(n - 256), scale=1.0))
                try:
                    row['append_fa4'] = timed(lambda: fwd(qa.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                          softmax_scale=1.0, causal=True)[0])
                except Exception as exc:
                    row['append_fa4'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            del q, k, v
            torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(argv[0], 'x', encoding='utf-8') as f:
        json.dump(dict(schema='v27_prefill_bench_v1', torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
