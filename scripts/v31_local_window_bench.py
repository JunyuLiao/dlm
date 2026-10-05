"""Micro-benchmark of vLLM's LOCAL (sliding-window) decode attention call for DiffusionGemma on FA4 (GPU).

Why: nsys of the official dense decode (mpk, 2026-10-05) shows each LOCAL layer's canvas call taking 0.22 ms at 32K and
0.83 ms at 128K (but 0.04 ms at 64K), against a 1024-token window that should cost about the same at any length;
25 LOCAL layers then cost more per forward than the 5 GLOBAL ones at 128K (20.7 vs 16.6 ms).
This reproduces the call exactly as vllm/v1/attention/backends/flash_attn.py makes it for a decode canvas
(head_dim 256, 16 query / 8 KV heads, paged cache with 128-token pages, window (1023, 1023), causal False,
dynamic_causal = tensor([False]), num_splits 1) and compares, per key length:
  A  the vLLM call (dynamic_causal tensor)          B  the same without dynamic_causal (static non-causal)
  C  no window (full attention, the cost ceiling)
timing (CUDA events, median of warm reps), max |A - B|, and A against an FP32 reference restricted to the window
(bottom-right aligned: query i sits at key position nk - nq + i). Prints one JSON row per key length.
usage: python v31_local_window_bench.py OUT.jsonl [--keys 16384,24576,...]
"""
import argparse
import json
import statistics

import torch

NQ, H, HK, D, PAGE, W = 256, 16, 8, 256, 128, 1023


def timed(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(statistics.median(ts), 4)


def reference(q, k, v, nk, scale):
    """FP32 windowed bidirectional attention of the canvas queries; q [NQ, H, D], k / v [nk, HK, D]."""
    pos_q = torch.arange(nk - NQ, nk, device=q.device)
    lo = max(0, nk - NQ - W)
    kk, vv = k[lo:nk].float(), v[lo:nk].float()
    pos_k = torch.arange(lo, nk, device=q.device)
    m = (pos_k[None, :] >= pos_q[:, None] - W) & (pos_k[None, :] <= pos_q[:, None] + W)
    out = torch.empty(NQ, H, D, device=q.device)
    for h in range(H):
        s = (q[:, h].float() @ kk[:, h // (H // HK)].T) * scale
        out[:, h] = torch.softmax(s.masked_fill(~m, float('-inf')), -1) @ vv[:, h // (H // HK)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--keys', default='16384,24576,32907,40960,49152,57344,65673,73728,81920,98304,114688,131201')
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    torch.manual_seed(0)
    scale = D ** -0.5
    rows = []
    for nk in (int(x) for x in a.keys.split(',')):
        pages = -(-nk // PAGE)
        kc = torch.randn(pages + 4, PAGE, HK, D, device='cuda', dtype=torch.bfloat16)
        vc = torch.randn(pages + 4, PAGE, HK, D, device='cuda', dtype=torch.bfloat16)
        table = torch.randperm(pages + 4, device='cuda')[:pages].to(torch.int32)[None]
        q = torch.randn(NQ, H, D, device='cuda', dtype=torch.bfloat16)
        cu_q = torch.tensor([0, NQ], device='cuda', dtype=torch.int32)
        used = torch.tensor([nk], device='cuda', dtype=torch.int32)
        dyn = torch.tensor([False], device='cuda')
        dyn_i = torch.tensor([0], device='cuda', dtype=torch.int32)     # vLLM with the PR #51994 fix keeps it int32
        out_buf = torch.empty_like(q)
        kw = dict(q=q, k=kc, v=vc, out=out_buf, cu_seqlens_q=cu_q, max_seqlen_q=NQ, seqused_k=used,
                  max_seqlen_k=pages * PAGE, softmax_scale=scale, block_table=table, fa_version=4, num_splits=1,
                  alibi_slopes=None, softcap=0.0, scheduler_metadata=None, s_aux=None)

        def call_a():
            return flash_attn_varlen_func(causal=False, window_size=[W, W], dynamic_causal=dyn, **kw)

        def call_b():
            return flash_attn_varlen_func(causal=False, window_size=[W, W], **kw)

        def call_c():
            return flash_attn_varlen_func(causal=False, window_size=None, **kw)
        row = dict(keys=nk, pages=pages)
        try:
            oa, ob = call_a(), call_b()
            oa = oa[0] if isinstance(oa, tuple) else oa
            ob = ob[0] if isinstance(ob, tuple) else ob
            k_lin = kc[table[0].long()].reshape(-1, HK, D)[:nk]
            v_lin = vc[table[0].long()].reshape(-1, HK, D)[:nk]
            ref = reference(q, k_lin, v_lin, nk, scale)
            row.update(max_abs_a_minus_b=float((oa.float() - ob.float()).abs().max()),
                       a_vs_window_ref_max_abs=float((oa.float() - ref).abs().max()),
                       b_vs_window_ref_max_abs=float((ob.float() - ref).abs().max()),
                       ms_a_vllm_dynamic=timed(call_a), ms_b_static=timed(call_b), ms_c_full=timed(call_c))
        except Exception as e:                                    # keep sweeping; report the failure
            import traceback
            row['error'] = f'{type(e).__name__}: {e}'[-1500:]
            if nk == int(a.keys.split(',')[0]):
                traceback.print_exc()
            try:                                                  # the int32 dynamic-causal buffer vLLM really passes
                def call_ai():
                    return flash_attn_varlen_func(causal=False, window_size=[W, W], dynamic_causal=dyn_i, **kw)
                call_ai()
                row['ms_a_vllm_dynamic_int32'] = timed(call_ai)
                row['ms_b_static'] = timed(call_b)
                row['ms_c_full'] = timed(call_c)
            except Exception as e2:
                row['error_int32'] = f'{type(e2).__name__}: {e2}'[-800:]
        print(json.dumps(row), flush=True)
        rows.append(row)
        del kc, vc
        torch.cuda.empty_cache()
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
