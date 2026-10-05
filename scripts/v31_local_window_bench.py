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
FA4's compile cache keys this call without the presence of dynamic_causal, so the variants run in separate processes:
--mode a (vLLM call, with the FP32 window check), b (static non-causal, with the check), c (no window, timing only).
--fix installs FA4_LOCAL_FIX (v31_vllm_paired_bench.apply_fa4_local_fix: the bidirectional LOCAL call keeps the window's
block range) before the first compile. --save OUT.pt keeps the outputs; --compare REF.pt reports, per key length, the
max |output - REF| and whether they are bitwise equal (the inputs are the same seeded draws in every process).
usage: python v31_local_window_bench.py OUT.jsonl --mode a|b|c [--fix] [--save X.pt] [--compare X.pt] [--keys ...]
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
    ap.add_argument('--mode', choices=('a', 'b', 'c'), required=True)
    ap.add_argument('--fix', action='store_true')
    ap.add_argument('--save')
    ap.add_argument('--compare')
    a = ap.parse_args()
    fix_path = None
    if a.fix:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from v31_vllm_paired_bench import apply_fa4_local_fix
        fix_path = apply_fa4_local_fix()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    ref = torch.load(a.compare) if a.compare else None
    saved = {}
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
        row = dict(keys=nk, pages=pages, mode=a.mode, fa4_local_fix=fix_path)
        call = {'a': call_a, 'b': call_b, 'c': call_c}[a.mode]
        try:
            o = call()
            o = o[0] if isinstance(o, tuple) else o
            if a.mode != 'c':
                k_lin = kc[table[0].long()].reshape(-1, HK, D)[:nk]
                v_lin = vc[table[0].long()].reshape(-1, HK, D)[:nk]
                row['vs_window_ref_max_abs'] = float((o.float() - reference(q, k_lin, v_lin, nk, scale)).abs().max())
            if a.save:
                saved[nk] = o.detach().cpu().clone()
            if ref is not None and nk in ref:
                r = ref[nk].to(o.device)
                row['vs_compare_max_abs'] = float((o.float() - r.float()).abs().max())
                row['vs_compare_bitwise'] = bool(torch.equal(o, r))
            row['ms'] = timed(call)
        except Exception as e:                                    # keep sweeping; report the failure
            row['error'] = f'{type(e).__name__}: {e}'[-600:]
        print(json.dumps(row), flush=True)
        rows.append(row)
        del kc, vc
        torch.cuda.empty_cache()
    if a.save:
        torch.save(saved, a.save)
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
