"""Split-KV for FA4 block sparsity without touching the kernel: S 'virtual splits' as batch entries that alias the same
paged K/V through the page table, each entry given one contiguous share of every kept-tile list, then an exact LSE
combine. Rationale: on SM90 the official kernel's num_splits > 1 with block sparsity re-does the whole list in every
split (time grows with S), and dense FA4 only splits effectively on its dynamic-causal path, which is what vLLM's
dense uses (~0.6x the num_splits=1 time). This puts the sparse path on the same footing: more CTAs, same kernel.
Checks each result against an fp32 masked reference; timings are medians of 30 CUDA-event runs incl. the combine.
usage: python v27_fa4_sparse_split_alias_bench.py OUT_JSONL"""
import json
import math
import sys

import torch
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

H, HK, D, CL, TN, MR, PAGE = 16, 2, 512, 256, 64, 128, 64


def timed(fn, reps=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(sorted(ts)[reps // 2], 4)


def split_lists(kept, S):
    """kept [H, M, N] -> lists for S batch entries; entry s takes the s-th contiguous share of each (h, m) list."""
    h, m, n = kept.shape
    order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True).to(torch.int32)      # kept first, ascending
    cnt = kept.sum(-1).to(torch.int32)                                                      # [H, M]
    lo = [(cnt * s) // S for s in range(S + 1)]
    idx = torch.zeros((S, h, m, n), device=kept.device, dtype=torch.int32)
    ar = torch.arange(n, device=kept.device)
    counts = []
    for s in range(S):
        c = (lo[s + 1] - lo[s])
        src = (lo[s][..., None] + ar).clamp_max(n - 1)
        idx[s] = torch.gather(order, -1, src.long())
        counts.append(c)
    return BST(mask_block_cnt=torch.zeros((S, h, m), device=kept.device, dtype=torch.int32),
               mask_block_idx=torch.zeros((S, h, m, 1), device=kept.device, dtype=torch.int32),
               full_block_cnt=torch.stack(counts).contiguous(), full_block_idx=idx.contiguous(), block_size=(MR, TN))


def combine(o, lse):
    """o [S, Q, H, D] bf16, lse [S, H, Q] fp32 -> [Q, H, D]"""
    w = torch.softmax(lse, dim=0)                                  # exact: exp(lse_s - logsumexp_s lse)
    return (o.float() * w.permute(0, 2, 1)[..., None]).sum(0).to(o.dtype)


def main():
    out = open(sys.argv[1], 'w')
    g = torch.Generator(device='cuda').manual_seed(23)
    scale = 1.0 / math.sqrt(D)
    dyn = torch.tensor([False], device='cuda')
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        npages = keys // PAGE
        kc, vc = k[0].view(npages, PAGE, HK, D), v[0].view(npages, PAGE, HK, D)
        table1 = torch.arange(npages, device='cuda', dtype=torch.int32)[None]
        kr = k[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        vr = v[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        sc = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
        rec = dict(keys=keys,
                   dense_vllm_dyn=timed(lambda: fwd(q, kc, vc, softmax_scale=scale, causal=True, dynamic_causal=dyn,
                                                    num_splits=0, page_table=table1,
                                                    seqused_k=torch.tensor([keys], device='cuda', dtype=torch.int32))))
        n = keys // TN
        for keep in (1.0, 0.25, 0.12):
            kept = torch.rand(H, CL // MR, n, device='cuda', generator=g) < keep
            kept[..., 0] = True
            kept[..., -(CL // TN):] = True
            mask = kept.repeat_interleave(MR, 1).repeat_interleave(TN, 2)
            ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
            for S in (1, 2, 3, 4):
                L = split_lists(kept, S)
                qS = q.expand(S, -1, -1, -1)
                tableS = table1.expand(S, -1).contiguous()
                usedS = torch.full((S,), keys, device='cuda', dtype=torch.int32)

                def run():
                    o, lse = fwd(qS, kc, vc, softmax_scale=scale, causal=False, page_table=tableS, seqused_k=usedS,
                                 block_sparse_tensors=L, num_splits=1, return_lse=True)[:2]
                    return o[0] if S == 1 else combine(o, lse)
                key = f'keep{keep}_S{S}'
                try:
                    res = run()
                    rel = float((res.float() - ref).abs().max() / ref.abs().max())
                    rec[key] = dict(ms=timed(run), rel_err=round(rel, 5))
                except Exception as e:
                    rec[key] = repr(e)[:240]
            del mask, ref
            torch.cuda.empty_cache()
        del sc, kr, vr
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
