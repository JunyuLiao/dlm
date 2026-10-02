"""vLLM port, step 2 check: block-sparse FA4 through vLLM's own varlen + paged calling convention.

vLLM's decoder GLOBAL call is `flash_attn_varlen_func(q, key_cache, value_cache, cu_seqlens_q, seqused_k,
block_table, causal=False, fa_version=4, num_splits=...)` over 32- or 64-token pages. This probe adds FA4's varlen
block-sparse lists (2D [H, total_m_blocks] / [H, total_n_blocks] with cu_total_m_blocks / cu_block_idx_offsets) to
that exact call at the DiffusionGemma GLOBAL geometry (canvas 256 queries, 16 q heads, 2 kv heads, head_dim 512) and
checks:
  1. correctness: sparse output vs an fp32 masked reference, with K/V scattered over shuffled pages that hold other
     data in their unused slots;
  2. per-call time (CUDA events, median of repeats): vLLM dense (num_splits 1/32/64, the SM90 hd512 warm-up family)
     vs the same call with block-sparse lists at several keep fractions, including all tiles kept.
Requires the FA4 paged block-sparse fix (patches/vllm_0.30.0_fa4_sm90_paged_block_sparse_tma.patch).
usage: python v27_vllm_varlen_sparse_probe.py OUT_JSONL [PAGE]
"""
import json
import math
import sys

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST

H, HK, D, CL = 16, 2, 512, 256
TN = 64


def varlen_lists(kept, m_rows):
    """kept: [H, M, N] bool -> varlen lists for one sequence (all kept tiles as 'full' blocks, none partial)."""
    h, m, n = kept.shape
    order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True).to(torch.int32)    # kept tiles first
    full_cnt = kept.sum(-1).to(torch.int32)
    return BST(mask_block_cnt=torch.zeros((h, m), device=kept.device, dtype=torch.int32),
               mask_block_idx=torch.zeros((h, m * n), device=kept.device, dtype=torch.int32),
               full_block_cnt=full_cnt.contiguous(), full_block_idx=order.reshape(h, m * n).contiguous(),
               cu_total_m_blocks=torch.tensor([0, m], device=kept.device, dtype=torch.int32),
               cu_block_idx_offsets=torch.tensor([0, m * n], device=kept.device, dtype=torch.int32),
               block_size=(m_rows, TN))


def timed(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out_path = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    g = torch.Generator(device='cuda').manual_seed(7)
    rec_out = open(out_path, 'w')
    scale = D ** -.5
    for keys in (32768, 65536, 94208):
        q = torch.randn(CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        npages = math.ceil(keys / page)
        perm = torch.randperm(npages + 7, device='cuda', generator=g)[:npages]
        kc = torch.randn(npages + 7, page, HK, D, device='cuda', dtype=torch.bfloat16, generator=g) * 5
        vc = torch.randn_like(kc) * 5
        pad = npages * page - keys
        kc[perm] = torch.cat((k, kc.new_zeros(pad, HK, D))).view(npages, page, HK, D)
        vc[perm] = torch.cat((v, vc.new_zeros(pad, HK, D))).view(npages, page, HK, D)
        table = perm.to(torch.int32)[None].contiguous()
        cu_q = torch.tensor([0, CL], device='cuda', dtype=torch.int32)
        used = torch.tensor([keys], device='cuda', dtype=torch.int32)
        out = torch.empty_like(q)

        def call(splits=1, bst=None):
            return flash_attn_varlen_func(q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=keys,
                                          seqused_k=used, softmax_scale=scale, causal=False, block_table=table,
                                          out=out, fa_version=4, num_splits=splits, block_sparse_tensors=bst)

        rec = dict(keys=keys, page=page, canvas=CL)
        for s in (1, 32, 64):
            try:
                rec[f'dense_ms_splits{s}'] = round(timed(lambda: call(s)), 4)
            except Exception as e:
                rec[f'dense_ms_splits{s}'] = repr(e)[:160]
        n = math.ceil(keys / TN)
        for m_rows in (64, 128):
            m = CL // m_rows
            try:
                for keep in (1.0, 0.25, 0.12):
                    kept = torch.rand(H, m, n, device='cuda', generator=g) < keep
                    kept[..., 0] = True
                    kept[..., -(CL // TN):] = True                       # canvas tiles always kept (as in the method)
                    L = varlen_lists(kept, m_rows)
                    call(1, L)
                    got = out.clone()
                    kr = k.transpose(0, 1).repeat_interleave(H // HK, 0).float()
                    vr = v.transpose(0, 1).repeat_interleave(H // HK, 0).float()
                    sc = torch.einsum('qhd,hkd->hqk', q.float(), kr) * scale
                    mask = kept.repeat_interleave(m_rows, 1).repeat_interleave(TN, 2)[:, :CL, :keys]
                    ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
                    del sc
                    rec[f'q{m_rows}_keep{keep}'] = dict(
                        kept_frac=round(float(kept.float().mean()), 4),
                        max_abs_err=round(float((got.float() - ref).abs().max()), 5),
                        ms=round(timed(lambda: call(1, L)), 4))
                    torch.cuda.empty_cache()
            except Exception as e:
                rec[f'q{m_rows}_error'] = repr(e)[:300]
        print(json.dumps(rec), flush=True)
        rec_out.write(json.dumps(rec) + '\n'); rec_out.flush()


if __name__ == '__main__':
    main()
