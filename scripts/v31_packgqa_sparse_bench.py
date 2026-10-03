"""Is GQA packing a lever for the sparse consumer? (regroup direction: share K/V tiles across the 8 query heads)

Our block-sparse consumer gives FA4 per-query-head keep lists, and FA4 then disables pack-GQA (interface: pack_gqa
requires the sparse head dimension to be 1), so each K/V tile is loaded once per query head instead of once per KV
head. Measured here at the GLOBAL decode geometry (256 queries, 16 q / 2 kv heads, head_dim 512), contiguous K/V,
fixed-length API, keep fraction f of 64-key tiles:
  dense_pack / dense_nopack                 FA4 dense with pack_gqa True / False
  sparse_perhead_f                          per-q-head lists (pack off, as now), random pattern, kept fraction f
  sparse_shared_pack_f / sparse_shared_nopack_f   ONE list shared by all heads (pack on / off), same kept fraction
Timing only (CUDA events, median of 30, each with 2-way page-alias split variants where noted) plus an exactness check
against an fp32 masked reference for the shared/pack case.
usage: python v31_packgqa_sparse_bench.py OUT_JSONL"""
import json
import math
import sys

import torch
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

H, HK, D, CL, TN, MR = 16, 2, 512, 256, 64, 128


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


def lists(kept):          # kept [Hl, M, N] bool -> 4D lists [1, Hl, M, N]
    kk = kept[None]
    order = torch.argsort((~kk).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return BST(mask_block_cnt=torch.zeros(kk.shape[:3], device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros(kk.shape[:3] + (1,), device='cuda', dtype=torch.int32),
               full_block_cnt=kk.sum(-1).to(torch.int32).contiguous(), full_block_idx=order.contiguous(),
               block_size=(MR, TN))


def main():
    out = open(sys.argv[1], 'w')
    g = torch.Generator(device='cuda').manual_seed(29)
    scale = D ** -.5
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        n, m = keys // TN, CL // MR
        rec = dict(keys=keys)
        for pk in (True, False):
            rec[f'dense_{"pack" if pk else "nopack"}'] = timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False,
                                                                           pack_gqa=pk, num_splits=1))
        for f in (0.25, 0.12):
            per = torch.rand(H, m, n, device='cuda', generator=g) < f
            per[..., 0] = True; per[..., -4:] = True
            Lp = lists(per)
            rec[f'sparse_perhead_{f}'] = timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False,
                                                           block_sparse_tensors=Lp, num_splits=1))
            sh = torch.rand(1, m, n, device='cuda', generator=g) < f
            sh[..., 0] = True; sh[..., -4:] = True
            Ls = lists(sh)
            for pk in (True, False):
                try:
                    rec[f'sparse_shared_{"pack" if pk else "nopack"}_{f}'] = timed(
                        lambda: fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=Ls, pack_gqa=pk,
                                    num_splits=1))
                except Exception as e:
                    rec[f'sparse_shared_{"pack" if pk else "nopack"}_{f}'] = repr(e)[:160]
            # exactness of the shared + pack path
            o = fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=Ls, pack_gqa=True, num_splits=1)[0][0]
            kr = k[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
            vr = v[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
            mask = sh.expand(H, -1, -1).repeat_interleave(MR, 1).repeat_interleave(TN, 2)
            sc = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
            ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
            rec[f'shared_pack_rel_err_{f}'] = round(float((o.float() - ref).abs().max() / ref.abs().max()), 5)
            del kr, vr, mask, sc, ref
            torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
