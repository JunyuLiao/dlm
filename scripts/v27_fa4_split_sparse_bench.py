"""Does FA4 block sparsity work with split-KV, and what does it buy at the GLOBAL decode geometry?

vLLM's dense GLOBAL call lets FA4 pick num_splits (heuristic min(SMs // m_blocks, 128, n_blocks); with pack-GQA,
256 queries x 8 heads -> 64 m-blocks -> 2 splits) and is ~1.8x faster than num_splits=1 at 35K keys. Our dense
control and sparse consumer used num_splits=1. Here, per keys in (32K, 64K, 94K), on one contiguous K/V (fixed-length
API, batch 1, paged=False and page 64 paged):
  dense_auto      : FA4 dense, num_splits=0 (heuristic), pack-GQA default        [= vLLM's choice]
  dense_s1        : FA4 dense, num_splits=1
  sparse_sS       : FA4 block-sparse (per-q-head 128x64 lists = our consumer) at keep 100% / 25% / 12%, num_splits S
Each sparse output is checked against an fp32 masked reference. Median of 30 CUDA-event timings.
usage: python v27_fa4_split_sparse_bench.py OUT_JSONL"""
import json
import math
import sys

import torch
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

H, HK, D, CL, TN = 16, 2, 512, 256, 64


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


def lists(kept, mr):
    kk = kept[None]
    m = kept.shape[1]
    order = torch.argsort((~kk).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return BST(mask_block_cnt=torch.zeros((1, H, m), device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros((1, H, m, 1), device='cuda', dtype=torch.int32),
               full_block_cnt=kk.sum(-1).to(torch.int32).contiguous(), full_block_idx=order.contiguous(),
               block_size=(mr, TN))


def main():
    out = open(sys.argv[1], 'w')
    g = torch.Generator(device='cuda').manual_seed(11)
    scale = 1.0 / math.sqrt(D)
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        rec = dict(keys=keys)
        rec['dense_auto'] = timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False, num_splits=0))
        rec['dense_s1'] = timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False, num_splits=1))
        rec['dense_s2'] = timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False, num_splits=2))
        n = keys // TN
        kr = k[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        vr = v[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        sc = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
        for mr in (128,):
            m = CL // mr
            for keep in (1.0, 0.25, 0.12):
                kept = torch.rand(H, m, n, device='cuda', generator=g) < keep
                kept[..., 0] = True
                kept[..., -(CL // TN):] = True
                L = lists(kept, mr)
                mask = kept.repeat_interleave(mr, 1).repeat_interleave(TN, 2)
                ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
                for s in (1, 2, 4, 8, 0):
                    key = f'sparse_q{mr}_keep{keep}_s{s}'
                    try:
                        o = fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=L, num_splits=s)[0]
                        err = float((o[0].float() - ref).abs().max())
                        rec[key] = dict(ms=timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False,
                                                             block_sparse_tensors=L, num_splits=s)), max_err=round(err, 5))
                    except Exception as e:
                        rec[key] = repr(e)[:200]
                del mask, ref
                torch.cuda.empty_cache()
        del sc, kr, vr
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
