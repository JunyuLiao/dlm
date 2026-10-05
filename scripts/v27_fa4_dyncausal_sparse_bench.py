"""vLLM's dense GLOBAL call is fast because it runs FA4's dynamic-causal path (causal=True + per-request
dynamic_causal=False) with num_splits=0: 0.95 ms vs 1.65 ms for the plain non-causal call at 35K keys, same output.
Does the same path work with block sparsity (our consumer), and how fast is it? Fixed-length API, contiguous K/V,
batch 1, per-q-head 128x64 lists at keep 100/25/12%; each output checked against an fp32 masked bidirectional reference
(relative error reported). usage: python v27_fa4_dyncausal_sparse_bench.py OUT_JSONL"""
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


def lists(kept):
    kk = kept[None]
    order = torch.argsort((~kk).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return BST(mask_block_cnt=torch.zeros((1, H, kept.shape[1]), device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros((1, H, kept.shape[1], 1), device='cuda', dtype=torch.int32),
               full_block_cnt=kk.sum(-1).to(torch.int32).contiguous(), full_block_idx=order.contiguous(),
               block_size=(MR, TN))


def main():
    out = open(sys.argv[1], 'w')
    g = torch.Generator(device='cuda').manual_seed(17)
    scale = 1.0 / math.sqrt(D)
    dyn = torch.tensor([False], device='cuda')
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
        kr = k[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        vr = v[0].transpose(0, 1).repeat_interleave(H // HK, 0).float()
        sc = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
        rec = dict(keys=keys)
        dense_ref = torch.einsum('hqk,hkd->qhd', sc.softmax(-1), vr)
        for name, kw in (('plain_s1', dict(causal=False, num_splits=1)), ('plain_s0', dict(causal=False, num_splits=0)),
                         ('dyn_s0', dict(causal=True, dynamic_causal=dyn, num_splits=0)),
                         ('dyn_s1', dict(causal=True, dynamic_causal=dyn, num_splits=1))):
            try:
                o = fwd(q, k, v, softmax_scale=scale, **kw)[0]
                rel = float((o[0].float() - dense_ref).abs().max() / dense_ref.abs().max())
                rec['dense_' + name] = dict(ms=timed(lambda: fwd(q, k, v, softmax_scale=scale, **kw)), rel_err=round(rel, 5))
            except Exception as e:
                rec['dense_' + name] = repr(e)[:200]
        n = keys // TN
        for keep in (1.0, 0.25, 0.12):
            kept = torch.rand(H, CL // MR, n, device='cuda', generator=g) < keep
            kept[..., 0] = True
            kept[..., -(CL // TN):] = True
            L = lists(kept)
            mask = kept.repeat_interleave(MR, 1).repeat_interleave(TN, 2)
            ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
            for name, kw in (('plain_s1', dict(causal=False, num_splits=1)),
                             ('dyn_s0', dict(causal=True, dynamic_causal=dyn, num_splits=0)),
                             ('dyn_s1', dict(causal=True, dynamic_causal=dyn, num_splits=1))):
                key = f'sparse_keep{keep}_{name}'
                try:
                    o = fwd(q, k, v, softmax_scale=scale, block_sparse_tensors=L, **kw)[0]
                    rel = float((o[0].float() - ref).abs().max() / ref.abs().max())
                    rec[key] = dict(ms=timed(lambda: fwd(q, k, v, softmax_scale=scale, block_sparse_tensors=L, **kw)),
                                    rel_err=round(rel, 5))
                except Exception as e:
                    rec[key] = repr(e)[:200]
            del mask, ref
            torch.cuda.empty_cache()
        del sc, kr, vr, dense_ref
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
