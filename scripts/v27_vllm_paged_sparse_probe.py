"""Feasibility probe for porting the method into vLLM (no model, random data): does vLLM 0.30.0's FA4 (CuTe, SM90,
head_dim 512) run block-sparse lists over a PAGED KV cache (vLLM's storage) with the same result as over contiguous K/V,
and how fast? GLOBAL decode geometry: 256 canvas queries, 16 Q heads, 2 KV heads, bidirectional, bf16, 64-row query
tiles x 64-key tiles. Pages are physically shuffled through the page table.
usage: python v27_vllm_paged_sparse_probe.py OUT.json
"""
import json
import math
import statistics
import sys

import torch


def lists_from(kept, bst_cls):
    b, h, qb, kt = kept.shape
    order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    zeros = torch.zeros((b, h, qb), device=kept.device, dtype=torch.int32)
    return bst_cls(mask_block_cnt=zeros, mask_block_idx=torch.zeros((b, h, qb, 1), device=kept.device, dtype=torch.int32),
                   full_block_cnt=kept.sum(-1).to(torch.int32).contiguous(), full_block_idx=order.contiguous(),
                   block_size=(64, 64))


def timed(fn, warm=3, reps=20):
    for _ in range(warm):
        fn()
    ev = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); ev.append((a, b))
    torch.cuda.synchronize()
    return round(statistics.median(a.elapsed_time(b) for a, b in ev), 4)


def main():
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd
    from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
    rows = []
    g = torch.Generator(device='cuda').manual_seed(0)
    for keys, page, keep in ((33000, 16, .18), (65000, 16, .09), (65000, 64, .09), (97000, 16, .06), (4133, 16, .5)):
        nq, h, hk, d = 256, 16, 2, 512
        q = torch.randn(1, nq, h, d, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g)
        npages = math.ceil(keys / page)
        perm = torch.randperm(npages + 7, device='cuda', generator=g)[:npages]          # physical page of each logical page
        kc = torch.zeros(npages + 7, page, hk, d, device='cuda', dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        pad = npages * page - keys
        kp = torch.cat((k[0], k.new_zeros(pad, hk, d))).view(npages, page, hk, d)
        vp = torch.cat((v[0], v.new_zeros(pad, hk, d))).view(npages, page, hk, d)
        kc[perm], vc[perm] = kp, vp
        table = perm.to(torch.int32)[None].contiguous()
        used = torch.tensor([keys], device='cuda', dtype=torch.int32)
        kt = math.ceil(keys / 64)
        kept = torch.rand(1, h, nq // 64, kt, device='cuda', generator=g) < keep
        kept[..., 0] = True
        kept[..., -4:] = True
        sparse = lists_from(kept, BST)
        allk = lists_from(torch.ones_like(kept), BST)
        scale = d ** -.5
        row = dict(keys=keys, page_size=page, keep=keep, kept_fraction=round(float(kept.float().mean()), 4))
        try:
            ref = fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=sparse)[0]
            got = fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table, seqused_k=used,
                      block_sparse_tensors=sparse)[0]
            dense_ref = fwd(q, k, v, softmax_scale=scale, causal=False)[0]
            dense_paged = fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table, seqused_k=used)[0]
            row.update(max_abs_paged_vs_contig_sparse=float((got.float() - ref.float()).abs().max()),
                       max_abs_paged_vs_contig_dense=float((dense_paged.float() - dense_ref.float()).abs().max()))
            # exact masked reference (fp32) for the sparse map
            kr = k[0].transpose(0, 1).repeat_interleave(h // hk, 0).float()             # [H,K,D]
            vr = v[0].transpose(0, 1).repeat_interleave(h // hk, 0).float()
            s = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
            m = kept[0].repeat_interleave(64, 1).repeat_interleave(64, 2)[:, :nq, :keys]
            exact = torch.einsum('hqk,hkd->qhd', s.masked_fill(~m, float('-inf')).softmax(-1), vr)
            row['max_abs_paged_sparse_vs_fp32'] = float((got[0].float() - exact).abs().max())
            row['ms'] = dict(
                contig_dense=timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False)),
                contig_allkept=timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=allk)),
                contig_sparse=timed(lambda: fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=sparse)),
                paged_dense=timed(lambda: fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table, seqused_k=used)),
                paged_allkept=timed(lambda: fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table,
                                                seqused_k=used, block_sparse_tensors=allk)),
                paged_sparse=timed(lambda: fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table,
                                               seqused_k=used, block_sparse_tensors=sparse)))
        except Exception as e:      # record, do not hide
            row['error'] = repr(e)[:600]
        rows.append(row)
        print(json.dumps(row), flush=True)
        del q, k, v, kc, vc
        torch.cuda.empty_cache()
    import vllm
    json.dump(dict(vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(), rows=rows),
              open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
