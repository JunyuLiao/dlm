"""Isolate the varlen block-sparse error seen in v27_vllm_varlen_sparse_probe.py: compare against an fp32 masked
reference for (a) fixed-length contiguous, (b) fixed-length paged, (c) varlen contiguous, (d) varlen paged, at 32K keys,
canvas 256, 64-row maps, keep 25%. Also (e) varlen paged with the n-block lists sorted ascending.
usage: python v27_vllm_varlen_sparse_isolate.py [STRIDE_TEST]
With STRIDE_TEST, adds (g): varlen paged lists laid out at a per-q-block stride of ceil(seqlen_k / 128), the
num_n_blocks the SM90 kernel computes when SeqlenInfoQK.create gets its default tile_n=128 (hypothesis: the varlen
offset uses that stride although head_dim 512 runs tile_n 64)."""
import json
import math

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

H, HK, D, CL, TN, MR = 16, 2, 512, 256, 64, 64
g = torch.Generator(device='cuda').manual_seed(3)
keys, page = 32768, 64
q = torch.randn(CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
k = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
v = torch.randn(keys, HK, D, device='cuda', dtype=torch.bfloat16, generator=g)
npages = keys // page
perm = torch.randperm(npages + 7, device='cuda', generator=g)[:npages]
kc = torch.randn(npages + 7, page, HK, D, device='cuda', dtype=torch.bfloat16, generator=g) * 5
vc = torch.randn_like(kc) * 5
kc[perm] = k.view(npages, page, HK, D)
vc[perm] = v.view(npages, page, HK, D)
table = perm.to(torch.int32)[None].contiguous()
m, n = CL // MR, keys // TN
kept = torch.rand(H, m, n, device='cuda', generator=g) < .25
kept[..., 0] = True
kept[..., -4:] = True
scale = D ** -.5
kr = k.transpose(0, 1).repeat_interleave(H // HK, 0).float()
vr = v.transpose(0, 1).repeat_interleave(H // HK, 0).float()
sc = torch.einsum('qhd,hkd->hqk', q.float(), kr) * scale
mask = kept.repeat_interleave(MR, 1).repeat_interleave(TN, 2)
ref = torch.einsum('hqk,hkd->qhd', sc.masked_fill(~mask, float('-inf')).softmax(-1), vr)
dense_ref = torch.einsum('hqk,hkd->qhd', sc.softmax(-1), vr)
print('ref magnitude', float(ref.abs().mean()), 'dense-vs-sparse ref diff', float((ref - dense_ref).abs().max()))


def order_of(kept_, ascending):
    if ascending:
        idx = torch.arange(kept_.shape[-1], device='cuda').expand_as(kept_)
        key = torch.where(kept_, idx, idx + kept_.shape[-1])
        return torch.argsort(key, dim=-1).to(torch.int32)
    return torch.argsort((~kept_).to(torch.int8), dim=-1, stable=True).to(torch.int32)


def fixed_lists(ascending=False):
    kk = kept[None]
    return BST(mask_block_cnt=torch.zeros((1, H, m), device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros((1, H, m, 1), device='cuda', dtype=torch.int32),
               full_block_cnt=kk.sum(-1).to(torch.int32).contiguous(),
               full_block_idx=order_of(kk, ascending).contiguous(), block_size=(MR, TN))


def varlen_lists(ascending=False):
    return BST(mask_block_cnt=torch.zeros((H, m), device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros((H, m * n), device='cuda', dtype=torch.int32),
               full_block_cnt=kept.sum(-1).to(torch.int32).contiguous(),
               full_block_idx=order_of(kept, ascending).reshape(H, m * n).contiguous(),
               cu_total_m_blocks=torch.tensor([0, m], device='cuda', dtype=torch.int32),
               cu_block_idx_offsets=torch.tensor([0, m * n], device='cuda', dtype=torch.int32),
               block_size=(MR, TN))


def varlen_lists_stride(stride):
    order = order_of(kept, False)[..., :stride]
    assert int(kept.sum(-1).max()) <= stride
    return BST(mask_block_cnt=torch.zeros((H, m), device='cuda', dtype=torch.int32),
               mask_block_idx=torch.zeros((H, m * stride), device='cuda', dtype=torch.int32),
               full_block_cnt=kept.sum(-1).to(torch.int32).contiguous(),
               full_block_idx=order.reshape(H, m * stride).contiguous(),
               cu_total_m_blocks=torch.tensor([0, m], device='cuda', dtype=torch.int32),
               cu_block_idx_offsets=torch.tensor([0, m * stride], device='cuda', dtype=torch.int32),
               block_size=(MR, TN))


cu_q = torch.tensor([0, CL], device='cuda', dtype=torch.int32)
used = torch.tensor([keys], device='cuda', dtype=torch.int32)
res = {}
cases = {
    'a_fixed_contig': lambda: fwd(q[None], k[None], v[None], softmax_scale=scale, causal=False,
                                  block_sparse_tensors=fixed_lists())[0][0],
    'b_fixed_paged': lambda: fwd(q[None], kc, vc, softmax_scale=scale, causal=False, page_table=table, seqused_k=used,
                                 block_sparse_tensors=fixed_lists())[0][0],
    'c_varlen_contig': lambda: flash_attn_varlen_func(q, k, v, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=keys,
                                                      cu_seqlens_k=torch.tensor([0, keys], device='cuda', dtype=torch.int32),
                                                      softmax_scale=scale, causal=False, fa_version=4, num_splits=1,
                                                      block_sparse_tensors=varlen_lists()),
    'd_varlen_paged': lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=keys,
                                                     seqused_k=used, softmax_scale=scale, causal=False, block_table=table,
                                                     fa_version=4, num_splits=1, block_sparse_tensors=varlen_lists()),
    'e_varlen_paged_ascending': lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q,
                                                               max_seqlen_k=keys, seqused_k=used, softmax_scale=scale,
                                                               causal=False, block_table=table, fa_version=4,
                                                               num_splits=1, block_sparse_tensors=varlen_lists(True)),
    'f_varlen_paged_dense': lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q,
                                                           max_seqlen_k=keys, seqused_k=used, softmax_scale=scale,
                                                           causal=False, block_table=table, fa_version=4, num_splits=1),
}
import sys
if len(sys.argv) > 1:
    cases = {'g_varlen_paged_stride128': lambda: flash_attn_varlen_func(
        q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=keys, seqused_k=used, softmax_scale=scale,
        causal=False, block_table=table, fa_version=4, num_splits=1,
        block_sparse_tensors=varlen_lists_stride(-(-keys // 128)))}
for name, fn in cases.items():
    try:
        out = fn()
        target = dense_ref if name.startswith('f_') else ref
        err = (out.float() - target).abs()
        res[name] = dict(max=round(float(err.max()), 5), mean=round(float(err.mean()), 6))
    except Exception as e:
        res[name] = repr(e)[:300]
    print(name, res[name], flush=True)
print(json.dumps(res))
