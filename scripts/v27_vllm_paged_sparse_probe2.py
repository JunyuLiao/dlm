import math, torch, json
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch as BST
g = torch.Generator(device='cuda').manual_seed(1)
def lists(kept, last_masked):
    b, h, qb, kt = kept.shape
    full = kept.clone()
    mcnt = torch.zeros((b, h, qb), device='cuda', dtype=torch.int32)
    midx = torch.zeros((b, h, qb, 1), device='cuda', dtype=torch.int32)
    if last_masked:
        full[..., -1] = False
        mcnt[:] = kept[..., -1].to(torch.int32); midx[..., 0] = kt - 1
    order = torch.argsort((~full).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return BST(mask_block_cnt=mcnt, mask_block_idx=midx, full_block_cnt=full.sum(-1).to(torch.int32).contiguous(),
               full_block_idx=order.contiguous(), block_size=(64, 64))
for keys, last_masked in ((65536, False), (65000, True), (65000, False)):
    nq, h, hk, d, page = 256, 16, 2, 512, 64
    q = torch.randn(1, nq, h, d, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g)
    npages = math.ceil(keys / page); perm = torch.randperm(npages + 5, device='cuda', generator=g)[:npages]
    kc = torch.randn(npages + 5, page, hk, d, device='cuda', dtype=torch.bfloat16, generator=g) * 5
    vc = torch.randn_like(kc) * 5
    pad = npages * page - keys
    kc[perm] = torch.cat((k[0], kc.new_zeros(pad, hk, d))).view(npages, page, hk, d)
    vc[perm] = torch.cat((v[0], vc.new_zeros(pad, hk, d))).view(npages, page, hk, d)
    if pad:
        lastp = perm[-1]; kc[lastp, page - pad:] = torch.randn(pad, hk, d, device='cuda', dtype=torch.bfloat16, generator=g) * 5
    table = perm.to(torch.int32)[None].contiguous(); used = torch.tensor([keys], device='cuda', dtype=torch.int32)
    kt = math.ceil(keys / 64)
    kept = torch.rand(1, h, nq // 64, kt, device='cuda', generator=g) < .09; kept[..., 0] = True; kept[..., -4:] = True
    L = lists(kept, last_masked)
    scale = d ** -.5
    try:
        ref = fwd(q, k, v, softmax_scale=scale, causal=False, block_sparse_tensors=L)[0]
        got = fwd(q, kc, vc, softmax_scale=scale, causal=False, page_table=table, seqused_k=used, block_sparse_tensors=L)[0]
        kr = k[0].transpose(0, 1).repeat_interleave(8, 0).float(); vr = v[0].transpose(0, 1).repeat_interleave(8, 0).float()
        s = torch.einsum('qhd,hkd->hqk', q[0].float(), kr) * scale
        m = kept[0].repeat_interleave(64, 1).repeat_interleave(64, 2)[:, :nq, :keys]
        exact = torch.einsum('hqk,hkd->qhd', s.masked_fill(~m, float('-inf')).softmax(-1), vr)
        print(json.dumps(dict(keys=keys, last_masked=last_masked, contig_vs_fp32=float((ref[0].float() - exact).abs().max()),
                              paged_vs_fp32=float((got[0].float() - exact).abs().max()))), flush=True)
    except Exception as e:
        print(json.dumps(dict(keys=keys, last_masked=last_masked, error=repr(e)[:400])), flush=True)
