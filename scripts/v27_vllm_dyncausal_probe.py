"""Why is vLLM's in-engine GLOBAL dense call ~1.0 ms at 35K keys while the same FA4 dense call measured standalone takes
~1.5 ms? vLLM passes causal=True with a per-request dynamic_causal tensor (False for denoising requests),
max_seqlen_k = max_model_len and num_splits=0. Replicate that call standalone (paged 64, varlen, scale 1.0) and vary
one argument at a time; check each output against a bidirectional fp32 reference.
usage: python v27_vllm_dyncausal_probe.py"""
import json
import math

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

H, HK, D, CL, PAGE = 16, 2, 512, 256, 64


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


g = torch.Generator(device='cuda').manual_seed(5)
keys, max_len = 35135, 40960
npages = max_len // PAGE
kv = torch.randn(npages + 8, HK, PAGE, 2 * D, device='cuda', dtype=torch.bfloat16, generator=g) * .1
kc, vc = kv.transpose(1, 2).split(D, dim=-1)
table = torch.randperm(npages + 8, device='cuda', generator=g)[:npages].to(torch.int32)[None].contiguous()
q = torch.randn(CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g) * .1
out = torch.empty_like(q)
cu_q = torch.tensor([0, CL], device='cuda', dtype=torch.int32)
used = torch.tensor([keys], device='cuda', dtype=torch.int32)
dyn = torch.tensor([False], device='cuda')
desc = torch.ones(1, HK, device='cuda', dtype=torch.float32)
K = kc[table[0].long()].reshape(-1, HK, D)[:keys]
V = vc[table[0].long()].reshape(-1, HK, D)[:keys]
kr = K.transpose(0, 1).repeat_interleave(H // HK, 0).float(); vr = V.transpose(0, 1).repeat_interleave(H // HK, 0).float()
ref = torch.einsum('hqk,hkd->qhd', (torch.einsum('qhd,hkd->hqk', q.float(), kr) * 1.0).softmax(-1), vr)


def call(causal, dynamic, splits, mk, descale):
    def f():
        flash_attn_varlen_func(q, kc, vc, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=mk, seqused_k=used,
                               softmax_scale=1.0, causal=causal, block_table=table, out=out, fa_version=4,
                               num_splits=splits, dynamic_causal=dyn if dynamic else None,
                               q_descale=desc if descale else None, k_descale=desc if descale else None,
                               v_descale=desc if descale else None)
    return f


res = {}
for name, args in {'vllm_exact': (True, True, 0, max_len, True), 'vllm_exact_s1': (True, True, 1, max_len, True),
                   'noncausal_s0_maxlen': (False, False, 0, max_len, True), 'noncausal_s0_keys': (False, False, 0, keys, False),
                   'noncausal_s1_keys': (False, False, 1, keys, False), 'dyn_s0_keys_nodesc': (True, True, 0, keys, False)}.items():
    try:
        f = call(*args)
        f(); torch.cuda.synchronize()
        err = float((out.float() - ref).abs().max())
        res[name] = dict(ms=timed(f), max_err=round(err, 5))
    except Exception as e:
        res[name] = repr(e)[:200]
    print(name, res[name], flush=True)
print(json.dumps(dict(ref_mag=float(ref.abs().mean()), **res)))
