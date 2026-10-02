"""Where does the vLLM adapter's per-step overhead come from? Per GLOBAL call at the 32K/64K GLOBAL geometry, measure
host (enqueue) time and device time of: vLLM's own dense call (flash_attn_varlen_func over the paged cache), the adapter's
canvas refresh (page gather + copy), and the FA4 all-kept call on the contiguous buffer (v27_fa4.dense).
Host time = wall time of N back-to-back calls without synchronization (CPU cost while the GPU is busy);
device time = CUDA events around N calls after warm-up.
usage (in a panel deploy dir, vLLM env): python v27_vllm_adapter_overhead.py OUT_JSON"""
import json
import math
import sys
import time

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

from experiments.numerical_qk_reuse import v27_fa4

H, HK, D, CL, PAGE = 16, 2, 512, 256, 64


def host_and_device(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    host = (time.perf_counter() - t) / n
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return round(host * 1e3, 4), round(a.elapsed_time(b) / n, 4)


def main():
    out = {}
    g = torch.Generator(device='cuda').manual_seed(0)
    for keys in (32768, 65536):
        prefix = keys - CL
        npages = math.ceil(keys / PAGE) + 8
        kv = torch.randn(npages, HK, PAGE, 2 * D, device='cuda', dtype=torch.bfloat16, generator=g)
        key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)       # vLLM's view of its cache
        table = torch.randperm(npages, device='cuda', generator=g)[:math.ceil(keys / PAGE)].to(torch.int32)
        q = torch.randn(CL, H, D, device='cuda', dtype=torch.bfloat16, generator=g)
        out_buf = torch.empty_like(q)
        cu_q = torch.tensor([0, CL], device='cuda', dtype=torch.int32)
        used = torch.tensor([keys], device='cuda', dtype=torch.int32)
        scale = D ** -.5

        def vllm_dense():
            flash_attn_varlen_func(q, key_cache, value_cache, max_seqlen_q=CL, cu_seqlens_q=cu_q, max_seqlen_k=keys,
                                   seqused_k=used, softmax_scale=scale, causal=False, block_table=table[None],
                                   out=out_buf, fa_version=4, num_splits=1)
        kbuf = torch.empty((1, HK, keys, D), device='cuda', dtype=torch.bfloat16)
        vbuf = torch.empty_like(kbuf)
        first, last = prefix // PAGE, (keys - 1) // PAGE
        off = prefix - first * PAGE

        def refresh():
            pages = table[first: last + 1].long()
            kbuf[0, :, prefix:].copy_(key_cache[pages].reshape(-1, HK, D)[off: off + CL].transpose(0, 1))
            vbuf[0, :, prefix:].copy_(value_cache[pages].reshape(-1, HK, D)[off: off + CL].transpose(0, 1))
        qh = q.transpose(0, 1).unsqueeze(0)

        def allkept():
            o = v27_fa4.dense(qh, kbuf, vbuf, scale)
            out_buf.view(CL, -1).copy_(o.reshape(CL, -1))

        def adapter_call():
            refresh()
            allkept()
        out[keys] = {name: dict(zip(('host_ms', 'device_ms'), host_and_device(fn)))
                     for name, fn in (('vllm_dense', vllm_dense), ('refresh', refresh), ('allkept_fa4', allkept),
                                      ('adapter_call', adapter_call))}
        print(keys, out[keys], flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
