"""v31 observation-path kernel bench (GLOBAL geometry, synthetic tensors, CUDA-event medians in ms).

Per key count (32K / 64K / 94K):
  vllm_dense          FA4 dense, vLLM's call form (causal kernel, dynamic causal off, auto split)
  fa4_observe         the same FA4 call plus the in-kernel prefix log-mass (v31_fa4_observe.observe_dense)
  triton_obs_mu       v27 fused observation with the rank-32 projected V (what main runs)
  triton_obs_nomu     the same without mu (compact pooled mu)
  dp_build_exact      v27 sequential dense-prefix build, exact mu       dp_chunk_exact    v31 chunked build
  dp_build_compact    v27 sequential build, compact pooled mu           dp_chunk_compact  v31 chunked build
usage: python v31_kernel_bench.py OUT_JSONL   (deployment cwd, PYTHONPATH=src:., V27_ADAPTER_DIR=<overlay with the v31 files>)
"""
import json
import math
import os
import sys

import torch
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

import experiments.numerical_qk_reuse as _pkg
if os.environ.get('V27_ADAPTER_DIR'):
    _pkg.__path__.insert(0, os.environ['V27_ADAPTER_DIR'])
from experiments.numerical_qk_reuse import v27_dense_prefix as dp  # noqa: E402
from experiments.numerical_qk_reuse.cached_executor import allocate_summary
from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
from experiments.numerical_qk_reuse.v31_dp_chunked import build_chunked
from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense

H, HK, D, CL = 16, 2, 512, 256


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(sorted(ts)[reps // 2], 4)


def main():
    out = open(sys.argv[1], 'w')
    g = torch.Generator(device='cuda').manual_seed(43)
    dyn = torch.tensor([False], device='cuda')
    scale = 1 / math.sqrt(D)
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, H, CL, D, device='cuda', generator=g).to(torch.bfloat16)
        k = torch.randn(1, HK, keys, D, device='cuda', generator=g).to(torch.bfloat16)
        v = torch.randn(1, HK, keys, D, device='cuda', generator=g).to(torch.bfloat16)
        sketch = torch.randn(1, HK, keys, 32, device='cuda', generator=g)
        pt = (keys - CL) // 64
        kt = math.ceil(keys / 64)
        qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        rec = dict(keys=keys, prefix_tiles=pt)
        rec['vllm_dense'] = timed(lambda: fwd(qs, ks, vs, softmax_scale=scale, causal=True, dynamic_causal=dyn,
                                              num_splits=0))
        z = torch.empty((H, 2, pt, 128), device='cuda')
        rec['fa4_observe'] = timed(lambda: observe_dense(qs, ks, vs, scale, z))
        for name, mu in (('triton_obs_mu', True), ('triton_obs_nomu', False)):
            s = allocate_summary(1, H, 2, kt, pt, 32 if mu else 0, 'cuda', ('b',))
            rec[name] = timed(lambda: fused_observe(q, k, v, sketch, scale, pt, s, splits=2, mu=mu,
                                                    mu_precision='bf16'))
            if mu:
                exact = s
            else:
                compact = s
        exact.active.fill_(1); exact.bad.zero_(); compact.active.fill_(1); compact.bad.zero_()
        pooled = torch.randn(1, HK, kt, 32, device='cuda', generator=g)
        rec['dp_build_exact'] = timed(lambda: dp.build(exact, CL, kt, HK), reps=10)
        rec['dp_chunk_exact'] = timed(lambda: build_chunked(exact, CL, kt, HK), reps=10)
        rec['dp_build_compact'] = timed(lambda: dp.build(compact, CL, kt, HK, pooled=pooled), reps=10)
        rec['dp_chunk_compact'] = timed(lambda: build_chunked(compact, CL, kt, HK, pooled=pooled), reps=10)
        del exact, compact
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
