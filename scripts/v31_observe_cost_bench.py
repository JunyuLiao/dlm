"""Where does the method's observation call spend its time? (GLOBAL geometry, synthetic tensors)

Per keys in (32K, 64K, 94K), median CUDA-event time of one GLOBAL-layer call:
  vllm_dense       FA4 dynamic-causal + auto split (vLLM's own dense call), the reference
  fused_mu         the method's fused observation (dense output + prefix z + rank-32 projected-V mu + tail), bf16 mu
  fused_nomu       the same kernel without the projected-V mu (what a mass-only risk would need)
  observe_only_mu  OUT=0 (no PV/output), with mu
Run in a panel deployment (PYTHONPATH=src:.) with the vLLM env.
usage: python v31_observe_cost_bench.py OUT_JSONL"""
import json
import math
import sys

import torch
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as fwd

from experiments.numerical_qk_reuse.cached_executor import allocate_summary
from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe

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
    g = torch.Generator(device='cuda').manual_seed(41)
    dyn = torch.tensor([False], device='cuda')
    for keys in (32768, 65536, 94208):
        q = torch.randn(1, H, CL, D, device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, HK, keys, D, device='cuda', dtype=torch.bfloat16, generator=g)
        sketch = torch.randn(1, HK, keys, 32, device='cuda', dtype=torch.float32, generator=g)
        pt = (keys - CL) // 64
        scale = 1.0
        rec = dict(keys=keys)
        qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        rec['vllm_dense'] = timed(lambda: fwd(qs, ks, vs, softmax_scale=scale, causal=True, dynamic_causal=dyn,
                                              num_splits=0))
        for name, kw in (('fused_mu', dict(mu=True)), ('fused_nomu', dict(mu=False)),
                         ('observe_only_mu', dict(mu=True, output=False))):
            summary = allocate_summary(1, H, CL // 128, math.ceil(keys / 64), pt, 32, 'cuda', ('v31',))
            rec[name] = timed(lambda: fused_observe(q, k, v, sketch, scale, pt, summary, splits=2, mu_precision='bf16',
                                                    **kw))
            del summary
            torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()


if __name__ == '__main__':
    main()
