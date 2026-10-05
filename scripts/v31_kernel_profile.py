"""CUDA-kernel time breakdown of one seed-paired request (diagnostic, CUPTI via torch.profiler; no added syncs).

Wraps scripts/v31_vllm_paired_bench.py without changing it. The warm-up request runs unprofiled; the first measured
request runs under torch.profiler (CUDA + CPU activities). Writes one JSON line: denoising forwards of that request,
profiled wall span, total CUDA kernel time, and the kernels aggregated by name (count, total ms), so per-forward GPU
time can be split into dense / sparse attention, observation, selection, copies and the rest of the model.
Usage: same arguments and env as v31_vllm_paired_bench.py with LIMIT=1, plus KPROF_OUT=<jsonl>.
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_vllm_paired_bench as bench  # noqa: E402


def main():
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm.v1.engine.llm_engine import LLMEngine
    out = open(os.environ['KPROF_OUT'], 'a', encoding='utf-8')
    state = dict(requests=0, prof=None, t0=None)

    def finish():
        p = state['prof']
        if p is None:
            return
        torch.cuda.synchronize()
        wall = time.perf_counter() - state['t0']
        p.__exit__(None, None, None)
        state['prof'] = None
        agg = collections.defaultdict(lambda: [0, 0.0])
        for e in p.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                a = agg[e.name[:160]]
                a[0] += 1
                a[1] += e.device_time / 1000.0
        kernels = sorted(([k, v[0], round(v[1], 3)] for k, v in agg.items()), key=lambda x: -x[2])
        rec = dict(wall_ms=round(1000 * wall, 2), cuda_ms=round(sum(k[2] for k in kernels), 2),
                   sampler_calls=state.get('calls'), kernels=kernels[:120])
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(json.dumps(dict(wall_ms=rec['wall_ms'], cuda_ms=rec['cuda_ms'], top=kernels[:12])), flush=True)

    inner_add = LLMEngine.add_request

    def add_request(self, *a, **k):
        finish()
        state['requests'] += 1
        r = inner_add(self, *a, **k)
        if state['requests'] == 2:                      # first measured request (the first one is the warm-up)
            state['prof'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            state['prof'].__enter__()
            torch.cuda.synchronize()
            state['t0'] = time.perf_counter()
        return r
    LLMEngine.add_request = add_request
    try:
        bench.main()
    finally:
        finish()


if __name__ == '__main__':
    main()
