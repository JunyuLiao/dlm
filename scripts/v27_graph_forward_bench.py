"""v27 substrate diagnostic: one decoder forward on a common real state -- eager vs CUDA graph, GLOBAL attention
FA4 dense vs the same FA4 kernel through its block-sparse interface at fixed keep fractions.

Runs the D_fa4 control arm of a v27 profile config on one manifest row (native adaptive request). At the listed
decoder-call indices the exact call arguments are reused for the measurements, then the request continues with
its normal dense call. Sparse rows are TIMING ONLY: synthetic keep maps of the stated density (seeded; the first
KV tile and the canvas tiles are always kept); their outputs are discarded and nothing about quality follows.
Per row: eager_* (CUDA-event span of one eager forward after a sync), host_* (CPU time to issue it),
graph_* (one CUDA-graph replay of the same forward).
usage: python v27_graph_forward_bench.py CONFIG_JSON MANIFEST ROW_INDEX BUDGET CALL_INDICES KEEPS
       e.g. ... 2 512 3,14 1.0,0.3,0.1
"""
from __future__ import annotations

import json
import statistics
import sys
import time


def main(argv=None):
    argv = argv or sys.argv[1:]
    config_path, manifest, row_index, budget = argv[0], argv[1], int(argv[2]), int(argv[3])
    call_indices = {int(x) for x in argv[4].split(',')}
    keeps = [float(x) for x in argv[5].split(',')]
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    import os
    config = json.load(open(config_path))
    arm = {a['name']: a for a in config['arms']}['D_fa4']
    row = dict(json.load(open(manifest))[row_index], generation_budget=budget)
    adapter = create_adapter('diffusion_gemma', config['model'], device='cuda', precision='bfloat16',
                             revision=config.get('revision')).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    dense_original = v27_fa4.dense
    mode = dict(keep=None, keys=None)
    lists = {}

    def attention(q, k, v, scale):
        mode['keys'] = k.shape[2]
        if mode['keep'] is None:
            return dense_original(q, k, v, scale)
        kt, qb = -(-k.shape[2] // 64), -(-q.shape[2] // 128)
        key = (q.shape[1], qb, kt, mode['keep'])
        if key not in lists:
            g = torch.Generator(device=q.device).manual_seed(0)
            kept = torch.rand(1, q.shape[1], qb, kt, device=q.device, generator=g) < mode['keep']
            kept[..., 0] = True
            canvas_tiles = -(-q.shape[2] // 64)
            kept[..., kt - canvas_tiles:] = True   # the canvas's own keys are the last tiles
            lists[key] = (v27_fa4.block_sparse_tensors(kept), float(kept.float().mean()))
        return v27_fa4.sparse_lists(q, k, v, lists[key][0], scale)

    def eager(fn, n=10):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        span, host = [], []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            s.record()
            fn()
            e.record()
            host.append((time.perf_counter() - t) * 1e3)
            torch.cuda.synchronize()
            span.append(s.elapsed_time(e))
        return round(statistics.median(span), 3), round(statistics.median(host), 3)

    def graph(fn, n=20):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        del g
        torch.cuda.synchronize()
        return round(statistics.median(times), 3)

    model_forward = type(adapter.model).forward
    calls = [0]
    results = []

    def forward(self, *a, **k):
        index = calls[0]
        calls[0] += 1
        if index in call_indices:
            fn = lambda: model_forward(self, *a, **k)
            out = dict(call_index=index)
            for keep in [None] + keeps:
                mode['keep'] = keep
                label = 'dense' if keep is None else f'sparse{keep}'
                try:
                    out[f'eager_{label}'], out[f'host_{label}'] = eager(fn)
                    out[f'graph_{label}'] = graph(fn)
                except Exception as exc:   # record, keep going
                    out[f'error_{label}'] = f'{type(exc).__name__}: {str(exc)[:300]}'
                if keep is not None:
                    match = [v[1] for kk, v in lists.items() if kk[-1] == keep]
                    out[f'kept_{label}'] = round(match[-1], 4) if match else None
            mode['keep'] = None
            out['keys'] = mode['keys']
            results.append(out)
            print(json.dumps(out), flush=True)
        return model_forward(self, *a, **k)

    v27_fa4.dense = attention
    type(adapter.model).forward = forward
    try:
        cfg = dict(arm['config'], condition=arm['condition'], plugin=arm['plugin'])
        cfg.setdefault('fingerprint', 'diagnostic_control_unfingerprinted')
        with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
            receipt = _one(adapter, row, 101, cfg)
        print(json.dumps(dict(done=True, torch=torch.__version__, prompt_tokens=row['prompt_token_count'],
                              calls=receipt['total_decoder_calls'], rows=len(results))), flush=True)
    finally:
        type(adapter.model).forward = model_forward
        v27_fa4.dense = dense_original


if __name__ == '__main__':
    main()
