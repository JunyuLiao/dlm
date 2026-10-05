"""v27: where the time of the M1/M2/M3 bootstrap-observation and decision calls goes, inside real requests of a
BOUND panel on the piecewise substrate. Wraps (CUDA events + host time) the observation call and its parts:
sketch projection (Sketches.get), summary allocation, the fused observation kernel, the route (selector), score /
decision publication, plus the held consume and the bootstrap-dense call. Numbers only.
usage: python -m scripts.v27_observe_profile --run-dir DIR --host IP --gpu-uuid UUID --stage S --dataset D
           --index I --budget N --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import cached_executor, integration, v27_consumer64
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    substrate = protocol.get('substrate', 'eager')
    if substrate != 'eager':
        from experiments.numerical_qk_reuse import v27_substrate
        v27_substrate.install(adapter.model)
    records = defaultdict(list)

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            records[name].append((s, e, (time.perf_counter() - t) * 1e3))
            return out
        return wrapper
    patches = [(integration.Attention, '_fused_bootstrap_observation'), (integration.Attention, '_bootstrap_call'),
               (integration.Attention, '_route'), (integration.Attention, '_route_dense_prefix'), (integration.Attention, '_consume'),
               (integration.Attention, '_dense_native'), (integration.Attention, '__call__'),
               (integration, 'allocate_summary'), (v27_consumer64, 'fused_observe')]
    for owner_name in ('Sketches', 'ScoreCache'):
        owner = getattr(integration, owner_name, None) or getattr(cached_executor, owner_name, None)
        if owner is not None:
            for method in ('get', 'publish_scores', 'publish_decision'):
                if hasattr(owner, method):
                    patches.append((owner, method))
    originals = [(obj, name, getattr(obj, name)) for obj, name in patches]
    for obj, name, fn in originals:
        setattr(obj, name, timed(f'{getattr(obj, "__name__", obj)}.{name}', fn))
    row = dict(rows[protocol['ids'][a.dataset][a.index]], generation_budget=a.budget)
    try:
        for arm in a.arm:
            config = configs[a.dataset][arm]
            if substrate != 'eager':
                v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(config))
            for attempt in range(2):            # the first run compiles / captures graphs
                records.clear()
                with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                    receipt = _one(adapter, row, protocol['seeds'][0], config)
                torch.cuda.synchronize()
            summary = {k: dict(n=len(v), event_median_ms=round(statistics.median(s.elapsed_time(e) for s, e, _ in v), 3),
                               host_median_ms=round(statistics.median(h for _, _, h in v), 3),
                               event_total_ms=round(sum(s.elapsed_time(e) for s, e, _ in v), 1))
                       for k, v in sorted(records.items())}
            print(json.dumps(dict(arm=arm, calls=receipt['total_decoder_calls'], prompt_tokens=row.get('prompt_token_count'),
                                  parts=summary)), flush=True)
    finally:
        for obj, name, fn in originals:
            setattr(obj, name, fn)


if __name__ == '__main__':
    main()
