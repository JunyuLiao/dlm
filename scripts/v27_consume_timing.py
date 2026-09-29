"""v27 in-model diagnosis: CUDA-event time of every GLOBAL-layer attention call inside real requests,
split by path (dense bootstrap / FA4-or-c64 sparse consume / fused observation / other), plus the kept
fraction of every consumed map and the FA4 block-list cache builds. Uses the arm configs of an
existing direct-cost profile config (.config.json) and one of its targets. Numbers only.
usage: python -m scripts.v27_consume_timing CONFIG_JSON MANIFEST ROW_INDEX BUDGET ARM [ARM ...]
(the row must match the arm configs' thinking mode; BUDGET caps generation to bound the run)
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict


def main(argv=None):
    argv = argv or sys.argv[1:]
    config_path, manifest, row_index, budget, arm_names = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4:]
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import integration
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    config = json.load(open(config_path))
    arms = {a['name']: a for a in config['arms']}
    row = dict(json.load(open(manifest))[row_index], generation_budget=budget)
    adapter = create_adapter('diffusion_gemma', config['model'], device='cuda', precision='bfloat16',
                             revision=config.get('revision')).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    records = defaultdict(list)
    kept_fraction = []

    def timed(name, fn):
        def wrapper(self, *a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(self, *a, **k)
            e.record()
            records[name].append((s, e))
            if name == 'consume':
                skipped, eligible = a[3], a[4]
                kept_fraction.append((eligible & ~skipped).float().mean())
            return out
        return wrapper

    originals = {n: getattr(integration.Attention, n) for n in ('_consume', '_dense_native', '_fused_bootstrap_observation',
                                                                '__call__')}
    integration.Attention._consume = timed('consume', originals['_consume'])
    integration.Attention._dense_native = timed('dense_native', originals['_dense_native'])
    integration.Attention._fused_bootstrap_observation = timed('fused_observation', originals['_fused_bootstrap_observation'])
    integration.Attention.__call__ = timed('global_call_total', originals['__call__'])
    try:
        for name in arm_names:
            records.clear()
            kept_fraction.clear()
            arm = arms[name]
            cfg = dict(arm['config'], condition=arm['condition'], plugin=arm['plugin'])
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                receipt = _one(adapter, row, 101, cfg)
            torch.cuda.synchronize()
            summary = {k: dict(n=len(v), median_ms=round(statistics.median(s.elapsed_time(e) for s, e in v), 4),
                               total_ms=round(sum(s.elapsed_time(e) for s, e in v), 1)) for k, v in records.items()}
            kf = [float(x) for x in kept_fraction]
            print(json.dumps(dict(arm=name, prompt_tokens=row['prompt_token_count'], calls=receipt['total_decoder_calls'],
                                  paths=summary,
                                  kept_fraction_median=round(statistics.median(kf), 4) if kf else None,
                                  fa4_list_builds=(receipt.get('counters') or {}).get('fa4_list_builds'))), flush=True)
    finally:
        for n, f in originals.items():
            setattr(integration.Attention, n, f)


if __name__ == '__main__':
    main()
