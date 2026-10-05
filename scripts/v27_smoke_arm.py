"""v27 pre-launch smoke: one real request per named arm of a BOUND panel (GPU).

Loads the run directory exactly as the panel worker does (validate_inputs), then runs each
requested arm once on the first id of a dataset through the same runner entry and the same
long-context prefill wrapper. Prints only numbers (decoder calls, wall, call counters) and the
error if any; never prints generated text. Writes nothing.

usage: python -m scripts.v27_smoke_arm --run-dir DIR --host IP --gpu-uuid UUID --stage STAGE
           --dataset ruler32k --arm D_c64 [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--arm', action='append', required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--repeat', type=int, default=1, help='runs per arm; with a compiled substrate the second run '
                                                          'must add no graphs')
    p.add_argument('--warm-budget', type=int, default=0, help='first run each arm once with this generation budget '
                                                              '(the runner warm-up), untimed')
    p.add_argument('--warm-index', type=int, default=None, help='warm on this id index (default: --index)')
    a = p.parse_args(argv)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(a.run_dir / 'protocol.json', a.run_dir / 'binding.json',
                                                       a.run_dir / 'manifests', a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    substrate = protocol.get('substrate', 'eager')
    from experiments.numerical_qk_reuse import v27_substrate
    if substrate != 'eager':
        print(json.dumps(dict(substrate_installed=v27_substrate.install(adapter.model, name=substrate))), flush=True)
    row = rows[protocol['ids'][a.dataset][a.index]]
    failures = 0
    if a.warm_budget:
        for arm in a.arm:
            config = configs[a.dataset][arm]
            if substrate != 'eager':
                v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(config))
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1',
                                 kernel=v27_substrate.prefill_kernel(substrate)):
                warm_row = row if a.warm_index is None else rows[protocol['ids'][a.dataset][a.warm_index]]
                _one(adapter, dict(warm_row, generation_budget=a.warm_budget), protocol['seeds'][0], config)
            print(json.dumps(dict(warm=arm, budget=a.warm_budget,
                                  graphs=v27_substrate.identity(adapter.model).get('dynamo_unique_graphs')
                                  if substrate != 'eager' else None)), flush=True)
    for arm in [x for x in a.arm for _ in range(a.repeat)]:
        config = configs[a.dataset][arm]
        graphs = None
        if substrate != 'eager':
            v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(config))
            graphs = v27_substrate.identity(adapter.model).get('dynamo_unique_graphs')
        started = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1',
                                 kernel=v27_substrate.prefill_kernel(substrate)):
                receipt = _one(adapter, row, protocol['seeds'][0], config)
            counters = receipt.get('counters') or {}
            sub = v27_substrate.identity(adapter.model) if substrate != 'eager' else {}
            print(json.dumps(dict(arm=arm, ok=True, decoder_calls=receipt['total_decoder_calls'],
                                  substrate_local=sub.get('local'),
                                  tokens_sha=__import__('hashlib').sha256(json.dumps(receipt.get('output_tokens')).encode()).hexdigest()[:16],
                                  new_graphs=(None if graphs is None else sub.get('dynamo_unique_graphs') - graphs),
                                  request_wall_s=round(receipt['request_wall_seconds'], 3),
                                  fingerprint=bool(receipt.get('fingerprint')),
                                  peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                  peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                                  counters={k: v for k, v in counters.items()
                                            if isinstance(v, (int, float, dict)) and
                                            ('calls' in k or 'routes' in k or k in ('fresh_fused_tiles',))})), flush=True)
        except Exception as exc:  # report and keep going: the point is to see every arm
            failures += 1
            print(json.dumps(dict(arm=arm, ok=False, error=f'{type(exc).__name__}: {exc}'[:400],
                                  peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                  seconds=round(time.perf_counter() - started, 1))), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
