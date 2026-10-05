"""v27 step-count diagnosis: per denoising call, the native stop statistic (mean token entropy of the
processed logits) and whether the stop rule fired, for a few bound arms on one manifest row (GPU).

StableAndConfidentStoppingCriteria.__call__ is wrapped to record, per call, the canvas index, the
mean entropy and the stable/confident/stop flags; nothing else changes. Prints per-canvas step
counts and entropy trajectories (numbers only, no text). Used to place the density-gate thresholds.
usage: python -m scripts.v27_entropy_probe --run-dir DIR --host IP --gpu-uuid UUID --stage STAGE
           --dataset D --index I --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    import transformers.models.diffusion_gemma.generation_diffusion_gemma as gen
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    substrate = protocol.get('substrate', 'eager')
    if substrate != 'eager':
        from experiments.numerical_qk_reuse import v27_substrate
        v27_substrate.install(adapter.model)
    trace = []
    original = gen.StableAndConfidentStoppingCriteria.__call__

    def recording(self, argmax_canvas, logits, **kw):
        stop = original(self, argmax_canvas, logits, **kw)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        trace.append((float(entropy), bool(stop.all())))
        return stop
    gen.StableAndConfidentStoppingCriteria.__call__ = recording
    row = rows[protocol['ids'][a.dataset][a.index]]
    try:
        for arm in a.arm:
            trace.clear()
            if substrate != 'eager':
                v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(configs[a.dataset][arm]))
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                receipt = _one(adapter, row, protocol['seeds'][0], configs[a.dataset][arm])
            canvases, current = [], []
            for entropy, stop in trace:
                current.append(round(entropy, 4))
                if stop:
                    canvases.append(current)
                    current = []
            if current:
                canvases.append(current + ['cap'])
            steps = [len(c) for c in canvases]
            counters = receipt.get('counters') or {}
            print(json.dumps(dict(arm=arm, calls=receipt['total_decoder_calls'], canvases=len(canvases),
                                  gate_entry_steps=counters.get('density_gate_entry_steps'),
                                  gate_dense_calls=counters.get('density_gate_dense_calls'),
                                  steps_per_canvas=steps, first_canvases=canvases[:3],
                                  median_last3=[c[-4:-1] for c in canvases[:6]])), flush=True)
    finally:
        gen.StableAndConfidentStoppingCriteria.__call__ = original


if __name__ == '__main__':
    main()
