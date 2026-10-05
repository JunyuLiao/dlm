"""v27 diagnostic: how much do the held keep maps drift between consecutive canvases (and between M3 decisions
within a canvas)? Feasibility check for carrying a B map across canvases (Prefilling-dLLM / PulseCol-style
reuse; engineering, not a contribution). Captures every published decision (layer, canvas, step) of real requests
of a BOUND panel on the piecewise substrate, then reports, on the tiles both maps cover (the older map's key
range), per layer: kept fraction, recall of the newer map's kept tiles by the older map, and Jaccard.
usage: python -m scripts.v27_map_drift --run-dir DIR --host IP --gpu-uuid UUID --stage S --dataset D --index I
           --budget N --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--budget', type=int, default=2048)
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.cache import ScoreCache
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    if protocol.get('substrate', 'eager') != 'eager':
        from experiments.numerical_qk_reuse import v27_substrate
        v27_substrate.install(adapter.model)
    captured = []
    original = ScoreCache.publish_decision

    def capture(self, identity, step, decision):
        original(self, identity, step, decision)
        kept = (decision.eligible & ~decision.skipped) | ~decision.eligible     # mandatory tiles count as kept
        captured.append((int(identity.layer), int(identity.canvas), int(step), kept.detach().clone()))
    ScoreCache.publish_decision = capture
    row = dict(rows[protocol['ids'][a.dataset][a.index]], generation_budget=a.budget)
    try:
        for arm in a.arm:
            captured.clear()
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                receipt = _one(adapter, row, protocol['seeds'][0], configs[a.dataset][arm])
            torch.cuda.synchronize()
            by_layer = defaultdict(list)
            for layer, canvas, step, kept in captured:
                by_layer[layer].append((canvas, step, kept))
            stats = {'cross_canvas': defaultdict(list), 'within_canvas': defaultdict(list)}
            kept_fraction = []
            for layer, seq in by_layer.items():
                for (c0, s0, k0), (c1, s1, k1) in zip(seq, seq[1:]):
                    kt = min(k0.shape[-1], k1.shape[-1])
                    old, new = k0[..., :kt], k1[..., :kt]
                    inter = (old & new).sum().item()
                    union = (old | new).sum().item()
                    recall = inter / max(1, new.sum().item())
                    kind = 'cross_canvas' if c1 != c0 else 'within_canvas'
                    stats[kind]['recall'].append(recall)
                    stats[kind]['jaccard'].append(inter / max(1, union))
                    stats[kind]['gap_canvases'].append(c1 - c0)
                kept_fraction += [float(k.float().mean()) for _, _, k in seq]
            summary = {kind: {k: (round(statistics.median(v), 4) if v else None) for k, v in d.items()} | {
                'n': len(d['recall']), 'recall_p10': round(sorted(d['recall'])[len(d['recall']) // 10], 4) if d['recall'] else None}
                for kind, d in stats.items()}
            print(json.dumps(dict(arm=arm, calls=receipt['total_decoder_calls'], decisions=len(captured),
                                  kept_fraction_median=round(statistics.median(kept_fraction), 4) if kept_fraction else None,
                                  **summary)), flush=True)
    finally:
        ScoreCache.publish_decision = original


if __name__ == '__main__':
    main()
