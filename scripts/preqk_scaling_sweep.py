"""Where the remaining cost actually is, as a function of context length.

The full-forward profile could only capture one state. This sweeps the real
per-layer geometries across the key lengths a full answer actually reaches
and times, for each:

  dense       -- the installed native SDPA path (the baseline to beat)
  observe+PV  -- what routing_only_current_output pays per ordinary step:
                 materialize all current scores, then PV on the bitmap
  preqk       -- the new consumer: current QK/PV on retained tiles only
  route       -- route_only, the selector that BOTH numerical arms pay

The comparison that decides the method is `preqk + route` against `dense`,
because a numerical step must pay both. Local layers cap at
``sliding_window-1 + canvas``; global layers grow with the answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

GEOMETRIES = {
    # (heads, kv_heads, head_dim, window) from the executed model config
    'local_D256_GQA16_8': (16, 8, 256, 1024),
    'global_D512_GQA16_2': (16, 2, 512, None),
}


def timed(run, warmup=3, reps=10):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def sweep(lengths, drop_fraction, queries, threshold):
    from experiments.numerical_qk_reuse.cached_executor import (attention, preqk_attention,
                                                                route_only)
    from experiments.numerical_qk_reuse.integration import Attention
    from scripts.native_reuse_phaseAB_diagnostic import true_dense

    rows = []
    for name, (h, hk, d, window) in GEOMETRIES.items():
        for nk in lengths:
            if window and nk > window + queries:
                continue          # a local layer's compact range cannot exceed this
            generator = torch.Generator(device='cuda').manual_seed(1)
            q = torch.randn(1, queries, h, d, generator=generator, device='cuda',
                            dtype=torch.bfloat16).transpose(1, 2) * .3
            k = torch.randn(1, nk, hk, d, generator=generator, device='cuda',
                            dtype=torch.bfloat16).transpose(1, 2) * .3
            v = torch.randn(1, nk, hk, d, generator=generator, device='cuda',
                            dtype=torch.bfloat16).transpose(1, 2)
            scores = Attention.observe_scores(q, k, None, 1., False, window, 0)
            z = torch.randn(1, hk, nk, 32, generator=generator, device='cuda',
                            dtype=torch.float32).contiguous()
            ref = (torch.rand(1, hk, generator=generator, device='cuda',
                              dtype=torch.float32) + .5).contiguous()
            routing = route_only(scores, z, ref, log_threshold=threshold)
            dropped = (torch.rand(routing.skipped.shape, generator=generator,
                                  device='cuda') < drop_fraction) & routing.eligible
            contiguous_v = v.contiguous()
            rows.append(dict(
                geometry=name, keys=nk, queries=queries,
                drop_fraction_actual=int(dropped.sum()) / max(1, int(routing.eligible.sum())),
                dense_ms=timed(lambda: true_dense(q, k, v, None, scaling=1., is_causal=False)),
                observe_plus_pv_ms=timed(lambda: attention(
                    Attention.observe_scores(q, k, None, 1., False, window, 0),
                    contiguous_v, skipped=dropped, eligible=routing.eligible)),
                preqk_ms=timed(lambda: preqk_attention(q, k, v, dropped, routing.eligible,
                                                       scale=1., window=window)),
                route_only_ms=timed(lambda: route_only(scores, z, ref, log_threshold=threshold)),
            ))
            row = rows[-1]
            row['preqk_plus_route_ms'] = row['preqk_ms'] + row['route_only_ms']
            row['preqk_plus_route_vs_dense'] = row['preqk_plus_route_ms'] / row['dense_ms']
            row['observe_pv_plus_route_vs_dense'] = (
                (row['observe_plus_pv_ms'] + row['route_only_ms']) / row['dense_ms'])
    return rows


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[384, 1279, 2048, 4096, 8192])
    parser.add_argument('--queries', type=int, default=256)
    parser.add_argument('--drop-fraction', type=float, default=.5)
    parser.add_argument('--threshold', type=float, default=-1.)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    rows = sweep(args.lengths, args.drop_fraction, args.queries, args.threshold)
    payload = dict(schema='preqk_scaling_sweep_v1', queries=args.queries,
                   requested_drop_fraction=args.drop_fraction, rows=rows,
                   note=('A numerical step pays route_only AND its consumer, so '
                         'preqk_plus_route_vs_dense is the ratio that decides the '
                         'method. Attention only: FFN and other shared model work '
                         'are identical across arms and excluded.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"{'geometry':>22}{'nk':>7}{'dense':>9}{'obs+PV':>9}{'preqk':>9}{'route':>9}"
          f"{'preqk+route':>13}{'vs dense':>10}")
    for row in rows:
        print(f"{row['geometry']:>22}{row['keys']:7d}{row['dense_ms']:9.3f}"
              f"{row['observe_plus_pv_ms']:9.3f}{row['preqk_ms']:9.3f}{row['route_only_ms']:9.3f}"
              f"{row['preqk_plus_route_ms']:13.3f}{row['preqk_plus_route_vs_dense']:10.2f}")


if __name__ == '__main__':
    main()
