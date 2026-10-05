"""Bounded parent-call and standalone ball-metadata cost on frozen T60 states.

Timing-buffer fields are per CTA, not additive GPU wall time. This is a
diagnostic of unchanged parent work, not candidate timing or request speedup.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.v18_ball_metadata import build, cpu_reference
from scripts.v18_ball_opportunity import expected_keys, file_sha, mask_for


FIELD_NAMES = ('qk', 'softmax', 'pz', 'decision', 'pv', 'risk')


def split_tiles(keys: int, queries: int) -> tuple[int, int]:
    if keys < 1 or queries < 1 or queries > keys:
        raise ValueError('invalid frozen state geometry')
    # Only complete tiles strictly before the canvas start are static-prefix tiles.
    return max(0, (keys - queries) // 64), (keys + 63) // 64


def timing_summary(timings: torch.Tensor) -> dict:
    data = timings.detach().cpu().numpy().reshape(-1, 6).astype(np.float64)
    active = data.sum(1) > 0
    data = data[active]
    if not len(data):
        raise ValueError('no measured CTAs')
    serial_router_stage = data[:, [0, 1, 2, 3, 5]].sum(1)
    share = np.divide(data[:, 2], serial_router_stage,
                      out=np.zeros(len(data), dtype=np.float64), where=serial_router_stage > 0)
    return dict(active_ctas=len(data), fields={name: dict(nonzero_ctas=int(np.count_nonzero(data[:, i])),
                                               median_ns=float(np.median(data[:, i])),
                                               p90_ns=float(np.quantile(data[:, i], .9)))
                                         for i, name in enumerate(FIELD_NAMES)},
                pz_over_serial_router_stage_per_cta_median=float(np.median(share)),
                pz_over_serial_router_stage_per_cta_p90=float(np.quantile(share, .9)))


def event_ms(fn, *, warm: int = 2, repeats: int = 7) -> list[float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        samples.append(float(a.elapsed_time(b)))
    return samples


def measure_state(kernel, state: dict) -> dict:
    q, k, v, z = (state[name].to('cuda').contiguous() for name in ('q', 'k', 'v', 'z'))
    ref = state['reference'].to('cuda').contiguous()
    sens = None if state['sensitivity'] is None else state['sensitivity'].to('cuda').contiguous()
    mask = mask_for(state, 0, k.shape[-2])
    kwargs = dict(mask=mask, scale=state['kernel_scale'], log_threshold=state['kernel_log_threshold'],
                  mode=state['kernel_mode'], precision=state['kernel_precision'],
                  tma=bool(state['kernel_tma']), sensitivity=sens, trace=False)

    def call():
        return kernel(q, k, v, z, ref, **kwargs)

    ordinary = call()
    captured_parity = {field: bool(torch.equal(getattr(ordinary, field).cpu(), state[captured]))
                       for field, captured in (('output', 'kernel_output'), ('skipped', 'skipped'),
                                               ('eligible', 'eligible'))}
    if not all(captured_parity.values()):
        raise ValueError(f'ordinary parent call differs from frozen capture: {captured_parity}')
    whole_call_ms = event_ms(call)
    b, heads, nq, _ = q.shape
    nk = k.shape[-2]
    timings = torch.zeros((b, heads, 2 * ((nq + 127) // 128), (nk + 63) // 64, 6),
                          device=q.device, dtype=torch.uint64)
    with_timings = kernel(q, k, v, z, ref, timings=timings, **kwargs)
    parity = {field: bool(torch.equal(getattr(ordinary, field), getattr(with_timings, field)))
              for field in ('output', 'skipped', 'eligible')}
    if not all(parity.values()):
        raise ValueError(f'timing-buffer call changes parent output/support: {parity}')
    summary = timing_summary(timings)

    metadata = torch.empty((z.shape[0], z.shape[1], (nk + 63) // 64, 33), device=z.device)
    static_tiles, total_tiles = split_tiles(nk, nq)
    build(z, out=metadata)
    expected = cpu_reference(state['z'])
    absolute_error = (metadata.cpu().double() - expected).abs()
    max_error = float(absolute_error.amax())
    if not np.isfinite(max_error) or max_error > 1e-3:
        raise ValueError(f'Z metadata disagrees with FP64 CPU reference: {max_error}')
    full_ms = event_ms(lambda: build(z, out=metadata))
    static_ms = event_ms(lambda: build(z, start_tile=0, end_tile=static_tiles, out=metadata)) if static_tiles else []
    canvas_ms = event_ms(lambda: build(z, start_tile=static_tiles, end_tile=total_tiles, out=metadata))
    return dict(source_id=state['source_id'], layer=state['layer'], step=state['step'],
                layer_kind=state['layer_kind'], queries=nq, keys=nk,
                parent_whole_call_ms_samples=whole_call_ms,
                parent_whole_call_ms_median=statistics.median(whole_call_ms),
                ordinary_capture_parity=captured_parity,
                timing_buffer_parity=parity, per_cta_timing_ns=summary,
                metadata_max_abs_error_vs_cpu_fp64=max_error,
                metadata_static_prefix_tiles=static_tiles, metadata_canvas_containing_tiles=total_tiles-static_tiles,
                metadata_bytes=metadata.numel() * metadata.element_size(),
                metadata_full_ms_samples=full_ms, metadata_full_ms_median=statistics.median(full_ms),
                metadata_static_prefix_ms_samples=static_ms,
                metadata_canvas_containing_ms_samples=canvas_ms,
                metadata_static_prefix_ms_median=statistics.median(static_ms) if static_ms else None,
                metadata_canvas_containing_ms_median=statistics.median(canvas_ms))


def main() -> None:
    from experiments.value_direction_hopper.cuda import Kernel

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('states', 'library', 'torch_library', 'out'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=300)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    states = torch.load(args.states, map_location='cpu', weights_only=True)
    real = [s for s in states if not s.get('derived_mask')]
    keys = {(s['source_id'].split('/')[0], s['layer'], s['step']) for s in real}
    if (len(real) != 12 or keys != expected_keys() or
            any(s.get('frontier_arm') != 'T60' or s.get('mask') is not None for s in real)):
        raise ValueError('cost harness requires exactly 12 frozen real mask-none T60 states')
    kernel = Kernel(args.library, torch_library=args.torch_library)
    if kernel.timing_fields != 6 or kernel.abi_version != 4:
        raise ValueError('requires unchanged v4 six-field timing ABI')
    start = time.monotonic()
    rows = []
    for state in real:
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError('bounded cost diagnostic deadline')
        rows.append(measure_state(kernel, state))
        print(json.dumps({'source_id': state['source_id'], 'parent_ms': rows[-1]['parent_whole_call_ms_median']}), flush=True)
    report = dict(schema='v18_ball_cost_diagnostic_v1', states_sha256=file_sha(args.states),
                  library_sha256=file_sha(args.library), frontier_arm='T60', expected_real_states=12,
                  measured_real_states=len(rows), trace=False, whole_call_warmups=2, whole_call_repeats=7,
                  timing_buffer_fields=list(FIELD_NAMES),
                  warning='Whole-call CUDA event spans include host launch gaps and allocations. Per-CTA PZ share is PZ/(QK+softmax+PZ+decision+risk), excludes overlapping PV, and is not whole-call latency fraction. Per-CTA fields cannot be summed as GPU wall time. Preallocated standalone metadata timing is a lower incremental cost, not integrated. No candidate/request speedup or finite-precision certificate.',
                  rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'measured_real_states': len(rows)}))


if __name__ == '__main__':
    main()
