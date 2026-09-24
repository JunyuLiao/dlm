"""Qualify the pre-QK current-output consumer against two references.

For each configuration the SAME Q/K/V and the SAME retained support feed:
  preqk    -- historical_route_preqk_current_output's consumer, which never
              computes a dropped tile's QK;
  ref_bf16 -- the diagnostic routing_only_current_output path: materialize
              all current scores with observe_scores, then PV on the bitmap;
  ref_fp32 -- an independent FP32 reference (FP32 QK, explicit mask, FP32
              softmax and PV) over exactly the same retained support.

ref_bf16 is NOT ground truth: it is the incumbent BF16 path. The question
this answers is whether preqk is *further from FP32* than the incumbent is.
Errors are reported, never tolerated-until-pass; the caller decides.

Geometry defaults follow the executed model: sliding layers use head_dim
256 with 16/8 GQA, global layers use global_head_dim 512 with 16/2 GQA.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def fp32_reference(q, k, v, skipped, eligible, *, scale, window):
    """Independent FP32 attention restricted to the retained support."""
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[-2]
    groups = h // hk
    qf = q.float()
    kf = k.float().repeat_interleave(groups, dim=1)
    vf = v.float().repeat_interleave(groups, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * float(scale)
    legal = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
    if window:
        qpos = torch.arange(nq, device=q.device) + (nk - nq)
        kpos = torch.arange(nk, device=q.device)
        legal = kpos[None, :] >= (qpos[:, None] - int(window) + 1)
    retained = (eligible & ~skipped)
    keep = retained.repeat_interleave(128, dim=-2).repeat_interleave(64, dim=-1)[..., :nq, :nk]
    scores = scores.masked_fill(~(legal[None, None] & keep), -math.inf)
    rows = torch.isfinite(scores).any(-1, keepdim=True)
    probabilities = torch.softmax(torch.where(rows, scores, torch.zeros_like(scores)), dim=-1)
    probabilities = torch.where(rows, probabilities, torch.zeros_like(probabilities))
    return torch.matmul(probabilities, vf)


def expected_programs(retained, nq):
    """Exact number of 16-row programs that must execute a retained tile.

    A [B,H,Qtile,Ktile] retained entry is executed once per 16-row program
    inside that 128-row query tile, and the last query tile may be short.
    """
    total = 0
    tiles = retained.sum(dim=(0, 1, 3)).tolist()   # retained tiles per query block
    for block, count in enumerate(tiles):
        rows = min(128, nq - block * 128)
        total += int(count) * ((rows + 15) // 16)
    return total


def relative(candidate, reference):
    difference = (candidate.float() - reference.float()).flatten()
    return float(difference.norm() / reference.float().flatten().norm().clamp_min(1e-12))


def one_case(name, *, h, hk, d, nq, nk, window, threshold, seed, all_kept=False,
             drop_fraction=0.0):
    from experiments.numerical_qk_reuse.cached_executor import (attention, preqk_attention,
                                                                route_only)
    from experiments.numerical_qk_reuse.integration import Attention

    generator = torch.Generator(device='cuda').manual_seed(seed)
    # Transposed views, exactly how the model hands attention its tensors.
    q = torch.randn(1, nq, h, d, generator=generator, device='cuda', dtype=torch.bfloat16).transpose(1, 2) * 1
    k = torch.randn(1, nk, hk, d, generator=generator, device='cuda', dtype=torch.bfloat16).transpose(1, 2) * 1
    v = torch.randn(1, nk, hk, d, generator=generator, device='cuda', dtype=torch.bfloat16).transpose(1, 2) * 1
    q, k, v = q * .3, k * .3, v * 1.
    q, k, v = q.transpose(1, 2).transpose(1, 2), k, v  # keep the non-contiguous layout
    scale = 1.0
    # Legality comes only from the geometry the consumer can see (window,
    # tails). Hand-doctoring the score tensor would test an input that cannot
    # arise from (q, k, window) and is outside the consumer's contract: the
    # incumbent would read the doctored value while the pre-QK consumer, like
    # the FP32 reference, recomputes from Q/K. Structurally illegal keys and
    # near-empty rows are therefore produced with a genuinely narrow window.
    scores = Attention.observe_scores(q, k, None, scale, False, window, 0).contiguous()

    z = torch.randn(1, hk, nk, 32, generator=generator, device='cuda', dtype=torch.float32).contiguous()
    reference = (torch.rand(1, hk, generator=generator, device='cuda', dtype=torch.float32) + .5).contiguous()
    sensitivity = (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                        dtype=torch.float32)).contiguous()
    routing = route_only(scores, z, reference, sensitivity=sensitivity,
                         log_threshold=-math.inf if all_kept else threshold)
    skipped = torch.zeros_like(routing.skipped) if all_kept else routing.skipped
    if drop_fraction:
        # The consumer's correctness does not depend on where the bitmap came
        # from, and random sketches rarely trip the frozen thresholds. Force a
        # realistic drop pattern so the skip path is actually exercised, and
        # give the SAME bitmap to all three implementations.
        chooser = torch.rand(skipped.shape, generator=generator, device='cuda')
        skipped = (chooser < drop_fraction) & routing.eligible

    incumbent = attention(scores, v.contiguous(), skipped=skipped, eligible=routing.eligible)
    candidate = preqk_attention(q, k, v, skipped, routing.eligible, scale=scale,
                                window=window, trace=True)
    exact = fp32_reference(q, k, v, skipped, routing.eligible, scale=scale, window=window)

    retained = int((routing.eligible & ~skipped).sum())
    eligible_total = int(routing.eligible.sum())
    counters = candidate.counters.sum(dim=(0, 1, 2)).tolist()
    return dict(
        case=name, heads=h, kv_heads=hk, head_dim=d, queries=nq, keys=nk,
        window=window, all_kept=all_kept, drop_fraction=drop_fraction,
        structurally_illegal_pairs=int((~torch.isfinite(scores)).sum()),
        retained_tiles=retained, eligible_tiles=eligible_total,
        skipped_tiles=eligible_total - retained,
        executed_tile_programs=int(counters[0]), kv_tile_loads=int(counters[1]),
        qk_tile_dots=int(counters[2]),
        expected_tile_programs=expected_programs(routing.eligible & ~skipped, nq),
        preqk_vs_incumbent=relative(candidate.output, incumbent.output),
        preqk_vs_fp32=relative(candidate.output, exact),
        incumbent_vs_fp32=relative(incumbent.output, exact),
        max_abs_preqk_vs_incumbent=float((candidate.output.float() - incumbent.output.float()).abs().max()),
    )


def cases() -> list[dict[str, Any]]:
    local = dict(h=16, hk=8, d=256)     # sliding layers: head_dim 256, GQA 2:1
    glob = dict(h=16, hk=2, d=512)      # full layers: global_head_dim 512, GQA 8:1
    threshold_local, threshold_global = -1.0099318265914916, -3.1366905212402343
    return [
        dict(name='local_short_prefix', **local, nq=256, nk=384, window=1024,
             threshold=threshold_local, seed=11),
        dict(name='local_saturated_window', **local, nq=256, nk=1279, window=1024,
             threshold=threshold_local, seed=12),
        dict(name='local_partial_tiles', **local, nq=200, nk=421, window=1024,
             threshold=threshold_local, seed=13),
        # A narrow window makes whole KV tiles structurally illegal for some
        # query blocks (eligible=False) and leaves others only partly legal.
        dict(name='local_narrow_window_illegal_tiles', **local, nq=256, nk=512, window=64,
             threshold=threshold_local, seed=14),
        dict(name='local_all_kept', **local, nq=256, nk=384, window=1024,
             threshold=threshold_local, seed=15, all_kept=True),
        dict(name='global_short_prefix', **glob, nq=256, nk=384, window=None,
             threshold=threshold_global, seed=21),
        dict(name='global_long_prefix', **glob, nq=256, nk=2048, window=None,
             threshold=threshold_global, seed=22),
        dict(name='global_partial_tiles', **glob, nq=200, nk=421, window=None,
             threshold=threshold_global, seed=23),
        dict(name='global_all_kept', **glob, nq=256, nk=384, window=None,
             threshold=threshold_global, seed=24, all_kept=True),
        dict(name='small_d64_gqa1', h=4, hk=4, d=64, nq=128, nk=192, window=None,
             threshold=-1.0, seed=31),
        dict(name='d128_gqa4', h=8, hk=2, d=128, nq=256, nk=384, window=256,
             threshold=-1.0, seed=32),
        # Forced drop patterns: these are the cases that prove skipped tiles
        # issue no QK dot and no K/V load.
        dict(name='local_dropped_40pct', **local, nq=256, nk=1024, window=1024,
             threshold=threshold_local, seed=41, drop_fraction=.4),
        dict(name='global_dropped_60pct', **glob, nq=256, nk=1024, window=None,
             threshold=threshold_global, seed=42, drop_fraction=.6),
        dict(name='global_dropped_90pct', **glob, nq=200, nk=1536, window=None,
             threshold=threshold_global, seed=43, drop_fraction=.9),
    ]


def run() -> dict[str, Any]:
    rows = []
    for case in cases():
        name = case.pop('name')
        rows.append(one_case(name, **case))
    return dict(schema='preqk_qualification_v1',
                note=('ref_bf16 (incumbent) is not ground truth; the comparison that '
                      'matters is preqk_vs_fp32 against incumbent_vs_fp32. Errors are '
                      'reported as measured.'),
                geometry=('sliding layers head_dim 256 GQA 16/8; full layers '
                          'global_head_dim 512 GQA 16/2, from the executed model config'),
                rows=rows)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for row in payload['rows']:
        print(f"{row['case']:38s} d={row['head_dim']:3d} retained={row['retained_tiles']:5d}/"
              f"{row['eligible_tiles']:5d} qk_dots={row['qk_tile_dots']:6d} "
              f"preqk~inc={row['preqk_vs_incumbent']:.3e} preqk~fp32={row['preqk_vs_fp32']:.3e} "
              f"inc~fp32={row['incumbent_vs_fp32']:.3e}")


if __name__ == '__main__':
    main()
