"""Bounded offline ideal-P ball opportunity on the 12 frozen real T50 states.

This replays the unchanged v4 kernel with trace=True. It is an optimistic
real-arithmetic opportunity estimate, not a finite-precision certificate or a
timing claim. A full-output/support mismatch invalidates that state. No kernel
source, acceptance rule, or generation output is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

LABELS = ('ruler_calibration_first', 'v15_selection_first')
LAYERS = (0, 5, 29)
STEPS = (1, 3)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_keys():
    return {(label, layer, step) for label in LABELS for layer in LAYERS for step in STEPS}


def packed_valid(words: torch.Tensor, keys: int) -> np.ndarray:
    data = words.detach().cpu().numpy()
    return np.unpackbits(data.view(np.uint8), axis=-1, bitorder='little')[..., :keys].astype(bool)


def ideal_ball_bound(z: np.ndarray, previous: np.ndarray, previous_lse: np.ndarray,
                     block_lse: np.ndarray, sensitivity: np.ndarray, reference: float) -> np.ndarray:
    """Per-row ideal convex-ball risk upper bound; z is valid block Z only."""
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError('nonempty [keys, rank] block required')
    z = np.asarray(z, dtype=np.float64)
    center = z.mean(0)
    radius = np.linalg.norm(z - center, axis=1).max()
    previous = np.asarray(previous, dtype=np.float64)
    pl = np.asarray(previous_lse, dtype=np.float64)
    bl = np.asarray(block_lse, dtype=np.float64)
    sens = np.asarray(sensitivity, dtype=np.float64)
    with np.errstate(invalid='ignore', over='ignore', under='ignore'):
        alpha = np.exp(bl - np.logaddexp(pl, bl))
        bound = sens * alpha * (np.linalg.norm(center - previous, axis=-1) + radius) / max(float(reference), 1e-12)
    return np.where(np.isfinite(bl), bound, 0.)


def mask_for(state: dict, start: int, end: int):
    from experiments.value_direction_hopper.masks import PackedMask, geometry, pack

    kind = state.get('kernel_mask_kind', 'reconstructed_original_mask')
    if kind == 'reconstructed_original_mask':
        original = state['mask']
        if original is None:
            return geometry(state['q'].shape[0], state['q'].shape[-2], end - start,
                            device='cuda', causal=False, window=0)[0]
        if original.dtype == torch.bool:
            return pack(original[..., start:end].to('cuda').contiguous())
        raise ValueError('capture predates exact kernel mask and has unsupported additive mask')
    if kind == 'none':
        return None
    data = state['kernel_mask']
    if kind == 'packed':
        if start % 64:
            raise ValueError('packed-mask slice must start at a complete tile')
        return PackedMask(data[..., start // 64:(end + 63) // 64].to('cuda').contiguous(), end - start)
    if kind == 'tensor':
        return data[..., start:end].to('cuda').contiguous()
    raise ValueError(f'unsupported kernel mask kind: {kind}')


def replay(kernel, state: dict, start: int, end: int, *, gpu_ms: list[float]):
    q = state['q'].to('cuda').contiguous()
    k = state['k'][..., start:end, :].to('cuda').contiguous()
    v = state['v'][..., start:end, :].to('cuda').contiguous()
    z = state['z'][..., start:end, :].to('cuda').contiguous()
    ref = state['reference'].to('cuda').contiguous()
    sens = None if state['sensitivity'] is None else state['sensitivity'].to('cuda').contiguous()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    result = kernel(q, k, v, z, ref, mask=mask_for(state, start, end),
                    scale=state['kernel_scale'], log_threshold=state['kernel_log_threshold'],
                    mode=state['kernel_mode'], precision=state['kernel_precision'],
                    tma=bool(state['kernel_tma']), sensitivity=sens, trace=True)
    b.record()
    b.synchronize()
    gpu_ms.append(a.elapsed_time(b))
    return result


def audit_state(kernel, state: dict, deadline: float) -> dict:
    started = time.monotonic()
    gpu_ms = []
    row = dict(source_id=state['source_id'], layer=state['layer'], step=state['step'],
               layer_kind=state['layer_kind'], derived_mask=False, valid=False,
               reason=None, eligible_tiles=0, parent_skipped_tiles=0,
               ideal_ball_pass_parent_skipped=0, apparent_false_certify=0,
               first_support_tiles=0, replay_calls=0,
               ball_metadata_bytes_fp64_naive_qblock=0,
               ball_metadata_bytes_fp32_shared_uniform_mask=None)
    try:
        nk = state['k'].shape[-2]
        full = replay(kernel, state, 0, nk, gpu_ms=gpu_ms)
        row['replay_calls'] += 1
        for field, captured in (('output', 'kernel_output'), ('skipped', 'skipped'), ('eligible', 'eligible')):
            if not torch.equal(getattr(full, field).cpu(), state[captured]):
                row['reason'] = f'full_trace_{field}_mismatch'
                return row
        parent_skip = state['skipped'].numpy()
        parent_eligible = state['eligible'].numpy()
        bsz, heads, qblocks, tiles = parent_skip.shape
        nq, hk = state['q'].shape[-2], state['k'].shape[1]
        z = state['z'].double().numpy()
        ref = state['reference'].double().numpy()
        sensitivity = (np.ones((bsz, nq), dtype=np.float64) if state['sensitivity'] is None
                       else state['sensitivity'].double().numpy())
        if state.get('kernel_mask_kind') == 'packed':
            valid = packed_valid(state['kernel_mask'], nk)
        elif state.get('kernel_mask_kind') == 'tensor':
            m = state['kernel_mask']
            valid = (m.numpy().astype(bool) if m.dtype == torch.bool else
                     (np.isfinite(m.numpy()) & (m.numpy() > -1.e4)))
        elif state.get('kernel_mask_kind') is None and state['mask'] is not None:
            m = state['mask']
            valid = (m.numpy().astype(bool) if m.dtype == torch.bool else
                     (np.isfinite(m.numpy()) & (m.numpy() > -1.e4)))
        else:
            valid = np.ones((bsz, 1, nq, nk), dtype=bool)
        row['captured_z_bytes'] = state['z'].numel() * state['z'].element_size()
        row['ball_metadata_bytes_fp64_naive_qblock'] = bsz * hk * qblocks * tiles * 33 * 8
        if state['mask'] is None:
            row['ball_metadata_bytes_fp32_shared_uniform_mask'] = bsz * hk * tiles * 33 * 4
        tau = math.exp(float(state['kernel_log_threshold']))
        skipped_ratios = []
        for j in range(tiles):
            if time.monotonic() > deadline:
                row['reason'] = 'diagnostic_deadline'
                return row
            if not parent_eligible[..., j].any():
                continue
            start, end = j * 64, min(nk, (j + 1) * 64)
            if j:
                prior = replay(kernel, state, 0, start, gpu_ms=gpu_ms)
                row['replay_calls'] += 1
                if (not torch.equal(prior.skipped.cpu(), state['skipped'][..., :j]) or
                        not torch.equal(prior.eligible.cpu(), state['eligible'][..., :j])):
                    row['reason'] = f'prefix_support_mismatch_tile_{j}'
                    return row
                previous = prior.projected_state.cpu().double().numpy()
                previous_lse = prior.log_normalizer.cpu().double().numpy()
            else:
                previous = np.zeros((bsz, heads, nq, 32), dtype=np.float64)
                previous_lse = np.full((bsz, heads, nq), -np.inf)
            block = replay(kernel, state, start, end, gpu_ms=gpu_ms)
            row['replay_calls'] += 1
            block_lse = block.log_normalizer.cpu().double().numpy()
            for batch in range(bsz):
                for head in range(heads):
                    kvhead = head // (heads // hk)
                    mh = head if valid.shape[1] == heads else 0
                    for qb in range(qblocks):
                        if not parent_eligible[batch, head, qb, j]:
                            continue
                        row['eligible_tiles'] += 1
                        skipped = bool(parent_skip[batch, head, qb, j])
                        row['parent_skipped_tiles'] += int(skipped)
                        sl = slice(qb * 128, min(nq, (qb + 1) * 128))
                        legal = valid[batch, mh, sl, start:end]
                        active = np.isfinite(block_lse[batch, head, sl])
                        if not active.any():
                            row['reason'] = f'eligible_without_active_row_tile_{j}'
                            return row
                        first_support = np.any(active & ~np.isfinite(previous_lse[batch, head, sl]))
                        if first_support:
                            row['first_support_tiles'] += 1
                            continue
                        union = legal.any(axis=0)
                        if not union.any():
                            row['reason'] = f'eligible_without_legal_keys_tile_{j}'
                            return row
                        bound = ideal_ball_bound(z[batch, kvhead, start:end][union], previous[batch, head, sl],
                                                 previous_lse[batch, head, sl], block_lse[batch, head, sl],
                                                 sensitivity[batch, sl], ref[batch, kvhead])
                        if not np.isfinite(bound[active]).all():
                            row['reason'] = f'nonfinite_ideal_bound_tile_{j}'
                            return row
                        if skipped:
                            skipped_ratios.append(float(bound[active].max() / tau))
                        certifies = bool(np.all(bound[active] < tau))
                        if certifies:
                            row['ideal_ball_pass_parent_skipped'] += int(skipped)
                            row['apparent_false_certify'] += int(not skipped)
        row['parent_skipped_bound_over_tau_median'] = (None if not skipped_ratios else
                                                        float(np.median(skipped_ratios)))
        row['parent_skipped_bound_over_tau_p90'] = (None if not skipped_ratios else
                                                     float(np.quantile(skipped_ratios, .9)))
        row['valid'] = row['apparent_false_certify'] == 0
        if not row['valid']:
            row['reason'] = 'apparent_false_certify'
    except Exception as exc:
        row['reason'] = f'{type(exc).__name__}: {exc}'[:300]
    finally:
        row['diagnostic_wall_s'] = time.monotonic() - started
        row['replay_gpu_ms'] = sum(gpu_ms)
    return row


def main() -> None:
    from experiments.value_direction_hopper.cuda import Kernel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--torch-library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-minutes', type=float, default=15.)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    states = torch.load(args.states, map_location='cpu', weights_only=True)
    real = [s for s in states if not s.get('derived_mask')]
    derived = [s for s in states if s.get('derived_mask')]
    keys = {(s['source_id'].split('/')[0], s['layer'], s['step']) for s in real}
    if len(keys) != len(real) or keys - expected_keys():
        raise ValueError('capture has duplicate or unplanned real states')
    kernel = Kernel(args.library, torch_library=args.torch_library)
    deadline = time.monotonic() + args.max_minutes * 60
    rows = []
    for state in real:
        if time.monotonic() > deadline:
            rows.append(dict(source_id=state['source_id'], layer=state['layer'], step=state['step'],
                             layer_kind=state['layer_kind'], valid=False, reason='diagnostic_deadline'))
        else:
            rows.append(audit_state(kernel, state, deadline))
        print(json.dumps({k: rows[-1].get(k) for k in ('source_id', 'valid', 'reason', 'parent_skipped_tiles',
                                                        'ideal_ball_pass_parent_skipped', 'apparent_false_certify')}), flush=True)
    report = dict(schema='v18_ideal_ball_opportunity_v1', claim='ideal normalized-P arithmetic opportunity only',
                  finite_precision_certificate=False, timing_claim=False,
                  states_sha256=file_sha(args.states), library_sha256=file_sha(args.library),
                  expected_real_states=12, present_real_states=len(real),
                  excluded_derived_states=[dict(source_id=s['source_id'], reason='controlled_mask_derivative_not_a_real_state')
                                           for s in derived],
                  missing_states=[dict(source=x[0], layer=x[1], step=x[2]) for x in sorted(expected_keys() - keys)],
                  rows=rows)
    for kind in ('global', 'local'):
        group = [r for r in rows if r.get('layer_kind') == kind]
        valid_group = [r for r in group if r.get('valid')]
        skipped = sum(r.get('parent_skipped_tiles', 0) for r in valid_group)
        passed = sum(r.get('ideal_ball_pass_parent_skipped', 0) for r in valid_group)
        report[kind] = dict(states=len(group), valid_states=len(valid_group),
                            parent_skipped_tiles=skipped, ideal_ball_pass_parent_skipped=passed,
                            skipped_coverage_fraction=None if not skipped else passed / skipped,
                            apparent_false_certify_all_rows=sum(r.get('apparent_false_certify', 0) for r in group))
    report['all_12_valid'] = len(real) == 12 and not report['missing_states'] and all(r.get('valid') for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(present=len(real), missing=len(report['missing_states']),
                          valid=sum(bool(r.get('valid')) for r in rows))))


if __name__ == '__main__':
    main()
