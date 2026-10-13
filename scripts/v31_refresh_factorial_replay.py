"""Complete frozen 2x2x4 selector replay, with a separate NumPy oracle.

Dense QK/PV is offline diagnostic work. Progress events and Q/K/V are imposed
by the source V31 trajectory, not recomputed from hypothetical model logits.
Pre-anchor bootstrap/carry calls are excluded identically from every arm.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.numerical_qk_reuse.v31_stable_refresh import StableConfig, StableState, mass_scores
from experiments.numerical_qk_reuse.v31_stable_observe import observe, tile_ids
from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
from experiments.numerical_qk_reuse import v27_fa4
from scripts.v31_refresh_factor_analysis import factorial, sha


@dataclass
class ReplayClock:
    period: int
    progress: bool
    initialized: bool = False
    reuse_steps: int = 0

    def step(self, selection_event):
        if selection_event and (not self.initialized or self.progress):
            reason = 'progress' if self.initialized else 'initial'
            self.initialized = True
            self.reuse_steps = 0
            return reason, False
        if not self.initialized:
            return 'preanchor', False
        self.reuse_steps += 1
        return 'reuse', bool(self.period and self.reuse_steps % self.period == 0)


def configurations(selector):
    result = {}
    for entry in factorial():
        cfg = dict(selector, mode=entry['mode'], hysteresis=0., probes=0, updates_enabled=True)
        result[entry['arm']] = (StableConfig(**cfg), entry['interval'], entry['progress_dense'])
    for progress in (True, False):
        name = 'v31_progress_dense_control' if progress else 'v31_no_progress_dense_hold'
        result[name] = (StableConfig(**dict(selector, mode='raw', hysteresis=0., updates_enabled=False)), 0, progress)
    return result


def logsumexp(array, axis):
    maximum = np.max(array, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        shifted = np.where(finite, array - maximum, -np.inf)
        result = maximum + np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return np.where(finite, result, -np.inf).squeeze(axis)


def cpu_scores(z):
    normalizer = logsumexp(z, 1)
    with np.errstate(invalid='ignore'):
        shares = np.where(np.isfinite(normalizer[:, None]), z - normalizer[:, None], -np.inf)
    return shares.max(-1)


def cpu_topk(score, k):
    selected = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(selected, np.argsort(-score, axis=-1, kind='stable')[:, :k], True, axis=1)
    return selected


class NumpyOracle:
    """Independent array implementation; loops are bounded by event cap/units."""
    def __init__(self, anchor, selected, config):
        self.anchor = anchor.astype(np.float64)
        self.selected = selected.copy()
        self.config = config
        self.k = int(selected.sum(1)[0])
        self.streak = np.zeros(selected.shape, dtype=np.int32)
        self.cumulative = np.zeros(selected.shape[0], dtype=np.int32)

    def observe(self, fresh, tail):
        old, cfg = self.selected, self.config
        fresh = fresh.astype(np.float64)
        hybrid = np.where(old[..., None], fresh, self.anchor)
        score = cpu_scores(hybrid)
        proposed = cpu_topk(score + (cfg.hysteresis * old if cfg.mode == 'stable' else 0), self.k)
        favorable = proposed & ~old
        self.streak = np.where(favorable, self.streak + 1, 0)
        selected = old.copy()
        if cfg.updates_enabled and cfg.mode == 'raw':
            selected = proposed
        elif cfg.updates_enabled and 0 < self.k < old.shape[1]:
            eligible = favorable & (self.streak >= cfg.persistence)
            removable = old & ~proposed
            norm = np.logaddexp(logsumexp(np.where(old[..., None], fresh, -np.inf), 1), tail)
            removed_mass = np.zeros(norm.shape)
            for _ in range(cfg.event_cap):
                ci = np.where(eligible, score, -np.inf).argmax(1)
                wi = np.where(removable, score, np.inf).argmin(1)
                units = np.arange(len(ci))
                with np.errstate(invalid='ignore'):
                    mass = np.nan_to_num(np.exp(fresh[units, wi] - norm))
                accept = (eligible.any(1) & removable.any(1)
                          & (score[units, ci] - score[units, wi] > cfg.hysteresis)
                          & (self.cumulative < cfg.cumulative_cap)
                          & ((removed_mass + mass).max(1) <= cfg.removal_tolerance))
                selected[units[accept], wi[accept]] = False
                selected[units[accept], ci[accept]] = True
                removed_mass += np.where(accept[:, None], mass, 0)
                self.cumulative += accept
                eligible[units, ci] = False
                removable[units, wi] = False
        if cfg.mode != 'stable':
            self.cumulative += (selected & ~old).sum(1)
        self.selected = selected
        return selected, score


def pooled(tensors):
    values = torch.cat(tensors) if tensors else torch.zeros(1)
    return dict(mean=float(values.mean()), p95=float(values.quantile(.95)), maximum=float(values.max()), rows=values.numel())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-study', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--attempt', default='attempt001')
    args = p.parse_args()
    source, study = args.source_study.resolve(), args.study.resolve()
    identity_path = study / 'specs/replay_identity.json'
    identity = json.loads(identity_path.read_text())
    root = Path(__file__).resolve().parents[1]
    for path, digest in identity['source_sha256'].items():
        if sha(root / path) != digest:
            raise RuntimeError('replay source identity changed')
    inventory_path = source / 'specs/real_snapshot_inventory.json'
    if sha(inventory_path) != identity['snapshot_inventory_sha256']:
        raise RuntimeError('snapshot inventory changed')
    inventory = json.loads(inventory_path.read_text())
    for path, digest in inventory['sha256'].items():
        if sha(source / path) != digest:
            raise RuntimeError('frozen Q/K/V content changed')
    parameter_path = source / 'specs/parameters_attempt004.json'
    if sha(parameter_path) != identity['parameters_sha256']:
        raise RuntimeError('selector parameters changed')
    selector = json.loads(parameter_path.read_text())['selector']
    configs = configurations(selector)
    binding = json.loads((source / 'operator/trajectory_binding.json').read_text())
    trajectory = source / binding['completed_leaf']
    for name in ('config', 'receipt'):
        if sha(trajectory / (name + '.json')) != binding[name + '_sha256']:
            raise RuntimeError('trajectory identity changed')
    output = study / 'operator' / args.attempt
    if output.exists():
        raise RuntimeError('preserve the prior replay attempt')
    output.mkdir(parents=True)
    captures = dict(count=0)
    capture_begin = torch.cuda.CUDAGraph.capture_begin
    def count_capture(graph, *a, **kw):
        captures['count'] += 1
        return capture_begin(graph, *a, **kw)
    torch.cuda.CUDAGraph.capture_begin = count_capture
    torch.cuda.reset_peak_memory_stats()
    groups = {}
    for path in sorted((trajectory / 'private/trajectories').glob('request*/canvas*_call*_layer*.pt')):
        groups.setdefault((path.parent.name, path.name.split('_')[0], path.name.split('_')[-1]), []).append(path)
    if sum(map(len, groups.values())) != inventory['step_files']:
        raise RuntimeError('snapshot coverage mismatch')
    errors = {arm: [] for arm in configs}
    reuse_errors = {arm: [] for arm in configs}
    changes = {arm: [] for arm in configs}
    ages = {arm: torch.zeros(128, dtype=torch.int64) for arm in configs}
    records, coverage = [], set()
    max_stat_error = max_score_error = 0.
    oracle_events = frames = excluded = 0
    started = time.perf_counter()
    for group, files in sorted(groups.items()):
        states, oracles, pending = {}, {}, {}
        clocks = {arm: ReplayClock(period, progress) for arm, (_, period, progress) in configs.items()}
        prefix = None
        for path in sorted(files):
            data = torch.load(path, map_location='cpu', weights_only=True)
            coverage.add((data['dataset'], data['layer']))
            if prefix is None:
                prefix = torch.load(path.parent / data['prefix_file'], map_location='cuda', weights_only=True)
            q = data['q'].cuda()
            k = torch.cat((prefix['k'], data['tail_k'].cuda()), 2)
            v = torch.cat((prefix['v'], data['tail_v'].cuda()), 2)
            n, pt, h = data['n'], data['prefix'] // 64, q.shape[1]
            qb = (n + 127) // 128
            z = torch.full((h, qb, pt, 128), -math.inf, device='cuda')
            dense = observe_dense(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), data['scale'], z).float()
            z = z.reshape(h * qb, pt, 128)
            tail_ids = torch.arange(pt, (k.shape[2] + 63) // 64, device='cuda')[None].expand(h * qb, -1).contiguous()
            tail = observe(q, k, data['scale'], tail_ids, n).logsumexp(1)
            dense_norm = torch.logaddexp(z.logsumexp(1), tail)
            valid = (torch.arange(qb * 128, device='cuda').reshape(qb, 128) < n).expand(h, -1, -1).reshape(h * qb, 128)
            oracle_z = None
            def output_for(selected):
                full = torch.ones((1, h, qb, (k.shape[2] + 63) // 64), dtype=torch.bool, device='cuda')
                full[0, :, :, :pt] = selected.reshape(h, qb, pt)
                return v27_fa4.sparse_lists(q, k, v, v27_fa4.block_sparse_tensors(full), data['scale']).float()
            actions = {arm: clock.step(data['selection_event']) for arm, clock in clocks.items()}
            if all(action[0] == 'preanchor' for action in actions.values()):
                excluded += 1
                continue
            frames += 1
            for arm, (cfg, period, progress) in configs.items():
                reason, due = actions[arm]
                dense_event = reason in ('initial', 'progress')
                old = states[arm].selected.clone() if arm in states else None
                if dense_event:
                    score = mass_scores(z)
                    if old is None:
                        selected = data['next'].cuda()[0, :, :, :pt].reshape(h * qb, pt)
                    else:
                        # Preserve native V31 dense-selector topk tie behavior.
                        selected = torch.zeros_like(old).scatter_(1, (score + 1.386 * old).topk(states[arm].k, -1).indices, True)
                    states[arm] = StableState(z, selected, cfg)
                    oracle_z = z.cpu().numpy() if oracle_z is None else oracle_z
                    oracles[arm] = NumpyOracle(oracle_z, selected.cpu().numpy(), cfg)
                    pending.pop(arm, None)
                    used = dense
                else:
                    used = output_for(old)
                state = states[arm]
                row_error = (used - dense).norm(dim=-1).reshape(-1)
                if not torch.isfinite(row_error).all():
                    raise RuntimeError('nonfinite operator output')
                errors[arm].append(row_error.cpu())
                if not data['selection_event']:
                    reuse_errors[arm].append(row_error.cpu())
                used_map = state.selected.clone()
                mass = torch.ones_like(tail)[valid] if dense_event else (
                    torch.logaddexp(z.masked_fill(~old[..., None], -math.inf).logsumexp(1), tail) - dense_norm).exp()[valid]
                change = torch.zeros_like(row_error)
                realized_mass = predicted = None
                if arm in pending and not dense_event:
                    previous, predicted = pending.pop(arm)
                    change = (used - output_for(previous)).norm(dim=-1).reshape(-1)
                    old_mass = (torch.logaddexp(z.masked_fill(~previous[..., None], -math.inf).logsumexp(1), tail) - dense_norm).exp()[valid]
                    realized_mass = float((mass - old_mass).mean())
                changes[arm].append(change.cpu())
                swaps = noops = reversals = qk_rectangles = 0
                qk_ms = ranking_ms = 0.
                age = clocks[arm].reuse_steps
                due = due and state.k < pt
                iou_previous = 1.
                if due:
                    ids = tile_ids(state.selected)
                    combined = torch.cat((ids, tail_ids), 1).contiguous()
                    qa, qe = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qa.record()
                    observed = observe(q, k, data['scale'], combined, n)
                    qe.record(); qe.synchronize()
                    qk_ms = qa.elapsed_time(qe)
                    fresh_part = observed[:, :state.k]
                    dense_part = z.gather(1, ids[..., None].expand(-1, -1, 128))
                    stat_error = float(torch.nan_to_num((fresh_part - dense_part).abs()).max())
                    max_stat_error = max(max_stat_error, stat_error)
                    fresh = torch.full_like(z, -math.inf).scatter_(1, ids[..., None].expand(-1, -1, 128), fresh_part)
                    current_tail = observed[:, state.k:].logsumexp(1)
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    prior = state.counts.clone()
                    a.record(); selected, info = state.observe(fresh, current_tail); b.record(); b.synchronize()
                    ranking_ms = a.elapsed_time(b)
                    reference_map, reference_score = oracles[arm].observe(fresh.cpu().numpy(), current_tail.cpu().numpy())
                    gpu_score = info['score'].cpu().numpy()
                    finite = np.isfinite(reference_score) & np.isfinite(gpu_score)
                    score_error = float(np.max(np.abs(reference_score[finite] - gpu_score[finite]), initial=0))
                    max_score_error = max(max_score_error, score_error)
                    if not np.array_equal(selected.cpu().numpy(), reference_map):
                        raise RuntimeError('GPU map differs from independent CPU oracle')
                    oracle_events += 1
                    inc = (state.counts - prior).cpu().tolist()
                    swaps, noops, reversals = int(inc[0]), int(inc[2]), int(inc[7])
                    iou_previous = inc[4] / len(selected)
                    qk_rectangles = combined.numel()
                    if cfg.mode == 'stable' and (int(info['swaps'].max()) > cfg.event_cap or int(state.cumulative.max()) > cfg.cumulative_cap):
                        raise RuntimeError('stable swap cap violated')
                    hybrid = torch.where(used_map[..., None], fresh, state.anchor)
                    norm = torch.logaddexp(hybrid.logsumexp(1), current_tail)
                    def hybrid_mass(sel):
                        return (torch.logaddexp(hybrid.masked_fill(~sel[..., None], -math.inf).logsumexp(1), current_tail) - norm).exp()[valid]
                    pending[arm] = (used_map, float((hybrid_mass(selected) - hybrid_mass(used_map)).mean()))
                    ages[arm] += torch.bincount(state.age.flatten().long().clamp_max(127), minlength=128).cpu()
                if not (state.selected.sum(1) == min(128, pt)).all():
                    raise RuntimeError('fixed prefix budget failed')
                records.append(dict(arm=arm, trajectory=group[0], canvas=data['canvas'], call=data['call'],
                    dataset=data['dataset'], layer=data['layer'], phase=reason, common_reuse=not data['selection_event'],
                    anchor_reuse_age=age, true_retained_mass_mean=float(mass.mean()),
                    row_error_mean=float(row_error.mean()), row_error_max=float(row_error.max()),
                    observation=due, swaps=swaps, unit_noops=noops, immediate_reversals=reversals,
                    iou_previous=iou_previous, selector_qk_rectangles=qk_rectangles, qk_audit_ms=qk_ms,
                    ranking_audit_ms=ranking_ms, predicted_mass_gain=predicted, realized_mass_gain=realized_mass,
                    max_cumulative_swaps=int(state.cumulative.max()),
                    state_bytes=sum(t.numel() * t.element_size() for t in vars(state).values() if isinstance(t, torch.Tensor))))
            del q, k, v, z, dense
        print(json.dumps(dict(group=list(group), common_frames=frames, oracle_events=oracle_events)), flush=True)
    expected = {(ds, layer) for ds in ('longbench_v2_0shot_think', 'aime26') for layer in (5, 11, 17, 23, 29)}
    if coverage != expected or max_stat_error > 3e-4 or max_score_error > 3e-5 or captures['count']:
        raise RuntimeError('operator coverage/parity/capture gate failed')
    aggregate = {}
    for arm, (cfg, period, progress) in configs.items():
        rr = [r for r in records if r['arm'] == arm]
        common = [r for r in rr if r['common_reuse']]
        aggregate[arm] = dict(selector=vars(cfg), interval=period, progress_dense=progress,
            frames=len(rr), common_reuse_frames=len(common), row_error=pooled(errors[arm]),
            common_reuse_row_error=pooled(reuse_errors[arm]), next_call_map_change=pooled(changes[arm]),
            retained_mass_mean=float(np.mean([r['true_retained_mass_mean'] for r in rr])),
            common_reuse_retained_mass_mean=float(np.mean([r['true_retained_mass_mean'] for r in common])),
            swaps=sum(r['swaps'] for r in rr), reversals=sum(r['immediate_reversals'] for r in rr),
            noops=sum(r['unit_noops'] for r in rr), observations=sum(r['observation'] for r in rr),
            dense_progress_events=sum(r['phase'] == 'progress' for r in rr),
            selector_qk_rectangles=sum(r['selector_qk_rectangles'] for r in rr),
            qk_audit_ms=sum(r['qk_audit_ms'] for r in rr), ranking_audit_ms=sum(r['ranking_audit_ms'] for r in rr),
            state_bytes_max=max(r['state_bytes'] for r in rr), max_cumulative_swaps=max(r['max_cumulative_swaps'] for r in rr),
            false_mass_admissions=sum(r['predicted_mass_gain'] is not None and r['predicted_mass_gain'] > 0 and r['realized_mass_gain'] < 0 for r in rr),
            age_histogram_event_weighted={str(i): int(v) for i, v in enumerate(ages[arm]) if v})
    record_path = output / 'records.jsonl'
    record_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    summary = dict(status='complete', schema='complete_factorial_operator_replay_v1',
        configurations=aggregate, snapshots=inventory['step_files'], input_files=inventory['files'], groups=len(groups),
        frames_per_arm=frames, excluded_preanchor_frames=excluded, coverage=sorted(coverage),
        max_fresh_statistic_error=max_stat_error, max_cpu_score_error=max_score_error,
        cpu_map_mismatches=0, cpu_reference_events=oracle_events, budget_errors=0, swap_cap_errors=0,
        cuda_graph_captures=captures['count'], replay_wall_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), identity_sha256=sha(identity_path),
        common_frame_scope='Post common first dense selection; pre-anchor bootstrap/carry calls excluded equally.',
        progress_scope='V31 snapshot progress events imposed; does not predict native progress clocks or accuracy.',
        error_scopes='row_error includes scheduled dense outputs; common_reuse_row_error excludes V31 selection frames equally.',
        clean_timing='paused; CUDA events are selector audit measurements; CPU oracle/replay wall time is not clean timing')
    summary_path = output / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (output / 'final_complete.json').write_text(json.dumps(dict(status='complete', arms=18,
        summary_sha256=sha(summary_path), records_sha256=sha(record_path), identity_sha256=sha(identity_path)), indent=2) + '\n')
    print(json.dumps(dict(status='complete', frames_per_arm=frames, max_cpu_score_error=max_score_error, oracle_events=oracle_events)), flush=True)


if __name__ == '__main__':
    main()
