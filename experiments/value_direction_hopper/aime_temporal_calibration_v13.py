"""Trajectory-based, coordinate-bracketed two-stage temporal calibration.

Reuse the existing runner, router, sampler and counters. Search exhaustion is
not evidence of physical infeasibility. Final evaluation requires a passed audit.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time

from . import aime_temporal_sweep as b

KINDS = ('local', 'global')
PHASES = ('call1', 'call2', 'late')
OLD = Path('/home/exouser/aime_temporal_v12')
ROOT = b.DLMDIR / 'results/query_adaptive_aime_temporal_v13'
BASE_CFG = b._cfg

def config(root):
    out = BASE_CFG(root)
    out.update(schema='aime26_temporal_v13_two_stage_bracket',
               calibration_rule='Calibrate shared early local/global pair, then freeze it and calibrate late pair; audit both early calls, late and pooled O/L/G within 2pp; revisit early if full trajectories drift; never launch on search failure.',
               search_rounds=4, coordinate_trials=9, log_threshold_bounds=[-12, 8])
    out['source_hashes'][str(Path(__file__).resolve())] = b.sha(Path(__file__))
    return out

b._cfg = config

def get_value(policy, phase, kind):
    key = 'call1' if phase == 'early' else phase
    return policy[key][kind]['log_threshold']

def set_value(policy, phase, kind, value):
    out = copy.deepcopy(policy)
    for key in (('call1', 'call2') if phase == 'early' else (phase,)):
        out[key][kind]['log_threshold'] = float(value)
    return out

def coordinate_rates(metrics, phase, kind):
    return [metrics['phase_sparsity'][p][kind]
            for p in (('call1', 'call2') if phase == 'early' else ('late',))]

def coordinate_error(metrics, phase, kind, goal):
    # Never allow the untouched attention type to mask this coordinate's error.
    return max(abs(x-goal) for x in coordinate_rates(metrics, phase, kind))

def violations(metrics, goal):
    failures = {}
    for phase in (*PHASES, 'pooled'):
        rates = metrics['sparsity'] if phase == 'pooled' else metrics['phase_sparsity'][phase]
        for kind in ('whole', *KINDS):
            if abs(rates[kind] - goal) > .02:
                failures[f'{phase}_{kind}'] = rates[kind]-goal
    return failures

def initial_policy(target):
    """Old measured curves supply seeds only, never final threshold decisions."""
    trace = json.loads((OLD/'calibration_traces'/f'temporal_s{target}.json').read_text())
    goal = target/100
    policy = copy.deepcopy(trace[0]['policy'])
    for phase in ('early', 'late'):
        for kind in KINDS:
            samples = []
            for p in trace:
                value = get_value(p['policy'], phase, kind)
                rates = coordinate_rates(p['metrics'], phase, kind)
                samples.append((value, sum(rates)/len(rates)))
            below = [x for x in samples if x[1] < goal]
            above = [x for x in samples if x[1] >= goal]
            value = min(samples, key=lambda x: abs(x[1]-goal))[0]
            if below and above:
                lo = min(below, key=lambda x: goal-x[1])
                hi = min(above, key=lambda x: x[1]-goal)
                if lo[0] < hi[0] and hi[1] > lo[1]:
                    value = lo[0]+(hi[0]-lo[0])*(goal-lo[1])/(hi[1]-lo[1])
            policy = set_value(policy, phase, kind, value)
    return policy

def calibrate_one(adapter, root, cfg, rows, projections, target):
    goal = target/100
    points = []
    memo = {}
    def evaluate(policy):
        key = b._hash(policy)[:20]
        if key in memo:
            return memo[key]
        results = [b._cached(adapter, root, 'calibration', f'temporal_s{target}_{key}',
                            row, 'temporal', policy, cfg, projections) for row in rows]
        for r in results:
            for step in r['step_records']:
                if step.get('iteration', 99) <= 2 and step.get('weight_active'):
                    raise AssertionError('Temporal history leaked into early calls')
        metrics = b._profile(results)
        p = dict(policy=policy, metrics=metrics, violations=violations(metrics, goal))
        points.append(p); memo[key] = p
        b._write(root/'calibration_traces'/f'temporal_s{target}.json', points)
        print(json.dumps(dict(event='candidate', target=target, key=key,
                              sparsity=metrics['sparsity'], phases=metrics['phase_sparsity'],
                              violations=p['violations'])), flush=True)
        return p

    def solve(current, phase, kind):
        base = current['policy']
        samples = [current]
        # All samples in a coordinate solve use exactly the same other thresholds.
        for attempt in range(cfg['coordinate_trials']):
            best = min(samples, key=lambda p: coordinate_error(p['metrics'], phase, kind, goal))
            if coordinate_error(best['metrics'], phase, kind, goal) <= .012:
                return best
            measured = [(get_value(p['policy'], phase, kind),
                         sum(coordinate_rates(p['metrics'], phase, kind))/len(coordinate_rates(p['metrics'], phase, kind)))
                        for p in samples]
            lower = [x for x in measured if x[1] < goal]
            upper = [x for x in measured if x[1] >= goal]
            brackets = [(lo, hi) for lo in lower for hi in upper if lo[0] < hi[0]]
            if brackets:
                lo, hi = min(brackets, key=lambda pair: pair[1][0]-pair[0][0])
                fraction = min(.8, max(.2, (goal-lo[1])/(hi[1]-lo[1])))
                value = lo[0]+fraction*(hi[0]-lo[0])
            else:
                x, rate = min(measured, key=lambda x: abs(x[1]-goal))
                # Expand instead of interpreting an unbracketed target as impossible.
                delta = min(2., .35*2**(attempt//2))
                value = x + (delta if rate < goal else -delta)
            value = min(8., max(-12., value))
            if any(abs(value-x)<1e-5 for x, _ in measured):
                break
            samples.append(evaluate(set_value(base, phase, kind, value)))
        return min(samples, key=lambda p: coordinate_error(p['metrics'], phase, kind, goal))

    current = evaluate(initial_policy(target))
    for outer in range(cfg['search_rounds']):
        # Stage 1: solve the early pair independently of late pooled errors.
        for _ in range(2):
            for kind in KINDS:
                current = solve(current, 'early', kind)
            if all(coordinate_error(current['metrics'], 'early', k, goal)<=.02 for k in KINDS):
                break
        early_pass = all(coordinate_error(current['metrics'], 'early', k, goal)<=.02 for k in KINDS)
        b._write(root/'stage_checks'/f'temporal_s{target}_round{outer}.json',
                 dict(early_pass=early_pass, point=current))
        if not early_pass:
            continue
        # Stage 2: freeze early; target late local/global directly, then audit pooled.
        for _ in range(2):
            for kind in KINDS:
                current = solve(current, 'late', kind)
            if not current['violations']:
                break
        if not current['violations']:
            break
        # Later thresholds can change later canvases, so early rates are rechecked.
    feasible = [p for p in points if not p['violations']]
    selected = min(feasible or points, key=lambda p: max([abs(x) for x in p['violations'].values()] or [0]))
    out = dict(fingerprint=cfg['fingerprint'], target=target, method='temporal',
               policy=selected['policy'], selected_metrics=selected['metrics'],
               violations=selected['violations'], calibration_ids=[r['id'] for r in rows],
               status='attained' if feasible else 'search_incomplete', candidates=len(points))
    b._write(root/'thresholds'/f'temporal_s{target}.json', out)
    return out

def calibrate(root):
    cfg, _, rows = b.prepare(root)
    adapter = b._adapter(cfg); projections = b.Projections()
    outcomes = []
    for target in b.TARGETS:
        path = root/'thresholds'/f'temporal_s{target}.json'
        old = json.loads(path.read_text()) if path.exists() else None
        if old and old['fingerprint']==cfg['fingerprint'] and old['status']=='attained':
            outcomes.append(old)
        else:
            outcomes.append(calibrate_one(adapter, root, cfg, rows, projections, target))
    passed = all(x['status']=='attained' for x in outcomes)
    b._write(root/'calibration_audit.json', dict(passed=passed, targets=b.TARGETS, outcomes=outcomes))
    if passed:
        b._write(root/'calibration_complete.json', dict(passed=True, finished=time.time()))
    else:
        raise RuntimeError('Calibration search incomplete; final evaluation is forbidden')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','calibrate'])
    parser.add_argument('--root', type=Path, default=ROOT); args=parser.parse_args()
    if args.stage=='prepare': b.prepare(args.root)
    else: calibrate(args.root)

if __name__=='__main__': main()
