"""Residual phase calibration and strictly audited three-seed execution.

Preserve v13. Its shared early pair seeds this refinement; if it misses the
phase gates, fit call1/call2 separately before fitting late thresholds.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

from . import aime_temporal_calibration_v13 as v

b = v.b
SOURCE = v.ROOT
ROOT = b.DLMDIR/'results/query_adaptive_aime_temporal_v14'
V13_CONFIG = b._cfg

def config(root):
    existing = root / "configuration.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old.get("schema") == "aime26_v14_residual_two_stage":
            return old
    cfg = V13_CONFIG(root)
    cfg.update(schema='aime26_v14_residual_two_stage',
               calibration_rule='Refine v13 calibration trajectories: stage1 fits call1/call2 local/global independently when a shared pair misses; stage2 freezes early and fits late; revalidate phase and pooled O/L/G within 2pp. Baselines use their existing schedules and pooled O/L/G matching only.',
               coordinate_trials=8, search_rounds=5)
    cfg['source_hashes'][str(Path(__file__).resolve())]=b.sha(Path(__file__))
    return cfg

b._cfg=config

def failures(metrics, target, temporal):
    if temporal:
        return v.violations(metrics,target/100)
    return {k:metrics['sparsity'][k]-target/100 for k in ('whole','local','global')
            if abs(metrics['sparsity'][k]-target/100)>.02}

def calibrate_one(adapter, root, cfg, rows, projections, method, target):
    temporal=method=='temporal'; goal=target/100
    # Temporal v13 did not necessarily materialize every target. Resume
    # from an existing v14 point for this target, otherwise use the nearest
    # available temporal policy as a starting point for coordinate search.
    if temporal:
        candidates = [root/'thresholds'/f'{method}_s{target}.json',
                      SOURCE/'thresholds'/f'{method}_s{target}.json']
        candidates += [SOURCE/'thresholds'/f'{method}_s{t}.json'
                       for t in (30, 40, 50, 60, 70)]
    else:
        candidates = [b.PRIOR_ROOT/'thresholds'/f'{method}_s{target}.json']
    parent = next((p for p in candidates if p.exists()), candidates[-1])
    seed=json.loads(parent.read_text()); initial=seed['policy']
    points=[]; memo={}
    def evaluate(policy):
        key=b._hash(policy)[:20]
        if key in memo: return memo[key]
        outputs=[b._cached(adapter,root,'calibration',f'{method}_s{target}_{key}',r,
                           method,policy,cfg,projections) for r in rows]
        metrics=b._profile(outputs)
        point=dict(policy=policy,metrics=metrics,violations=failures(metrics,target,temporal))
        memo[key]=point; points.append(point)
        b._write(root/'calibration_traces'/f'{method}_s{target}.json',points)
        print(json.dumps(dict(event='candidate',method=method,target=target,key=key,
                              sparsity=metrics['sparsity'],phases=metrics['phase_sparsity'],
                              violations=point['violations'])),flush=True)
        return point
    def rate(point,phase,kind):
        return (point['metrics']['sparsity'] if phase=='all' else
                point['metrics']['phase_sparsity'][phase])[kind]
    def solve(current,phase,kind):
        if abs(rate(current,phase,kind)-goal)<=.012: return current
        # Offsets preserve the pre-existing baseline threshold schedule.
        base=current['policy']; trials=[(0.,current)]
        for attempt in range(cfg['coordinate_trials']):
            best=min(trials,key=lambda x:abs(rate(x[1],phase,kind)-goal))
            if abs(rate(best[1],phase,kind)-goal)<=.012: return best[1]
            low=[x for x in trials if rate(x[1],phase,kind)<goal]
            high=[x for x in trials if rate(x[1],phase,kind)>=goal]
            brackets=[(lo,hi) for lo in low for hi in high if lo[0]<hi[0]]
            if brackets:
                lo,hi=min(brackets,key=lambda p:p[1][0]-p[0][0])
                fraction=(goal-rate(lo[1],phase,kind))/(rate(hi[1],phase,kind)-rate(lo[1],phase,kind))
                offset=lo[0]+min(.8,max(.2,fraction))*(hi[0]-lo[0])
            else:
                offset=best[0]+(.12*2**(attempt//2))*(1 if rate(best[1],phase,kind)<goal else -1)
            if any(abs(offset-x[0])<1e-6 for x in trials): break
            policy=copy.deepcopy(base)
            for ph in (tuple(policy) if phase=='all' else (phase,)):
                field=next(iter(policy[ph][kind]))
                policy[ph][kind][field]+=offset
            trials.append((offset,evaluate(policy)))
        return min(trials,key=lambda x:abs(rate(x[1],phase,kind)-goal))[1]
    current=evaluate(initial)
    for outer in range(cfg['search_rounds']):
        if not current['violations']: break
        stages=(('call1','call2'),('late',)) if temporal else (('all',),)
        for stage in stages:
            for phase in stage:
                for kind in v.KINDS:
                    current=solve(current,phase,kind)
            # Early thresholds must pass before advancing to stage 2.
            if temporal and stage==('call1','call2') and any(
                    abs(rate(current,p,k)-goal)>.02 for p in stage for k in v.KINDS):
                break
    good=[p for p in points if not p['violations']]
    selected=min(good or points,key=lambda p:max(map(abs,p['violations'].values()),default=0))
    out=dict(fingerprint=cfg['fingerprint'],method=method,target=target,
             policy=selected['policy'],selected_metrics=selected['metrics'],
             violations=selected['violations'],status='attained' if good else 'search_incomplete',
             seed_policy_source=str(parent),calibration_ids=[r['id'] for r in rows])
    b._write(root/'thresholds'/f'{method}_s{target}.json',out)
    return out

def audit(root):
    cfg=json.loads((root/'configuration.json').read_text()); checked=[]
    for method in b.METHODS:
        for target in b.TARGETS:
            p=root/'thresholds'/f'{method}_s{target}.json'
            d=json.loads(p.read_text())
            assert d['fingerprint']==cfg['fingerprint'], str(p)
            assert d['status']=='attained' and not failures(d['selected_metrics'],target,method=='temporal'),str(p)
            checked.append(dict(method=method,target=target,sha=b.sha(p)))
    return checked

def calibrate(root):
    cfg,_,rows=b.prepare(root); adapter=b._adapter(cfg); projections=b.Projections()
    for method in b.METHODS:
        for target in b.TARGETS:
            path=root/'thresholds'/f'{method}_s{target}.json'
            if path.exists():
                d=json.loads(path.read_text())
                if d['fingerprint']==cfg['fingerprint'] and d['status']=='attained': continue
            calibrate_one(adapter,root,cfg,rows,projections,method,target)
    checked=audit(root)
    b._write(root/'calibration_complete.json',dict(passed=True,thresholds=checked,finished=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','run','report']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if a.stage=='calibrate':calibrate(a.root)
    elif a.stage=='run':
        audit(a.root)
        os.environ['AIME_FINAL_SEEDS']='42,43,44'
        b.run_final(a.root)
    else:b.report(a.root)

if __name__=='__main__':main()
