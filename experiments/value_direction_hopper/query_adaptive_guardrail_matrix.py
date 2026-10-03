"""Frozen-phase query-sensitivity comparison for the trajectory guardrail study.

This module deliberately imports the v1 execution path instead of copying the
decoder or attention implementation. The threshold search uses complete native
generations on disjoint calibration and validation prompts, never final scores.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.diffusion_gemma_jl_output_aware.projections import Projections

from .experiment import atomic, fingerprint, sha
from . import query_adaptive_guardrail as core


METHODS = ('unweighted', 'C', 'M', 'T_uniform', 'T_tile_mean', 'T_shuffle')


def matrix_config(root, cfg):
    root = Path(root)
    path = root/'configs/matrix_configuration.json'
    spec = dict(schema='trajectory_guardrail_matrix_v1', core_fingerprint=cfg['fingerprint'],
        source_sha256=sha(Path(__file__)), methods=list(METHODS), targets=list(core.TARGETS),
        early_policy='identical frozen T early local/global pair for every method at the same target',
        late_search='complete-generation, bounded frozen log-threshold offsets around T',
        selection='calibration-feasible minimum max whole/local/global target deviation; validation before final',
        final_scores_used_for_selection=False)
    spec['fingerprint'] = fingerprint(spec)
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError('Matrix specification changed; use a new versioned result root')
    else:
        atomic(path, spec)
    return spec


def _shifted_policy(early, late, local_shift, global_shift):
    base = core.pair(late['local']['log_threshold']+local_shift,
                     late['global']['log_threshold']+global_shift)
    return core.phase_policy(early, base)


def candidate_policies(method, t_policy):
    early, late = t_policy['early'], t_policy['late']
    shifts = {'unweighted':(-.55,-.35,-.15,.05,.25,.45),
              'C':(-.1,.15,.4,.65,.9,1.15),
              'M':(-.1,.15,.4,.65,.9,1.15),
              'T_uniform':(-.3,-.1,.1,.3,.5,.7),
              'T_tile_mean':(-.3,-.1,.1,.3,.5,.7),
              'T_shuffle':(-.3,-.1,.1,.3,.5,.7)}[method]
    return core._unique_policies([_shifted_policy(early,late,s,s) for s in shifts])


def refine_policies(best, early):
    late = best['policy']['late']
    return core._unique_policies([
        core.phase_policy(early, core.pair(late['local']['log_threshold']+dl,
                                          late['global']['log_threshold']+dg))
        for dl,dg in ((-.2,0),(.2,0),(0,-.2),(0,.2),(-.1,-.1),(.1,.1))])


def _rank(point, goal):
    info=point['metrics']
    return (bool(point['violations']),core.goal_error(info,goal),
            info['mean_steps'],info['executed_tiles'])


def calibrate_one(adapter, root, cfg, manifests, projections, method, target):
    condition=core.label(method,target)
    path=Path(root)/'configs/thresholds'/f'{condition}.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen['fingerprint']!=cfg['fingerprint']:
            raise ValueError(f'Frozen fingerprint mismatch: {condition}')
        return frozen
    t=core.calibrate_t(adapter,root,cfg,manifests,projections,target)
    if t['status']!='attained':
        frozen=dict(fingerprint=cfg['fingerprint'],condition=condition,
            status='not_run_T_anchor_unattainable',target=target,method=method)
        atomic(path,frozen)
        return frozen
    goal=t['calibration']['metrics']['overall']
    cal,val=manifests['calibration'],manifests['validation']
    points=[]

    def evaluate(policy):
        point=core._evaluate_policy(adapter,root,cfg,cal,projections,
            'calibration',method,target,policy)
        point['violations']=core.violations(point['metrics'],target,goal=goal)
        points.append(point)
        core._write_trace(root,condition,points)
        print(json.dumps(dict(event='matrix_calibration_point',condition=condition,
            key=point['key'],overall=point['metrics']['overall'],
            early=point['metrics']['phase_sparsity'],
            mean_steps=point['metrics']['mean_steps'],
            violations=point['violations'])),flush=True)

    for policy in candidate_policies(method,t['policy']):
        evaluate(policy)
    feasible=sorted((p for p in points if not p['violations']),key=lambda p:_rank(p,goal))
    if not feasible:
        anchor=min(points,key=lambda p:_rank(p,goal))
        for policy in refine_policies(anchor,t['policy']['early']):
            if any(p['policy']==policy for p in points):continue
            evaluate(policy)
        feasible=sorted((p for p in points if not p['violations']),key=lambda p:_rank(p,goal))
    ranked=feasible or sorted(points,key=lambda p:_rank(p,goal))
    dense=json.loads((Path(root)/'configs/validation_dense.json').read_text())['metrics']
    validation=[];selected=None
    for point in ranked[:4]:
        item=core._evaluate_policy(adapter,root,cfg,val,projections,'validation',
                                   method,target,point['policy'])
        item['violations']=core.violations(item['metrics'],target,
                                           validation=True,goal=goal)
        if item['metrics']['accuracy']<dense['accuracy']-.03:
            item['violations'].append('accuracy_drop_gt_3pp')
        validation.append(item)
        if not item['violations'] and not point['violations']:
            selected=point
            break
    if selected is None:selected=ranked[0]
    status='attained' if selected in feasible and any(
        v['policy']==selected['policy'] and not v['violations'] for v in validation
    ) else 'unattainable_under_guardrails'
    frozen=dict(fingerprint=cfg['fingerprint'],condition=condition,method=method,
        target=target,status=status,policy=selected['policy'],
        matched_T_goal=goal,calibration=selected,validation=validation,
        calibration_ids=[r['id'] for r in cal],validation_ids=[r['id'] for r in val],
        tested_points=len(points),source='frozen shared early policy; method-specific late pair')
    atomic(path,frozen)
    return frozen


def _run_rows(adapter,root,cfg,rows,projections,stage,method,target,policy):
    failures=[]
    for row in rows:
        try:core.cached(adapter,root,stage,method,target,row,policy,cfg,projections)
        except Exception as error:
            failures.append(dict(stage=stage,condition=core.label(method,target),
                id=row['id'],error=repr(error),traceback=traceback.format_exc()))
            path=Path(root)/'matrix_failures.json'
            prior=json.loads(path.read_text()) if path.exists() else []
            atomic(path,prior+[failures[-1]])
            if 'illegal memory' in str(error) or 'device-side assert' in str(error):
                raise
    return failures


def run(root,stage):
    root=Path(root)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests=core.prepare(root)
        matrix_config(root,cfg)
        adapter=core._adapter(cfg);projections=Projections()
        core.smoke(adapter,root,cfg,manifests,projections)
        if stage=='calibrate':
            for target in core.TARGETS:
                for method in METHODS:
                    try:calibrate_one(adapter,root,cfg,manifests,projections,method,target)
                    except Exception as error:
                        error_path=root/'matrix_failures.json'
                        errors=json.loads(error_path.read_text()) if error_path.exists() else []
                        errors.append(dict(condition=core.label(method,target),
                            error=repr(error),traceback=traceback.format_exc()))
                        atomic(error_path,errors)
                        if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
        elif stage=='final':
            for target in core.TARGETS:
                for method in ('T',)+METHODS:
                    path=root/'configs/thresholds'/f'{core.label(method,target)}.json'
                    if not path.exists():
                        raise ValueError(f'Missing frozen calibration: {path}')
                    frozen=json.loads(path.read_text())
                    if 'policy' not in frozen:
                        print(json.dumps(dict(event='skip_unattainable',
                            condition=core.label(method,target),status=frozen['status'])),flush=True)
                        continue
                    if frozen['status']!='attained':
                        print(json.dumps(dict(event='evaluate_target_miss',
                            condition=core.label(method,target),status=frozen['status'])),flush=True)
                    _run_rows(adapter,root,cfg,manifests['final'],projections,
                              'final',method,target,frozen['policy'])
        elif stage=='run-anchor':
            frozen=core.calibrate_t(adapter,root,cfg,manifests,projections,70)
            print(json.dumps(dict(event='anchor_status',status=frozen['status'],
                violations=frozen['calibration']['violations'])),flush=True)
            _run_rows(adapter,root,cfg,manifests['final'],projections,
                      'final','T',70,frozen['policy'])
        elif stage=='anchor-and-archive':
            frozen=core.calibrate_t(adapter,root,cfg,manifests,projections,70)
            print(json.dumps(dict(event='anchor_status',status=frozen['status'],
                violations=frozen['calibration']['violations'])),flush=True)
            _run_rows(adapter,root,cfg,manifests['final'],projections,
                      'final','T',70,frozen['policy'])
            _run_rows(adapter,root,cfg,manifests['final'],projections,
                      'archived_reproduction','T',70,core.phase_policy(cfg['old_t70']))
        elif stage=='confirm':
            frozen=json.loads((root/'configs/thresholds/T_s70.json').read_text())
            if frozen['status']!='attained':raise ValueError('T70 not attained')
            for seed in cfg['confirmation_seeds']:
                rows=[dict(r,seed=seed) for r in manifests['final']]
                _run_rows(adapter,root,cfg,rows,projections,
                          f'confirmation/seed{seed}','T',70,frozen['policy'])
        elif stage=='reproduce-old':
            # Predeclared archived T70 policy, never selected from these final
            # answers. It is a direct cross-cohort reproduction reference.
            policy=core.phase_policy(cfg['old_t70'])
            _run_rows(adapter,root,cfg,manifests['final'],projections,
                      'archived_reproduction','T',70,policy)
        else:raise ValueError(stage)


def launch(root,stage):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m',
        'experiments.value_direction_hopper.query_adaptive_guardrail_matrix',
        stage,'--root',str(root)]
    logfile=root/f'matrix_{stage}_{int(time.time())}.log'
    with logfile.open('xb') as out:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],
            env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    atomic(root/f'matrix_{stage}_launch.json',dict(pid=process.pid,log=str(logfile),
        command=command,started=time.time()))
    print(json.dumps(dict(pid=process.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('calibrate','final','confirm','reproduce-old','run-anchor',
        'anchor-and-archive','launch-calibrate','launch-final','launch-confirm',
        'launch-reproduce-old','launch-run-anchor','launch-anchor-and-archive'))
    parser.add_argument('--root',type=Path,default=core.ROOT)
    args=parser.parse_args()
    if args.stage.startswith('launch-'):
        launch(args.root,args.stage.removeprefix('launch-'))
    else:run(args.root,args.stage)
