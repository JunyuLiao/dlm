"""Versioned, calibration-only refinement if the frozen v1 T70 grid misses.

This is a separate policy-search module so v1 code hashes and failed trials
remain intact. Its candidate grid is informed by *v1 calibration*, never by
the current 130-prompt final answers. It reuses the unchanged v1 execution
path and native decoder.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.diffusion_gemma_jl_output_aware.projections import Projections

from .experiment import atomic, fingerprint, sha
from . import query_adaptive_guardrail as core


ROOT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v2'
SOURCE=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1'


def candidates(target):
    old=json.loads((core.ARCHIVE/'configs/thresholds'/f'T_s{target}.json').read_text())['policy']
    ol,og=(old[k]['log_threshold'] for k in ('local','global'))
    early=core.pair(ol-.15,og-.15) if target==70 else core.pair(ol,og)
    if target==70:
        offsets=((0,.05),(0,.10),(0,.15),(0,.20),
                 (-.07,.10),(.07,.10),(-.07,.15),(.07,.15),
                 (0,-.07),(-.07,.05),(.07,.05))
    else:
        offsets=((0,-.15),(0,-.08),(0,.08),(0,.15),
                 (-.07,0),(.07,0),(-.07,.08),(.07,.08))
    return core._unique_policies([
        core.phase_policy(early,core.pair(ol+dl,og+dg)) for dl,dg in offsets])


def _prepare(root):
    root=Path(root)
    cfg,manifests=core.prepare(root)
    source_path=SOURCE/'calibration_traces/T_s70.json'
    if not source_path.exists():raise ValueError('v1 calibration must finish first')
    plan=dict(schema='trajectory_guardrail_refinement_v2',
        core_fingerprint=cfg['fingerprint'],source_sha256=sha(Path(__file__)),
        v1_calibration_trace_sha256=sha(source_path),
        source='v1 calibration and disjoint validation evidence: the old-minus-0.15 '
            'early pair passes the early ceilings; late global values between '
            'archived and fresh pairs may resolve the validation p90/global trade-off',
        candidate_policies={str(t):candidates(t) for t in core.TARGETS},
        final_scores_used_for_search=False)
    plan['fingerprint']=fingerprint(plan)
    path=root/'configs/refinement_configuration.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Refinement source changed; choose another result root')
    atomic(path,plan)
    return cfg,manifests,plan


def calibrate(adapter,root,cfg,manifests,projections,plan,target):
    condition=core.label('T',target)
    dest=Path(root)/'configs/thresholds'/f'{condition}.json'
    if dest.exists():
        frozen=json.loads(dest.read_text())
        if frozen.get('refinement_fingerprint')!=plan['fingerprint']:
            raise ValueError('Frozen refinement provenance mismatch')
        return frozen
    source_threshold=SOURCE/'configs/thresholds'/f'{condition}.json'
    if target==50 and source_threshold.exists():
        previous=json.loads(source_threshold.read_text())
        if previous['status']=='attained' and previous['fingerprint']==cfg['fingerprint']:
            frozen=dict(previous,refinement_fingerprint=plan['fingerprint'],
                reused_from_v1=str(source_threshold.resolve()))
            atomic(dest,frozen)
            return frozen
    points=[]
    for policy in candidates(target):
        item=core._evaluate_policy(adapter,root,cfg,manifests['calibration'],
                                   projections,'calibration','T',target,policy)
        points.append(item);core._write_trace(root,condition,points)
        print(json.dumps(dict(event='refinement_point',condition=condition,
            key=item['key'],overall=item['metrics']['overall'],
            mean_steps=item['metrics']['mean_steps'],
            p90_steps=item['metrics']['p90_steps'],
            violations=item['violations'])),flush=True)
    goal=target/100
    feasible=sorted((p for p in points if not p['violations']),
                    key=lambda p:core.rank_point(p,goal))
    ranked=feasible or sorted(points,key=lambda p:core.rank_point(p,goal))
    unpruned=core.phase_policy(core.pair(float('-inf'),float('-inf')))
    dense=[core.cached(adapter,root,'validation_dense','kernel_dense',None,row,
        unpruned,cfg,projections) for row in manifests['validation']]
    dense_info=core.profile(dense)
    atomic(Path(root)/'configs/validation_dense.json',dict(
        ids=[r['id'] for r in manifests['validation']],metrics=dense_info,
        source='same matched v4 H100 kernel, all eligible tiles retained'))
    validation=[];selected=None
    for point in ranked[:5]:
        item=core._evaluate_policy(adapter,root,cfg,manifests['validation'],
            projections,'validation','T',target,point['policy'])
        item['violations']=core.violations(item['metrics'],target,validation=True)
        if item['metrics']['accuracy']<dense_info['accuracy']-.03:
            item['violations'].append('accuracy_drop_gt_3pp')
        validation.append(item)
        if not item['violations'] and not point['violations']:
            selected=point;break
    if selected is None:selected=ranked[0]
    status='attained' if selected in feasible and any(
        v['policy']==selected['policy'] and not v['violations'] for v in validation
    ) else 'unattainable_under_guardrails'
    frozen=dict(fingerprint=cfg['fingerprint'],refinement_fingerprint=plan['fingerprint'],
        condition=condition,method='T',target=target,status=status,
        policy=selected['policy'],calibration=selected,validation=validation,
        calibration_ids=[r['id'] for r in manifests['calibration']],
        validation_ids=[r['id'] for r in manifests['validation']],
        selection='v2 fixed calibration grid; validation before final; no final score tuning')
    atomic(dest,frozen)
    return frozen


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests,plan=_prepare(root)
        adapter=core._adapter(cfg);projections=Projections()
        core.smoke(adapter,root,cfg,manifests,projections)
        for target in (70,50):
            calibrate(adapter,root,cfg,manifests,projections,plan,target)


def launch(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m',
        'experiments.value_direction_hopper.query_adaptive_guardrail_refine',
        'run','--root',str(root)]
    logfile=root/f'refine_{int(time.time())}.log'
    with logfile.open('xb') as output:
        worker=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],
            env=env,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(root/'refine_launch.json',dict(pid=worker.pid,log=str(logfile),
        command=command,started=time.time()))
    print(json.dumps(dict(pid=worker.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('run','launch'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.stage=='launch':launch(args.root)
    else:run(args.root)
