"""Resumable, phase-guarded beta/gamma sweep for temporal query sensitivity.

The router, H100 kernel and native DiffusionGemma decoder are imported from
the audited trajectory-guardrail experiment. This module changes only the
frozen beta/gamma constants and calibrates one local/global late pair for
each grid point on disjoint development prompts.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.diffusion_gemma_jl_output_aware.projections import Projections

from . import query_adaptive_guardrail as core
from .experiment import atomic, fingerprint, sha, shard_path


ROOT = Path(__file__).resolve().parents[2]/'results'/'query_adaptive_temporal_sweep_v1'
REFERENCE = Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v2'
BETAS = (0., 1., 3., 6.)
GAMMAS = (0., .5, .9)
GRID = ((0., 0.),) + tuple((beta, gamma) for beta in BETAS[1:]
                              for gamma in GAMMAS)
SHIFTS = (-1.2, -.8, -.4, -.15, 0., .2, .5, .9)
REFINEMENTS = ((-.2, 0.), (.2, 0.), (0., -.2), (0., .2),
               (-.1, -.1), (.1, .1))


def label(beta, gamma, target):
    def number(value):
        return f'{value:g}'.replace('.', 'p')
    return f'b{number(beta)}_g{number(gamma)}_s{target}'


def combinations():
    return tuple((beta, gamma, target) for target in core.TARGETS
                 for beta, gamma in GRID)


def condition_root(root, beta, gamma, target):
    return Path(root)/'conditions'/label(beta, gamma, target)


def _reference_policy(target):
    path = REFERENCE/'configs/thresholds'/f'T_s{target}.json'
    frozen = json.loads(path.read_text())
    if frozen['status'] != 'attained':
        raise ValueError(f'Previous T{target} anchor did not attain its guardrails')
    return frozen


def _goal(reference):
    return dict(reference['calibration']['metrics']['overall'])


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base, manifests = core.prepare(REFERENCE)
    audit = json.loads((REFERENCE/'audit.json').read_text())
    if not audit['complete'] or not audit['timing_complete']:
        raise ValueError('Audited previous result and timing are required')
    reference_files = [REFERENCE/'configs/configuration.json',
                       REFERENCE/'configs/dataset_audit.json']
    reference_files += [REFERENCE/'configs'/f'{part}_manifest.json'
                        for part in ('calibration', 'validation', 'smoke', 'final')]
    reference_files += [REFERENCE/'configs/thresholds'/f'T_s{target}.json'
                        for target in core.TARGETS]
    specification = dict(schema='temporal_sensitivity_sweep_v1',
        source_sha256=sha(Path(__file__)),
        core_source_sha256=sha(Path(core.__file__)),
        base_fingerprint=base['fingerprint'],
        reference_hashes={str(p.resolve()): sha(p) for p in reference_files},
        grid=[dict(beta=beta, gamma=gamma) for beta, gamma in GRID],
        targets=list(core.TARGETS), shifts=list(SHIFTS),
        refinements=[list(x) for x in REFINEMENTS],
        early='common frozen previous T early pair; unit query weights for calls 1-2',
        goals={str(target): _goal(_reference_policy(target))
               for target in core.TARGETS},
        calibration='complete native generations; same O/G/L and trajectory guardrails',
        validation='disjoint 26 prompts; top four calibration-ranked points',
        final_scores_used_for_policy_selection=False)
    specification['fingerprint'] = fingerprint(specification)
    path = root/'configs/specification.json'
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError('Frozen sweep source or reference changed; choose another root')
    else:
        atomic(path, specification)
        for split, rows in manifests.items():
            atomic(root/f'configs/{split}_manifest.json', rows)
    for split, rows in manifests.items():
        saved = json.loads((root/f'configs/{split}_manifest.json').read_text())
        if [(x['id'], x['prompt_hash'], x['seed']) for x in saved] != [
            (x['id'], x['prompt_hash'], x['seed']) for x in rows]:
            raise ValueError(f'Frozen {split} manifest changed')
    return base, manifests, specification


def config_for(base, specification, beta, gamma, target):
    if (beta, gamma) not in GRID or target not in core.TARGETS:
        raise ValueError('Parameter combination not in predeclared grid')
    cfg = dict(base, beta=float(beta), gamma=float(gamma))
    cfg['fingerprint'] = fingerprint([base['fingerprint'],
        specification['fingerprint'], beta, gamma, target])
    return cfg


def candidate_policies(target):
    reference = _reference_policy(target)['policy']
    early, late = reference['early'], reference['late']
    return core._unique_policies([core.phase_policy(early, core.pair(
        late['local']['log_threshold'] + shift,
        late['global']['log_threshold'] + shift)) for shift in SHIFTS])


def refined_policies(anchor):
    late = anchor['late']
    return core._unique_policies([core.phase_policy(anchor['early'], core.pair(
        late['local']['log_threshold'] + dl,
        late['global']['log_threshold'] + dg))
        for dl, dg in REFINEMENTS])


def rank_point(point, goal):
    info = point['metrics']
    return (bool(point['violations']), core.goal_error(info, goal),
            info['mean_steps'], info['executed_tiles'])


def smoke(root, adapter, base, manifests, specification, projections):
    destination = Path(root)/'configs/smoke.json'
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get('fingerprint') != specification['fingerprint'] or not saved.get('passed'):
            raise ValueError('Prior smoke provenance mismatch')
        return saved
    policy = _reference_policy(70)['policy']
    checks = []
    for row in manifests['final'][:2]:
        cfg = config_for(base, specification, 3., .5, 70)
        result, _ = core.generate(adapter, row, cfg['methods']['T'],
                                  policy, cfg, projections, diagnostics=False)
        previous = json.loads(shard_path(REFERENCE/'final', 'T_s70', row).read_text())
        if (result['steps'], result['completion_tokens']) != (
            previous['steps'], previous['completion_tokens']):
            raise AssertionError(f'Beta3/gamma0.5 reference parity failed: {row["id"]}')
        cfg_zero = config_for(base, specification, 0., 0., 70)
        zero, _ = core.generate(adapter, row, cfg_zero['methods']['T'],
                                policy, cfg_zero, projections, diagnostics=False)
        plain, _ = core.generate(adapter, row, base['methods']['unweighted'],
                                 policy, base, projections, diagnostics=False)
        if (zero['steps'], zero['completion_tokens']) != (
            plain['steps'], plain['completion_tokens']):
            raise AssertionError(f'Beta0/unweighted parity failed: {row["id"]}')
        checks.append(dict(id=row['id'], beta3_reference_parity=True,
                           beta0_unweighted_parity=True, calls=result['steps']))
    saved = dict(fingerprint=specification['fingerprint'], passed=True, checks=checks)
    atomic(destination, saved)
    return saved


def calibrate_one(root, adapter, base, manifests, specification,
                  projections, beta, gamma, target):
    name = label(beta, gamma, target)
    branch = condition_root(root, beta, gamma, target)
    destination = branch/'configs/threshold.json'
    cfg = config_for(base, specification, beta, gamma, target)
    if destination.exists():
        frozen = json.loads(destination.read_text())
        if (frozen['config_fingerprint'] != cfg['fingerprint'] or
            frozen['specification_fingerprint'] != specification['fingerprint']):
            raise ValueError(f'Frozen calibration changed: {name}')
        return frozen
    goal = specification['goals'][str(target)]
    points = []

    def evaluate(policy):
        point = core._evaluate_policy(adapter, branch, cfg,
            manifests['calibration'], projections, 'calibration', 'T', target, policy)
        point['violations'] = core.violations(point['metrics'], target, goal=goal)
        points.append(point)
        atomic(branch/'calibration_trace.json', points)
        print(json.dumps(dict(event='calibration_point', condition=name,
            key=point['key'], rates=point['metrics']['overall'],
            mean_calls=point['metrics']['mean_steps'],
            violations=point['violations'])), flush=True)
        return point

    for policy in candidate_policies(target):
        evaluate(policy)
    feasible = sorted((p for p in points if not p['violations']),
                      key=lambda p: rank_point(p, goal))
    if not feasible:
        anchor = min(points, key=lambda p: rank_point(p, goal))
        for policy in refined_policies(anchor['policy']):
            if not any(point['policy'] == policy for point in points):
                evaluate(policy)
        feasible = sorted((p for p in points if not p['violations']),
                          key=lambda p: rank_point(p, goal))
    ranked = feasible or sorted(points, key=lambda p: rank_point(p, goal))
    dense = json.loads((REFERENCE/'configs/validation_dense.json').read_text())
    if dense['ids'] != [row['id'] for row in manifests['validation']]:
        raise ValueError('Matched dense validation IDs changed')
    validation = []
    selected = None
    for point in ranked[:4]:
        item = core._evaluate_policy(adapter, branch, cfg,
            manifests['validation'], projections, 'validation', 'T', target,
            point['policy'])
        item['violations'] = core.violations(item['metrics'], target,
                                            validation=True, goal=goal)
        if item['metrics']['accuracy'] < dense['metrics']['accuracy'] - .03:
            item['violations'].append('accuracy_drop_gt_3pp')
        validation.append(item)
        atomic(branch/'validation_trace.json', validation)
        if not point['violations'] and not item['violations']:
            selected = point
            break
    if selected is None:
        selected = ranked[0]
    status = ('attained' if selected in feasible and any(
        v['policy'] == selected['policy'] and not v['violations'] for v in validation)
              else 'unattainable_under_guardrails')
    frozen = dict(schema='temporal_sensitivity_calibration_v1',
        condition=name, beta=beta, gamma=gamma, target=target,
        config_fingerprint=cfg['fingerprint'],
        specification_fingerprint=specification['fingerprint'],
        calibration_ids=[row['id'] for row in manifests['calibration']],
        validation_ids=[row['id'] for row in manifests['validation']],
        goal=goal, tested_points=len(points), status=status,
        policy=selected['policy'], calibration=selected,
        validation=validation,
        selection='calibration-feasible min max O/G/L error, then calls/tiles; first validation pass')
    atomic(destination, frozen)
    print(json.dumps(dict(event='calibration_frozen', condition=name,
        status=status, key=selected['key'], rates=selected['metrics']['overall'],
        mean_calls=selected['metrics']['mean_steps'])), flush=True)
    return frozen


def final_one(root, adapter, base, manifests, specification,
              projections, beta, gamma, target):
    name = label(beta, gamma, target)
    branch = condition_root(root, beta, gamma, target)
    destination = branch/'configs/threshold.json'
    if not destination.exists():
        raise ValueError(f'Missing frozen calibration: {name}')
    frozen = json.loads(destination.read_text())
    cfg = config_for(base, specification, beta, gamma, target)
    if frozen['config_fingerprint'] != cfg['fingerprint']:
        raise ValueError(f'Calibration provenance mismatch: {name}')
    failures = []
    if frozen['status'] != 'attained':
        print(json.dumps(dict(event='evaluate_guardrail_miss', condition=name,
                              status=frozen['status'])), flush=True)
    for row in manifests['final']:
        try:
            core.cached(adapter, branch, 'final', 'T', target, row,
                        frozen['policy'], cfg, projections)
        except Exception as error:
            failure = dict(condition=name, id=row['id'], error=repr(error),
                           traceback=traceback.format_exc())
            failures.append(failure)
            path = Path(root)/'failures.json'
            previous = json.loads(path.read_text()) if path.exists() else []
            atomic(path, previous + [failure])
            if 'illegal memory' in str(error) or 'device-side assert' in str(error):
                raise
    return failures


def run(root, stage):
    root = Path(root)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base, manifests, specification = prepare(root)
        adapter = core._adapter(base)
        projections = Projections()
        smoke(root, adapter, base, manifests, specification, projections)
        failures = []
        for beta, gamma, target in combinations():
            name = label(beta, gamma, target)
            atomic(root/'status.json', dict(stage=stage, condition=name,
                started=time.time(), pid=os.getpid()))
            try:
                if stage == 'calibrate':
                    calibrate_one(root, adapter, base, manifests, specification,
                                  projections, beta, gamma, target)
                elif stage == 'final':
                    failures.extend(final_one(root, adapter, base, manifests,
                        specification, projections, beta, gamma, target))
                else:
                    raise ValueError(stage)
            except Exception as error:
                failure = dict(stage=stage, condition=name, error=repr(error),
                               traceback=traceback.format_exc())
                failures.append(failure)
                path = root/'failures.json'
                previous = json.loads(path.read_text()) if path.exists() else []
                atomic(path, previous + [failure])
                print(json.dumps(dict(event='condition_failed', condition=name,
                                      error=repr(error))), flush=True)
                if 'illegal memory' in str(error) or 'device-side assert' in str(error):
                    raise
        atomic(root/f'{stage}_terminal.json', dict(stage=stage,
            completed=len(combinations()) - len({x['condition'] for x in failures}),
            total=len(combinations()), errors=len(failures),
            finished=time.time()))


def launch(root, stage):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH='src:.', CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1', OMP_NUM_THREADS='4')
    command = [sys.executable, '-m',
        'experiments.value_direction_hopper.temporal_sensitivity_sweep',
        stage, '--root', str(root)]
    logfile = root/f'{stage}_{int(time.time())}.log'
    with logfile.open('xb') as stream:
        worker = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
            env=environment, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    atomic(root/f'{stage}_launch.json', dict(pid=worker.pid, log=str(logfile),
        command=command, started=time.time()))
    print(json.dumps(dict(pid=worker.pid, log=str(logfile))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'calibrate', 'final',
        'launch-calibrate', 'launch-final'))
    parser.add_argument('--root', type=Path, default=ROOT)
    arguments = parser.parse_args()
    if arguments.stage.startswith('launch-'):
        launch(arguments.root, arguments.stage.removeprefix('launch-'))
    elif arguments.stage == 'prepare':
        prepare(arguments.root)
    elif arguments.stage == 'smoke':
        base, manifests, specification = prepare(arguments.root)
        adapter = core._adapter(base)
        print(json.dumps(smoke(arguments.root, adapter, base, manifests,
                               specification, Projections()), indent=2))
    else:
        run(arguments.root, arguments.stage)
