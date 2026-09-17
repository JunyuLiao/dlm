"""Gaussian rank32 follow-on, gated by the completed reference audit.

Only scheduling, full-budget calibration and provenance change. Frozen numerical
routers/kernels and all earlier result bundles remain untouched.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from dllm.models import create_adapter

from experiments import diffusion_gemma_jl_fullbudget as fb
from experiments import diffusion_gemma_jl_accepted_round2 as accepted
from experiments.diffusion_gemma_jl_focused import protocol as parent, report as backend
from experiments.diffusion_gemma_jl_focused.reuse import dispatch, audit_policy as prior_audit
from experiments.diffusion_gemma_jl_output_aware import runner, shared_analysis
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint

ROOT = Path('results/diffusion_gemma_jl_gaussian32_fullbudget_v4')
REFERENCE = fb.ROOT
MODULE = 'experiments.diffusion_gemma_jl_projection_fullbudget'
NEW = {n: c for n, c in parent.PROJECTED.items() if n != 'full_centered'}
ALL = parent.PROJECTED
read, sha, frozen_write = parent.read, parent.sha, parent.frozen_write
TEST_COUNT = 7


def require_reference(root=REFERENCE, check_artifacts=True):
    """A completed generation phase alone is insufficient to start the next GPU job."""
    audit = read(root/'audit.json')
    proof = read(root/'regeneration_verification.json')
    terminal = read(root/'terminal.json')
    supervisor = read(root/'accepted_round2/supervisor_terminal.json')
    if not (audit['complete'] and audit['completed'] == audit['expected'] == 720
            and not audit['missing'] and not audit['violations']
            and proof['passed'] and proof['completed'] == 720
            and proof['inference_performed'] is False
            and proof['audit_sha256'] == sha((root/'audit.json').read_bytes())
            and terminal['complete'] and terminal['completed'] == 720
            and supervisor['exit_code'] == 0):
        raise ValueError('Full-dimensional reference and independent audit must finish first')
    if check_artifacts:
        check_sources({str(root/p): h for p, h in audit['artifacts'].items()})
    return {str(root/p): sha((root/p).read_bytes()) for p in
        ('audit.json', 'regeneration_verification.json', 'terminal.json',
         'accepted_round2/supervisor_terminal.json')}


def prepare(root=ROOT):
    setup = deepcopy(fb.prepare(REFERENCE))
    setup.update(schema='jl_gaussian32_fullbudget_v4', conditions=parent.CONDITIONS,
        configs=parent.CONFIGS, projected_methods_paused=False,
        calibration_policy='Both Gaussian rank32 methods: same six calibration IDs per benchmark, full2048/4096-token budgets; existing local/global empirical-rank proposals and scalar fallback, max12 joint verification points; require overall/local/global physical count-weighted sparsity within2pp. No final-score fitting.',
        baseline_source=str(REFERENCE),
        reference_policy='Reuse completed full-budget reference, including the explicitly accepted LongBench75 round2 exception; do not refit or rerun.',
        failure_policy='Preserve failed attempts and continue independent methods/targets/benchmarks; no accuracy-based early exclusion. Off-target projected policies are documented failures, never silently deployed.',
        predecessor_gate='Complete720-output reference audit AND independent raw-only regeneration AND successful worker exit required before launch.',
        monitoring_interval_seconds=900,
        authorization='After the full-dimensional oracle test finishes, begin with the projection methods:32-dimensional Gaussian, targets50% and75%.')
    for b in ('aime26', 'longbench_v2'):
        fb.calibration_rows(setup, b)
    frozen_write(root/'setup.json', setup)
    for split in ('calibration', 'development', 'final'):
        frozen_write(root/f'{split}_manifest.json', setup[split])
    frozen_write(root/'selection.json', read(parent.ROOT/'selection.json'))
    dataset_audit = read(parent.ROOT/'dataset_audit.json')
    dataset_audit.update(cached_baselines=560, cached_full_reference=160,
        reused_outputs=720, new_generations=320)
    frozen_write(root/'dataset_audit.json', dataset_audit)
    return setup


def execution(root=ROOT):
    setup = prepare(root)
    previous = fb.execution(REFERENCE)
    extra = read(REFERENCE/'accepted_round2/extension_contract.json')
    test_path = root/'tests.xml'
    suites = ET.parse(test_path).getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'Projection continuation tests incomplete: {counts}')
    paths = (Path(__file__), Path('tests/test_jl_projection_fullbudget.py'), test_path,
        REFERENCE/'execution_contract.json', REFERENCE/'accepted_round2/extension_contract.json')
    sources = {**previous['sources'], **extra['sources'],
        **{str(p): sha(p.read_bytes()) for p in paths}}
    check_sources(sources)
    data = dict(schema='jl_gaussian32_fullbudget_execution_v4',
        parent_fingerprint=previous['fingerprint'], setup_sha256=sha((root/'setup.json').read_bytes()),
        sources=sources, runtime=previous['runtime'], numerical_algorithms_unchanged=True,
        projection_seed=setup['projection_seed'], projection_family='gaussian', projection_rank=32,
        calibration_budgets={'aime26': 2048, 'longbench_v2': 4096},
        tolerance=fb.TOLERANCE, max_verified_points=fb.MAX_POINTS, hardware_speedup_claim=False)
    data['fingerprint'] = _fingerprint(data)
    frozen_write(root/'execution_contract.json', data)
    return data


def audit_policy(root, p, setup, contract):
    if p['fingerprint'] != contract['fingerprint'] or p['heldout_used']:
        raise ValueError('Policy identity or final-data fitting violation')
    if p.get('imported_reference_bundle'):
        check_sources(p['sources'])
        old = read(p['imported_reference_bundle'])
        accepted.audit(REFERENCE, old, fb.prepare(REFERENCE), fb.execution(REFERENCE))
        if any(p[k] != old[k] for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')):
            raise ValueError('Imported reference/baseline policy changed')
        return
    if p['name'] not in NEW or p['config'] != NEW[p['name']]:
        raise ValueError('Unexpected projected configuration')
    # The existing full-budget auditor checks all raw IDs/budgets/counts and
    # selected joint thresholds. CONFIG is process-local, never a source edit.
    with patch.object(fb, 'CONFIG', NEW[p['name']]):
        fb.audit_policy(root, p, setup, contract)


def reuse(root, setup, contract):
    sources = require_reference()
    previous = fb.execution(REFERENCE)
    for label in fb.CONDITIONS:
        c = read(REFERENCE/'final_configs'/f'{label}.json')
        check_sources(c['sources'])
        for benchmark in ('aime26', 'longbench_v2'):
            if label == 'dense':
                continue
            src = REFERENCE/'policies'/benchmark/f'{label}.json'
            old = read(src)
            p = {k: deepcopy(old[k]) for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')}
            p.update(fingerprint=contract['fingerprint'], heldout_used=False,
                imported_reference_bundle=str(src),
                sources={str(src): sha(src.read_bytes()), **sources},
                calibration_budget=old.get('calibration_budget'),
                cap_one_unattainable=old.get('cap_one_unattainable'),
                user_accepted_tolerance_exception=old.get('user_accepted_tolerance_exception', False),
                rule='Exact completed reference/baseline policy; no calibration or inference repeated.')
            audit_policy(root, p, setup, contract)
            frozen_write(root/'policies'/benchmark/f'{label}.json', p)
        for row in setup['final']:
            stage = 'dense' if label == 'dense' else 'final'
            fb.alias(root, row, stage, label, c['name'], c['config'],
                c['thresholds'][row['benchmark']], contract,
                shard_path(REFERENCE, stage, label, row['id']), previous['fingerprint'])
        print('reused completed condition', label, 80, flush=True)
    frozen_write(root/'predecessor_audit.json', dict(passed=True, reused_outputs=720, sources=sources))
    validation = backend.smoke_audit(parent.ROOT, parent.prepare(parent.ROOT), parent.execution(parent.ROOT))
    frozen_write(root/'inherited_validation.json', dict(passed=True, sources=validation,
        note='Unchanged numerical kernels and dispatch: raw-audited two-example, three-router native/unpruned and trusted/pruned exact parity; calibration changes only.'))
    states = read(REFERENCE/'shared_state_index.json')
    check_sources({r['path']: r['sha256'] for r in states})
    frozen_write(root/'shared_state_index.json', states)
    reuse_shared_diagnostics(root, contract)


def reuse_shared_diagnostics(root, contract):
    # Same QKV, operator and policy: preserve already audited reference/baseline
    # diagnostics rather than repeating their full-dimensional PV replays.
    imported = {}
    for item in read(REFERENCE/'shared_diagnostics_index.json'):
        src = Path(item['path']); check_sources({str(src): item['sha256']})
        data = read(src); ident = data['identity']
        check_sources({ident['policy_path']: ident['policy_sha256'], **ident['diagnostic_sources']})
        old_policy = read(ident['policy_path'])
        dest_policy = root/'policies'/old_policy['benchmark']/Path(ident['policy_path']).name
        new_policy = read(dest_policy)
        if (old_policy['config'], old_policy['policy']) != (new_policy['config'], new_policy['policy']):
            raise ValueError('Shared diagnostic reuse changed operator/thresholds')
        ident.update(fingerprint=contract['fingerprint'], policy_path=str(dest_policy),
            policy_sha256=sha(dest_policy.read_bytes()))
        dest = root/'shared_diagnostics'/src.name
        frozen_write(dest, data)
        imported[str(src)] = item['sha256']
    frozen_write(root/'inherited_shared_diagnostics.json', dict(sources=imported,
        note='Exact shared-QKV/reference/baseline diagnostics; only output namespace and policy provenance rebased.'))


def warm_start(name, benchmark, target):
    source_root = parent.ROOT if name == 'contribution_gaussian_r32' else parent.OLD
    path = source_root/'policies'/benchmark/f'{name}_s{int(target*100)}.json'
    old = read(path)
    if source_root == parent.ROOT:
        prior_audit(source_root, old, parent.prepare(source_root), parent.execution(source_root))
    else:
        fb.old_audit(source_root, old, read(source_root/'setup.json'), read(source_root/'execution_contract.json'))
    if old['config'] != NEW[name]:
        raise ValueError('Warm-start operator differs from requested Gaussian rank32')
    values, sources = fb.distributions(source_root, name, benchmark)
    sources[str(path)] = sha(path.read_bytes())
    return old, {k: np.maximum(v, -1e30) for k, v in values.items()}, sources


def calibrate(adapter, root, setup, contract, name, benchmark, target):
    label = f'{name}_s{int(target*100)}'
    cfg = NEW[name]
    dest = root/'policies'/benchmark/f'{label}.json'
    if dest.exists():
        p = read(dest); audit_policy(root, p, setup, contract); return p
    rows = fb.calibration_rows(setup, benchmark)
    old, values, dist_sources = warm_start(name, benchmark, target)
    trace_path = root/'calibration_traces'/benchmark/f'{label}.json'
    trace = read(trace_path) if trace_path.exists() else []
    previous = parent.execution(parent.ROOT)
    for iteration in range(len(trace), fb.MAX_POINTS):
        if trace:
            best, _ = fb.select_point(trace, target)
            if fb.within(best['achieved'], target):
                break
        policy = deepcopy(old['policy']) if not trace else fb.proposal(trace, target, values)
        if any(policy == p['policy'] for p in trace):
            break
        tag = f'{label}/{benchmark}/{_fingerprint(policy)[:16]}'
        fb.phase(root, 'full_budget_calibration', method=name, benchmark=benchmark,
            target=target, point=iteration+1, generation_budget=rows[0]['generation_budget'],
            previous_achieved=trace[-1]['achieved'] if trace else None)
        outputs, sources = [], {}
        for row in rows:
            path = shard_path(root, 'calibration', tag, row['id'])
            source = shard_path(parent.ROOT, 'final', label, row['id'])
            if not path.exists() and source.exists():
                saved = runner.load_output(source)
                try:
                    check_result(saved, row, previous['fingerprint'], cfg, policy)
                except ValueError:
                    pass
                else:
                    fb.alias(root, row, 'calibration', tag, name, cfg, policy, contract, source, previous['fingerprint'])
            out = runner.cached(adapter, root, row, 'calibration', tag, name, cfg, policy, contract)
            outputs.append(out)
            sources[row['id']] = dict(path=str(path), sha256=sha(path.read_bytes()))
        metrics, achieved = fb.measured(outputs)
        trace.append(dict(iteration=iteration, policy=policy, achieved=achieved, metrics=metrics,
            sources=sources, source_ids=[r['id'] for r in rows], source_condition=tag))
        _write(trace_path, trace)
        print('full-budget verified', name, benchmark, target, iteration+1, achieved, flush=True)
    best, _ = fb.select_point(trace, target)
    if not fb.within(best['achieved'], target):
        _write(root/'calibration_failures'/benchmark/f'{label}.json', dict(target=target, trace=trace,
            best_achieved=best['achieved'], deploy=False, reason='No joint full-budget point within2pp.'))
        raise RuntimeError(f'Full-budget calibration failed: {benchmark}/{label} {best["achieved"]}')
    p = dict(fingerprint=contract['fingerprint'], benchmark=benchmark, name=name, config=cfg, target=target,
        policy=best['policy'], measured=best['achieved'], trace=trace, selected_round=best['iteration'],
        calibration_ids=[r['id'] for r in rows], calibration_budget=rows[0]['generation_budget'],
        heldout_used=False, distribution_sources=dist_sources, tolerance=fb.TOLERANCE,
        rule='Existing rank proposals/scalar fallback; full-budget joint sparse-trajectory verification, max12 points. Only achieved physical sparsity selects thresholds; never accuracy.')
    audit_policy(root, p, setup, contract)
    frozen_write(dest, p)
    return p


def freeze_conditions(root, setup, contract):
    proof = read(root/'predecessor_audit.json')
    check_sources(proof['sources'])
    sources = {**contract['sources'], **proof['sources'],
        **fb.inherited_smoke(root, setup, contract),
        **read(root/'inherited_shared_diagnostics.json')['sources'],
        str(root/'inherited_shared_diagnostics.json'): sha((root/'inherited_shared_diagnostics.json').read_bytes()),
        str(root/'predecessor_audit.json'): sha((root/'predecessor_audit.json').read_bytes())}
    result = {}
    for label in setup['conditions']:
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        target = 0. if name == 'dense' else int(label.rsplit('_s', 1)[1])/100
        cfg = {} if name == 'dense' else parent.CONFIGS[name]
        thresholds, own_sources, missing = {}, dict(sources), []
        for benchmark in ('aime26', 'longbench_v2'):
            if name == 'dense':
                thresholds[benchmark] = None; continue
            path = root/'policies'/benchmark/f'{label}.json'
            if not path.exists():
                missing.append(benchmark); continue
            p = read(path); audit_policy(root, p, setup, contract)
            if (p['name'], p['config'], p['target']) != (name, cfg, target):
                raise ValueError('Final condition does not match policy')
            thresholds[benchmark] = p['policy']; own_sources[str(path)] = sha(path.read_bytes())
        c = dict(fingerprint=contract['fingerprint'], name=name, target=target, config=cfg,
            thresholds=thresholds, sources=own_sources, unavailable_benchmarks=missing,
            expected_per_benchmark={'aime26': 30, 'longbench_v2': 50})
        if not missing:
            frozen_write(root/'final_configs'/f'{label}.json', c)
        else:
            _write(root/'pending_configs'/f'{label}.json', c)
        result[label] = c
    return result


BASE_REPORT = backend.write_report


def revised_report(root, setup, rows, tasks, compared, diag, audit, policies):
    BASE_REPORT(root, setup, rows, tasks, compared, diag, audit, policies)
    path = root/'report.md'; text = path.read_text()
    start = text.index('Cached AIME baseline policies are unchanged.')
    end = text.index('## Main results', start)
    text = text[:start] + (
        'Baseline policies and all720 dense/BLASST/mass/full-reference outputs are reused from the completed, independently audited reference bundle. Both Gaussian rank32 methods are newly verified on the same6 calibration IDs/benchmark at the full2048 AIME/4096 LongBench output budgets. Existing empirical-rank proposals and scalar fallback allow at most12 jointly verified points; projected policies require overall/local/global physical sparsity within2pp. The reference LongBench75 round2 exception remains explicitly user-accepted:73.28% overall,72.56% global,76.64% local on calibration, not a strict2pp pass. No tolerance exception is inherited by projected methods. LongBench projected/reference calibration covers3 domains; historical baseline calibration covers6 domains/12 questions. Early512-token shared QKV snapshots supply proposals/diagnostics only, not full-budget verification. Actual final sparsity is never retuned.\n\n'
        'The cached mass baseline uses its existing max-based candidate mass bound; it is not the exact online mass-only control. Differences against it do not isolate directional information from mass estimation. Prior unsuccessful/superseded outputs remain separate. The same seed1729 is frozen for both projected methods, with no score-based matrix selection.\n\n'
    ) + text[end:]
    text = text.replace('experiments.diffusion_gemma_jl_focused.workflow', MODULE)
    text = text.replace('# Focused JL output-aware routing:', '# Full-budget Gaussian rank32 routing:')
    text = text.replace('at unchanged thresholds. They compare',
        'at the historical512-token calibration thresholds, not the new final thresholds. They compare')
    path.write_text(text)


def regenerate(root=ROOT):
    with patch.object(backend, 'prepare', prepare), patch.object(backend, 'execution', execution), \
         patch.object(backend, 'PROJECTED', ALL), patch.object(backend, 'audit_policy', audit_policy), \
         patch.object(backend, 'smoke_audit', fb.inherited_smoke), \
         patch.object(backend, 'write_report', revised_report):
        return backend.regenerate(root)


def verify(root=ROOT):
    before = read(root/'audit.json')
    if not before['complete'] or before['completed'] != 1040:
        raise ValueError('Require all1040 audited outputs')
    check_sources({str(root/p): h for p, h in before['artifacts'].items()})
    after = regenerate(root)
    if before != after:
        raise ValueError('Raw-only report regeneration changed')
    proof = dict(passed=True, completed=1040, inference_performed=False,
        audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json', proof)
    return proof


def work(root):
    require_reference()
    setup, contract = prepare(root), execution(root)
    fb.phase(root, 'reuse_completed_reference_and_baselines', expected=720)
    reuse(root, setup, contract)
    torch.backends.cuda.matmul.allow_tf32 = False
    fb.phase(root, 'load_model')
    adapter = create_adapter('diffusion_gemma', parent.MODEL, device='cuda',
        precision='bfloat16', revision=parent.REVISION).load()
    with dispatch():
        for target in parent.TARGETS:
            for name in NEW:
                for benchmark in ('aime26', 'longbench_v2'):
                    try:
                        calibrate(adapter, root, setup, contract, name, benchmark, target)
                    except Exception:
                        fb.failure(root, 'calibration', method=name, benchmark=benchmark, target=target)
        conditions = freeze_conditions(root, setup, contract)
        fb.phase(root, 'shared_operator_diagnostics')
        try:
            with patch.object(shared_analysis, 'PROJECTED', ALL), patch.object(shared_analysis, 'BASELINES', parent.BASELINES):
                shared_analysis.analyze(root, read(root/'shared_state_index.json'), contract)
        except Exception:
            fb.failure(root, 'shared_operator_diagnostics')
        for target in parent.TARGETS:
            for name, cfg in NEW.items():
                label = f'{name}_s{int(target*100)}'; c = conditions[label]
                fb.phase(root, 'final', condition=label, expected=80)
                for row in sorted(setup['final'], key=lambda r: (len(r['prompt_tokens']), r['id'])):
                    if row['benchmark'] not in c['thresholds']:
                        continue
                    try:
                        threshold = c['thresholds'][row['benchmark']]
                        if row['benchmark'] == 'aime26' and row['calibration']:
                            p = read(root/'policies/aime26'/f'{label}.json')
                            point = next(x for x in p['trace'] if x['iteration'] == p['selected_round'])
                            fb.alias(root, row, 'final', label, name, cfg, threshold, contract,
                                Path(point['sources'][row['id']]['path']), contract['fingerprint'])
                        else:
                            runner.cached(adapter, root, row, 'final', label, name, cfg, threshold, contract)
                    except Exception:
                        fb.failure(root, 'final', condition=label, id=row['id'])
    del adapter; torch.cuda.empty_cache()
    fb.phase(root, 'report'); audit = regenerate(root)
    if audit['complete']:
        fb.phase(root, 'independent_report_verification'); verify(root)
    _write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'],
        expected=1040, finished=time.time()))
    fb.phase(root, 'finished', complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_reference(); gpu_idle()
        with (root/'run.log').open('a', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'work', '--output', str(root)],
                stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=child.pid, supervisor_pid=os.getpid(), started=time.time()))
            while True:
                try:
                    code = child.wait(timeout=900); break
                except subprocess.TimeoutExpired:
                    state = {k: read(root/f'{k}.json') if (root/f'{k}.json').exists() else None
                        for k in ('phase', 'progress')}
                    _append(root/'monitor.jsonl', dict(time=time.time(), pid=child.pid, **state))
            _write(root/'supervisor_terminal.json', dict(exit_code=code, pid=child.pid, finished=time.time()))
            if code:
                raise SystemExit(code)


def launch(root):
    require_reference(); execution(root); gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise', '--output', str(root)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, output=str(root), new_final_outputs=320,
        reused_outputs=720, health_interval_seconds=900)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'launch', 'supervise', 'work', 'report', 'verify'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command in ('report', 'verify'):
        print(json.dumps((regenerate if args.command == 'report' else verify)(args.output)))
    else:
        globals()[args.command](args.output)
