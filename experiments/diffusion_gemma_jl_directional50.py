"""Three predeclared directional routers at50%; frozen numerical cores reused.

Preparation can run on CPU while the reference completes. No GPU work may start
until the reference50 final audit/report passes. Calibration membership is an
explicit recorded choice, never inferred from benchmark scores.
"""
from copy import deepcopy
import argparse
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
from experiments import diffusion_gemma_jl_50_only_reference as reference50
from experiments import diffusion_gemma_jl_accepted_round2 as accepted
from experiments import diffusion_gemma_jl_projection_fullbudget as continuation
from experiments.diffusion_gemma_jl_focused import protocol as parent, report as backend
from experiments.diffusion_gemma_jl_output_aware import runner, screen, shared_analysis, validation
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint

ROOT = Path('results/diffusion_gemma_jl_directional50_v6')
NEW = {
    'jl_gaussian_r32': dict(family='gaussian', rank=32),
    'jl_sign_r32': dict(family='sign', rank=32),
    'cancel_guard_gaussian_r32': dict(method='cancellation_guard', family='gaussian', rank=32),
}
ALL = {**NEW, 'full_centered': {'family': 'identity'}}
CONFIGS = {**parent.BASELINES, **ALL}
CONDITIONS = list(reference50.CONDITIONS) + [f'{name}_s50' for name in NEW]
CALIBRATION_SEED = 20260915
MODULE = 'experiments.diffusion_gemma_jl_directional50'
TEST_COUNT = 7
BASE_SMOKE = backend.smoke_audit
BASE_REPORT = backend.write_report
read, sha, frozen_write = parent.read, parent.sha, parent.frozen_write


def select_longbench_calibration(rows):
    """One of the existing50 IDs per domain, independent of scores and length."""
    pool = [r for r in rows if r['benchmark'] == 'longbench_v2']
    if len(pool) != 50 or len({r['id'] for r in pool}) != 50:
        raise ValueError('Require the preserved50 unique LongBench IDs')
    domains = sorted({r['task'] for r in pool})
    if len(domains) != 6:
        raise ValueError('Require all six LongBench domains')
    result = []
    for domain in domains:
        selected = min((r for r in pool if r['task'] == domain),
            key=lambda r: (sha(f'jl50-calibration/{CALIBRATION_SEED}/{r["id"]}'), r['id']))
        result.append(dict(deepcopy(selected), split='calibration', calibration=True))
    return sorted(result, key=lambda r: r['id'])


def prepare(root=ROOT, calibration_profile=None):
    path = root/'setup.json'
    if calibration_profile is None and path.exists():
        calibration_profile = read(path)['longbench_calibration_profile']
    if calibration_profile not in ('final50', 'historical_disjoint'):
        raise ValueError('Explicit LongBench calibration membership required before freezing the continuation')
    previous = fb.prepare(fb.ROOT)
    setup = deepcopy(previous)
    aime = [r for r in setup['calibration'] if r['benchmark'] == 'aime26']
    lb = (select_longbench_calibration(setup['final']) if calibration_profile == 'final50' else
        [r for r in setup['calibration'] if r['benchmark'] == 'longbench_v2'])
    calibration = aime + lb
    ids = {r['id'] for r in calibration}
    for row in setup['final']:
        # Only reporting membership changes; all inference fields stay verbatim.
        row['calibration'] = row['id'] in ids
    setup.update(schema='jl_directional50_v6', conditions=CONDITIONS, configs=CONFIGS,
        targets=[.5], calibration=calibration, projected_methods_paused=False,
        longbench_calibration_profile=calibration_profile,
        calibration_selection_seed=CALIBRATION_SEED,
        calibration_ids={b: [r['id'] for r in calibration if r['benchmark'] == b]
            for b in ('aime26', 'longbench_v2')},
        calibration_overlap={b: [r['id'] for r in setup['final'] if r['benchmark'] == b and r['id'] in ids]
            for b in ('aime26', 'longbench_v2')},
        calibration_policy='Six fixed IDs/benchmark at full2048/4096 budgets; unchanged local/global scalar fitting, strict48-52% measured physical count-weighted overall/global/local calibration sparsity. No inherited75% exception; no accuracy-based fitting.',
        reference_policy='Keep the completed reference50 thresholds and results unchanged; its historical LongBench calibration is disjoint from final50.',
        projection_policy='Centered Gaussian32, centered random-sign32, then Gaussian32 cancellation guard. Primary seed1729; guard independent seed2718. No score-based method, seed or matrix selection.',
        guard_policy='Existing rule: if projected kappa<0.25 and online alpha>=0.1, use max(primary centered risk, independent sketch centered risk); otherwise primary risk. Two32-dimensional sketches, not a single32-dimensional storage footprint.',
        headline_accuracy='All30 AIME26 and all50 LongBench questions, including any calibration members; never use the noncalibration subset as headline.',
        final_generation_slots=240, reused_reference_and_baseline_slots=400,
        calibration_validation_accounting='Additional inference is explicitly counted separately from240 final sample-condition slots; exact calibration/final aliases are not duplicate generation calls.',
        predecessor_gate='Complete400-output reference50 audit, independent regeneration, and successful worker exit.',
        failure_policy='Preserve failures and continue independent configurations. No75% or uncentered contribution runs. Do not publish an unverified/off-target policy as target-achieving.',
        monitoring_interval_seconds=900,
        exposure='Previously examined data; calibration overlap is explicit. No fresh held-out confirmation claim.')
    expected = previous['final']
    for original, row in zip(expected, setup['final']):
        if {k: v for k, v in original.items() if k != 'calibration'} != {k: v for k, v in row.items() if k != 'calibration'}:
            raise ValueError('Preserved evaluation fields changed')
    for benchmark in ('aime26', 'longbench_v2'):
        calibration_rows(setup, benchmark)
    frozen_write(path, setup)
    for split in ('final', 'calibration', 'development'):
        frozen_write(root/f'{split}_manifest.json', setup[split])
    frozen_write(root/'calibration_membership.json', dict(profile=calibration_profile,
        ids=setup['calibration_ids'], overlap=setup['calibration_overlap'],
        selection_rule='One per domain by SHA256(seed,ID), never score/prediction/length' if calibration_profile == 'final50' else 'Unchanged historical six',
        headlines={'aime26': 30, 'longbench_v2': 50}, final_scores_used=False))
    return setup


def calibration_rows(setup, benchmark):
    rows = [r for r in setup['calibration'] if r['benchmark'] == benchmark]
    if len(rows) != 6 or [r['id'] for r in rows] != setup['calibration_ids'][benchmark]:
        raise ValueError('Require the six frozen calibration IDs')
    budget = 2048 if benchmark == 'aime26' else 4096
    if any(r['split'] != 'calibration' or r['generation_budget'] != budget for r in rows):
        raise ValueError('Calibration must use full generation budgets')
    finals = {r['id']: r for r in setup['final']}
    if benchmark == 'longbench_v2' and setup['longbench_calibration_profile'] == 'final50':
        if rows != select_longbench_calibration(setup['final']):
            raise ValueError('LongBench calibration must reproduce score-blind domain selection')
    for row in rows:
        if row['id'] in finals:
            if any(row[k] != finals[row['id']][k] for k in ('prompt_hash', 'prompt_tokens', 'generation_budget', 'seed')):
                raise ValueError('Calibration/final inference mismatch')
    return rows


def require_reference(check_artifacts=True):
    view = reference50.control(fb.ROOT)/'report'
    folder = reference50.control(fb.ROOT)
    audit = read(view/'audit.json'); proof = read(view/'regeneration_verification.json')
    terminal = read(folder/'terminal.json'); supervisor = read(folder/'supervisor_terminal.json')
    if not (audit['complete'] and audit['completed'] == audit['expected'] == 400
            and not audit['missing'] and not audit['violations']
            and proof['passed'] and proof['completed'] == 400 and proof['inference_performed'] is False
            and proof['audit_sha256'] == sha((view/'audit.json').read_bytes())
            and terminal['complete'] and terminal['completed'] == terminal['expected'] == 400
            and supervisor['exit_code'] == 0):
        raise ValueError('Require completed, independently audited50%-only reference first')
    if check_artifacts:
        fb.check_sources({str(view/p): h for p, h in audit['artifacts'].items()})
    paths = (view/'audit.json', view/'regeneration_verification.json',
        folder/'terminal.json', folder/'supervisor_terminal.json')
    return {str(p): sha(p.read_bytes()) for p in paths}


def execution(root=ROOT):
    setup = prepare(root); previous = fb.execution(fb.ROOT)
    _, _, schedule = reference50.scheduling_contract(fb.ROOT)
    tests = root/'tests.xml'
    suites = ET.parse(tests).getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'Directional50 tests incomplete: {counts}')
    paths = (Path(__file__), Path(continuation.__file__), Path('tests/test_jl_directional50.py'), tests,
        root/'calibration_membership.json', reference50.control(fb.ROOT)/'scheduling_contract.json')
    sources = {**previous['sources'], **schedule['sources'], **{str(p): sha(p.read_bytes()) for p in paths}}
    fb.check_sources(sources)
    data = dict(schema='jl_directional50_execution_v6', parent_fingerprint=previous['fingerprint'],
        setup_sha256=sha((root/'setup.json').read_bytes()), sources=sources, runtime=previous['runtime'],
        numerical_algorithms_unchanged=True, projection_dtype='float32', matmul_allow_tf32=False,
        targets=[.5], projection_seed=1729, guard_seed=2718, max_verified_points=fb.MAX_POINTS,
        tolerance=.02, full_budget_calibration=True, final_slots=240, additional_calibration_validation=True,
        hardware_speedup_claim=False)
    data['fingerprint'] = _fingerprint(data); frozen_write(root/'execution_contract.json', data)
    return data


def audit_policy(root, p, setup, contract):
    if p['fingerprint'] != contract['fingerprint'] or p['heldout_used'] or p['target'] != .5:
        raise ValueError('Wrong identity, target or fitting scope')
    if p.get('imported_reference50_policy'):
        fb.check_sources(p['sources']); old = read(p['imported_reference50_policy'])
        accepted.audit(fb.ROOT, old, fb.prepare(fb.ROOT), fb.execution(fb.ROOT))
        if any(old[k] != p[k] for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')):
            raise ValueError('Cached reference/baseline policy changed')
        return
    if p['name'] not in NEW or p['config'] != NEW[p['name']] or p.get('user_accepted_tolerance_exception'):
        raise ValueError('Unrequested operator or tolerance exception')
    with patch.object(fb, 'CONFIG', p['config']), patch.object(fb, 'calibration_rows', calibration_rows):
        fb.audit_policy(root, p, setup, contract)


def reuse(root, setup, contract):
    sources = require_reference(); view = reference50.control(fb.ROOT)/'report'
    previous = fb.execution(fb.ROOT)
    for label in reference50.CONDITIONS:
        c = read(view/'final_configs'/f'{label}.json'); fb.check_sources(c['sources'])
        for benchmark in ('aime26', 'longbench_v2'):
            if label == 'dense':
                continue
            src = view/'policies'/benchmark/f'{label}.json'; old = read(src)
            p = {k: deepcopy(old[k]) for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')}
            p.update(fingerprint=contract['fingerprint'], heldout_used=False,
                imported_reference50_policy=str(src), sources={str(src): sha(src.read_bytes()), **sources},
                calibration_ids=old.get('calibration_ids'), calibration_budget=old.get('calibration_budget'),
                cap_one_unattainable=old.get('cap_one_unattainable'),
                rule='Completed reference50/baseline policy, unchanged. Current follow-on calibration annotations do not alter source thresholds.')
            audit_policy(root, p, setup, contract); frozen_write(root/'policies'/benchmark/f'{label}.json', p)
        for row in setup['final']:
            stage = 'dense' if label == 'dense' else 'final'
            fb.alias(root, row, stage, label, c['name'], c['config'], c['thresholds'][row['benchmark']], contract,
                shard_path(view, stage, label, row['id']), previous['fingerprint'])
        print('reused completed50 condition', label, 80, flush=True)
    frozen_write(root/'predecessor_audit.json', dict(passed=True, reused_outputs=400, sources=sources))
    states = read(fb.ROOT/'shared_state_index.json')
    fb.check_sources({s['path']: s['sha256'] for s in states})
    frozen_write(root/'shared_state_index.json', states)
    # The50-only view contains the same QKV states but only50% diagnostics.
    with patch.object(continuation, 'REFERENCE', view):
        continuation.reuse_shared_diagnostics(root, contract)


def smoke(adapter, root, setup, contract):
    # Twenty short inference calls on first execution: two native baselines,
    # twelve unpruned/pruned custom runs and six trusted pruned runs. They are
    # separate from240 final slots and each is resumably cached.
    with patch.object(runner, 'PROJECTED', NEW), patch.object(validation, 'PROJECTED', NEW):
        return validation.smoke(adapter, root, setup, contract)


def smoke_audit(root, setup, contract):
    with patch.object(backend, 'PROJECTED', NEW):
        sources = BASE_SMOKE(root, setup, contract)
    sources.update(fb.inherited_smoke(fb.ROOT, fb.prepare(fb.ROOT), fb.execution(fb.ROOT)))
    return sources


def guard_proposals(root, contract):
    folder = root/'guard_proposals'
    config = {'cancel_guard_gaussian_r32': NEW['cancel_guard_gaussian_r32']}
    with patch.object(screen, 'PROJECTED', config), patch.object(screen, 'BASELINES', {}):
        return screen.shared_screen(folder, read(root/'shared_state_index.json'), contract)


def warm_start(root, name, benchmark, target):
    warm_name = 'cancel_guard_gaussian_r16' if name == 'cancel_guard_gaussian_r32' else name
    path = parent.OLD/'policies'/benchmark/f'{warm_name}_s50.json'; old = read(path)
    fb.old_audit(parent.OLD, old, read(parent.OLD/'setup.json'), read(parent.OLD/'execution_contract.json'))
    if name != 'cancel_guard_gaussian_r32' and old['config'] != NEW[name]:
        raise ValueError('Unexpected warm-start operator')
    proposal_root = root/'guard_proposals' if name == 'cancel_guard_gaussian_r32' else parent.OLD
    values, sources = fb.distributions(proposal_root, name, benchmark)
    sources[str(path)] = sha(path.read_bytes())
    # Rank16 guard thresholds are merely an initial scalar trial; the proposal
    # CDF is recomputed with the exact requested rank32 two-sketch router.
    return old, {k: np.maximum(v, -1e30) for k, v in values.items()}, sources


def calibrate(adapter, root, setup, contract, name, benchmark):
    with patch.object(continuation, 'NEW', NEW), \
         patch.object(continuation, 'warm_start', lambda n, b, t: warm_start(root, n, b, t)), \
         patch.object(continuation, 'audit_policy', audit_policy), \
         patch.object(fb, 'calibration_rows', calibration_rows):
        return continuation.calibrate(adapter, root, setup, contract, name, benchmark, .5)


def freeze_conditions(root, setup, contract):
    proof = read(root/'predecessor_audit.json')
    common = {**contract['sources'], **proof['sources'], **smoke_audit(root, setup, contract),
        **read(root/'inherited_shared_diagnostics.json')['sources']}
    for path in (root/'predecessor_audit.json', root/'inherited_shared_diagnostics.json'):
        common[str(path)] = sha(path.read_bytes())
    fb.check_sources(common)
    result = {}
    for label in CONDITIONS:
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        cfg = {} if name == 'dense' else CONFIGS[name]
        thresholds, sources, missing = {}, dict(common), []
        for benchmark in ('aime26', 'longbench_v2'):
            if name == 'dense':
                thresholds[benchmark] = None; continue
            path = root/'policies'/benchmark/f'{label}.json'
            if not path.exists():
                missing.append(benchmark); continue
            p = read(path); audit_policy(root, p, setup, contract)
            if p['name'] != name or p['config'] != cfg:
                raise ValueError('Final condition differs from verified policy')
            thresholds[benchmark] = p['policy']; sources[str(path)] = sha(path.read_bytes())
        c = dict(fingerprint=contract['fingerprint'], name=name, target=0. if name == 'dense' else .5,
            config=cfg, thresholds=thresholds, sources=sources, unavailable_benchmarks=missing,
            expected_per_benchmark={'aime26': 30, 'longbench_v2': 50})
        if not missing:
            frozen_write(root/'final_configs'/f'{label}.json', c)
        else:
            _write(root/'pending_configs'/f'{label}.json', c)
        result[label] = c
    return result


def accounting(root):
    """Count actual completed calls separately from aliases/result slots."""
    counts = {}
    for stage in ('calibration', 'final', 'dense'):
        rows = [read(p) for p in (root/stage).glob('**/shards/*.json')]
        counts[stage] = dict(stored_results=len(rows), completed_new_inference=sum(not o.get('imported_source') for o in rows),
            imported_or_exact_reused=sum(bool(o.get('imported_source')) for o in rows))
    validation_root = root/'validation'
    smoke_calls = sum(1 for p in validation_root.glob('**/shards/*.json') if not read(p).get('imported_source'))
    native_calls = len(list(validation_root.glob('*/*.native.json')))
    result = dict(final_generation_slots=240, reused_reference_baseline_slots=400,
        completed_by_stage=counts, completed_validation_inference=smoke_calls+native_calls,
        maximum_initial_smoke_calls=20,
        completed_new_inference=sum(v['completed_new_inference'] for v in counts.values())+smoke_calls+native_calls,
        note='Completed calls, not failed partial attempts. Calibration aliases may also fill final slots without inference; shared-QKV proposal/diagnostic replays are additional GPU work but not generation calls.')
    _write(root/'inference_accounting.json', result)
    return result


def write_report(root, setup, rows, tasks, compared, diag, audit, policies):
    audit.update(expected=640, complete=audit['completed'] == 640 and not audit['missing'] and not audit['violations'])
    BASE_REPORT(root, setup, rows, tasks, compared, diag, audit, policies)
    path = root/'report.md'; text = path.read_text()
    text = text.replace('# Focused JL output-aware routing:', '# 50%-only directional routing:')
    text = text.replace('/1040 audited final outputs. Three new routers, two targets;',
        '/640 audited final outputs. Three new routers at50% only;')
    text = text.replace('All selected LongBench final IDs/hashes are disjoint from both studies’ calibration/development sets.',
        f"Follow-on LongBench calibration profile: {setup['longbench_calibration_profile']}; current overlap is {len(setup['calibration_overlap']['longbench_v2'])}/50. Reference/baseline calibration remains unchanged. Headline scores include all30 AIME and all50 LongBench problems, including calibration members.")
    start = text.index('Gaussian centered:'); end = text.index('The unchanged BLASST implementation', start)
    text = text[:start] + (
        'Three prespecified methods, in order: centered Gaussian32, centered random-sign32, and the existing Gaussian32 cancellation guard. Primary seed1729; independent guard seed2718. The guard uses max(primary,second) centered risk only where projected kappa<0.25 and online alpha>=0.1. It stores two32-dimensional sketches (64 coordinates total). Full-dimensional reference results are reused. All retain strict worst-query128x64 tile gating, ties/first support, native masks/GQA, prefix+canvas eligibility, FP32 routing and ordinary original-V renormalization. No algorithm is changed or chosen by final accuracy.\n\n'
    ) + text[end:]
    start = text.index('Cached AIME baseline policies are unchanged.'); end = text.index('## Main results', start)
    text = text[:start] + (
        'All400 dense/BLASST/mass/reference50 outputs are reused from the independently audited reference bundle. The three follow-on methods are calibrated on six fixed questions/benchmark at full2048/4096-token budgets. Existing empirical-rank proposals and scalar fallback allow at most12 joint points. Deployment requires measured physical overall/global/local calibration sparsity between48% and52%, with no inherited75% exception. Historical thresholds are warm starts only. Gaussian/sign32 reuse historical calibration-state CDFs; the guard32 CDF is recomputed with the exact requested operator on those saved states, while a historical rank16 guard scalar is only the first verified trial. No final accuracy is used for threshold selection.\n\n'
        'Historical540 early512-token dense calibration states remain the common diagnostic/proposal population; they are not substitutes for full-budget verification and do not exhaustively cover later generation. Calibration membership differs from historical baselines/reference where documented; headline denominators never drop calibration questions. Achieved final sparsity can still transfer imperfectly and is reported without retuning from final scores. Cached mass-only is the existing max-based candidate mass bound, not mass_exact, so its comparison does not isolate direction from mass estimation alone.\n\n'
    ) + text[end:]
    text = text.replace('No Gaussian/sign or rank8/16 comparison and no end-to-end seed-sensitivity conclusion is claimed in this user-narrowed study.',
        'Gaussian versus random-sign is tested at rank32 only. No rank8/16 comparison or end-to-end seed-sensitivity conclusion is claimed.')
    text = text.replace('prior JL32 calibration-only shared-state checks', 'prior Gaussian/sign32 calibration-only shared-state checks')
    text = text.replace('at unchanged thresholds.', 'at the historical512-token thresholds, not the new final thresholds.')
    text = text.replace('experiments.diffusion_gemma_jl_focused.workflow', MODULE)
    text += '\n## Inference accounting\n\nThe240 slots are three methods times80 final questions, in addition to400 cached reference/baseline slots. Calibration trials and20 initial short validation calls are separately accounted in inference_accounting.json. Exact selected calibration outputs also serve as final outputs without repeated inference. Failed attempts remain in failures.jsonl. All75% and uncentered-contribution follow-ons are superseded, not scientific exclusions.\n'
    path.write_text(text)


def regenerate(root=ROOT):
    with patch.object(backend, 'prepare', prepare), patch.object(backend, 'execution', execution), \
         patch.object(backend, 'PROJECTED', ALL), patch.object(backend, 'TARGETS', (.5,)), \
         patch.object(backend.common, 'TARGETS', (.5,)), \
         patch.object(backend, 'audit_policy', audit_policy), patch.object(backend, 'smoke_audit', smoke_audit), \
         patch.object(backend, 'write_report', write_report):
        audit = backend.regenerate(root)
    accounting(root)
    audit['artifacts']['inference_accounting.json'] = sha((root/'inference_accounting.json').read_bytes())
    _write(root/'audit.json', audit)
    return audit


def verify(root=ROOT):
    before = read(root/'audit.json')
    if not before['complete'] or before['completed'] != 640:
        raise ValueError('Require all640 audited50%-only outputs')
    fb.check_sources({str(root/p): h for p, h in before['artifacts'].items()})
    if regenerate(root) != before:
        raise ValueError('Raw-only report regeneration changed')
    proof = dict(passed=True, completed=640, inference_performed=False,
        audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json', proof)
    return proof


def work(root):
    require_reference(); setup, contract = prepare(root), execution(root)
    fb.phase(root, 'reuse_completed_reference50_and_baselines', expected=400); reuse(root, setup, contract)
    torch.backends.cuda.matmul.allow_tf32 = False
    fb.phase(root, 'load_model')
    adapter = create_adapter('diffusion_gemma', parent.MODEL, device='cuda', precision='bfloat16', revision=parent.REVISION).load()
    with patch.object(runner, 'PROJECTED', NEW):
        fb.phase(root, 'three_method_two_example_smoke', maximum_generation_calls=20)
        smoke(adapter, root, setup, contract); accounting(root)
        fb.phase(root, 'exact_guard32_shared_proposals', generation_calls=0)
        guard_proposals(root, contract)
        for name in NEW:
            for benchmark in ('aime26', 'longbench_v2'):
                try:
                    calibrate(adapter, root, setup, contract, name, benchmark)
                except Exception:
                    fb.failure(root, 'calibration', method=name, benchmark=benchmark, target=.5)
                accounting(root)
        conditions = freeze_conditions(root, setup, contract)
        fb.phase(root, 'shared_operator_diagnostics', generation_calls=0)
        try:
            with patch.object(shared_analysis, 'PROJECTED', ALL), patch.object(shared_analysis, 'BASELINES', parent.BASELINES), \
                 patch.object(shared_analysis, 'TARGETS', (.5,)):
                shared_analysis.analyze(root, read(root/'shared_state_index.json'), contract)
        except Exception:
            fb.failure(root, 'shared_operator_diagnostics')
        for name, cfg in NEW.items():
            label = name+'_s50'; c = conditions[label]
            fb.phase(root, 'final', condition=label, expected=80)
            for row in sorted(setup['final'], key=lambda r: (len(r['prompt_tokens']), r['id'])):
                if row['benchmark'] not in c['thresholds']:
                    continue
                try:
                    threshold = c['thresholds'][row['benchmark']]
                    if row['id'] in setup['calibration_ids'][row['benchmark']]:
                        p = read(root/'policies'/row['benchmark']/f'{label}.json')
                        point = next(x for x in p['trace'] if x['iteration'] == p['selected_round'])
                        fb.alias(root, row, 'final', label, name, cfg, threshold, contract,
                            Path(point['sources'][row['id']]['path']), contract['fingerprint'])
                    else:
                        runner.cached(adapter, root, row, 'final', label, name, cfg, threshold, contract)
                except Exception:
                    fb.failure(root, 'final', condition=label, id=row['id'])
            accounting(root)
    del adapter; torch.cuda.empty_cache()
    fb.phase(root, 'report'); audit = regenerate(root)
    if audit['complete']:
        fb.phase(root, 'independent_report_verification'); verify(root)
    _write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'], expected=640, finished=time.time()))
    fb.phase(root, 'finished', complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_reference(); fb.gpu_idle()
        with (root/'run.log').open('a', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'work', '--output', str(root)], stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=child.pid, supervisor_pid=os.getpid(), started=time.time()))
            while True:
                try:
                    code = child.wait(timeout=900); break
                except subprocess.TimeoutExpired:
                    state = {k: read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase', 'progress')}
                    _append(root/'monitor.jsonl', dict(time=time.time(), pid=child.pid, **state))
            _write(root/'supervisor_terminal.json', dict(exit_code=code, pid=child.pid, finished=time.time()))
            if code:
                raise SystemExit(code)


def launch(root):
    require_reference(); execution(root); fb.gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise', '--output', str(root)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, final_slots=240, reused_slots=400,
        calibration_validation_additional=True, targets=[.5])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'launch', 'supervise', 'work', 'report', 'verify'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--lb-calibration', choices=('final50', 'historical_disjoint'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.lb_calibration)
    elif args.command in ('report', 'verify'):
        print(json.dumps((regenerate if args.command == 'report' else verify)(args.output)))
    else:
        globals()[args.command](args.output)
