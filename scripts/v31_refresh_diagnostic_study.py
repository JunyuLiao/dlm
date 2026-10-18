"""Frozen, budget-bounded native refresh diagnostic; never resumes older studies.

Preparation is CPU-only except for the read-only runtime/GPU identity query.
Every child owns a new process group and immutable output leaf. All raw inputs
and completions stay below private/. Completed native requests are retained even
when a deadline prevents complete planned coverage; no budget changes the sampler.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official
from scripts.v31_vllm_paired_bench import request_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / 'results/v31_refresh_diagnostic_20261018/attempt001'
LB_PARENT = ROOT / 'results/v31_value_e2e_longbench503_20261017/native_attempt001'
RULER_PARENT = ROOT / 'results/v31_v2_ruler128k_seed42_tau021_20261015'
SPEC = 'docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md'
RECOVERY_SPEC = 'docs/REFRESH_DIAGNOSTIC_RECOVERY_20261018.md'
FACTORIAL = ('dense_fresh', 'dense_hold', 'sparse_fresh', 'sparse_hold')
BUDGET = dict(qualification=900., headroom=1500., factorial=3900., reserve=900.)
RECOVERY_BUDGET = dict(qualification=900., headroom=1800., factorial=3900., reserve=600.)
REFERENCE_ENGINES = (1, 2, 3, 4, 5)
TOTAL_SECONDS = 7200.
MIN_START_SECONDS = 180.
SALT = 'v31-refresh-diagnostic-20261018-metadata-v1'
REQUIRED = [SPEC, RECOVERY_SPEC, 'scripts/v31_refresh_diagnostic_study.py',
    'scripts/v31_refresh_diagnostic_worker.py', 'scripts/v31_refresh_headroom_profile.py',
    'scripts/v31_refresh_diagnostic_report.py', 'experiments/numerical_qk_reuse/v31_refresh_diagnostic.py',
    'tests/test_v31_refresh_diagnostic.py', 'tests/test_v31_refresh_diagnostic_worker.py',
    'tests/test_v31_refresh_diagnostic_study.py',
    'tests/test_v31_refresh_headroom_profile.py', 'tests/test_v31_refresh_diagnostic_report.py']
RECOVERY_SOURCE_ALLOWLIST = frozenset((RECOVERY_SPEC,
    'scripts/v31_refresh_diagnostic_study.py', 'scripts/v31_refresh_diagnostic_report.py',
    'scripts/v31_refresh_headroom_profile.py', 'tests/test_v31_refresh_diagnostic_study.py',
    'tests/test_v31_refresh_diagnostic_report.py', 'tests/test_v31_refresh_headroom_profile.py'))


def digest_metadata(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def token_length(cell):
    value = cell['prompt_tokens']
    return len(value) if isinstance(value, list) else int(value)


def metadata_key(cell):
    # Never consult previous generation, score, latency, or denoising outcomes.
    return digest_metadata([SALT, cell['dataset'], int(cell['index'])])


def choose_cohorts(lb, ruler, ruler_meta):
    bins = [(0, 32768), (32768, 65536), (65536, 98304), (98304, math.inf)]
    chosen = []
    for lo, hi in bins:
        candidates = sorted((c for c in lb if lo <= token_length(c) < hi), key=metadata_key)
        if len(candidates) < 2:
            raise ValueError('insufficient LongBench metadata coverage')
        chosen.extend(candidates[:2])
    by_task = defaultdict(list)
    for cell in ruler:
        by_task[ruler_meta[cell['index']]['task_base']].append(cell)
    tasks = sorted(by_task, key=lambda t: digest_metadata([SALT, 'ruler_task', t]))[:4]
    if len(tasks) != 4:
        raise ValueError('four distinct RULER task families required')
    selected_ruler = [min(by_task[t], key=metadata_key) for t in tasks]
    ordered = sorted(chosen, key=lambda c: (token_length(c), metadata_key(c)))
    qualification = ordered[(len(ordered) - 1) // 2]
    headroom = []
    for lo, hi, target in [(32768, 49152, 40960), (98304, 122880, 110592)]:
        candidates = [c for c in lb if lo <= token_length(c) < hi]
        if not candidates:
            raise ValueError('headroom length band has no input')
        headroom.append(min(candidates, key=lambda c: (abs(token_length(c) - target), metadata_key(c))))
    return dict(longbench_v2=chosen, ruler128k=selected_ruler,
                qualification=qualification, headroom=headroom)


def entry(policy, label=None, *, warmup=False, group_id=None):
    return dict(policy=policy, label=label or policy, warmup=warmup, group_id=group_id)


def factorial_schedule(pool, repeat, seeds=(1, 42)):
    cells, arms = [], [entry('control', 'implicit_warmup_control', warmup=True)]
    warm = dict(pool[0], seed=42)
    for policy in FACTORIAL:
        cells.append(dict(warm)); arms.append(entry(policy, 'warmup_' + policy, warmup=True))
    for ordinal, cell in enumerate(pool):
        for seed in seeds:
            shifted = (ordinal + seed + repeat) % len(FACTORIAL)
            policies = list(FACTORIAL[shifted:] + FACTORIAL[:shifted])
            if repeat % 2:
                policies.reverse()
            group_id = f"{cell['dataset']}:{cell['index']}:{seed}"
            group = [entry('control', 'control_before', group_id=group_id)]
            group += [entry(p, group_id=group_id) for p in policies]
            group += [entry('control', 'control_after', group_id=group_id)]
            cells.extend(dict(cell, seed=seed) for _ in group); arms.extend(group)
    return cells, arms


def headroom_schedule(pool, family, clean):
    cells, arms = [], [entry('control', 'implicit_warmup_control', warmup=True)]
    policies = ('control', 'fixed16') if family == 'shared' else ('control',)
    # Same request/input warmup inventory for every substrate family.
    warm_policies = policies if family == 'shared' else ('control', 'control')
    for ordinal, policy in enumerate(warm_policies):
        cells.append(dict(pool[0], seed=42))
        arms.append(entry(policy, f'warmup_{ordinal}_' + policy, warmup=True))
    for rep in range(2 if clean else 1):
        for cell in pool:
            for policy in (policies[::-1] if rep % 2 else policies):
                cells.append(dict(cell, seed=42))
                label = ('dynamic' if policy == 'control' else policy) if family == 'shared' else family
                arms.append(entry(policy, label, group_id=f"{cell['index']}:42:{rep}"))
    return cells, arms


def validate_budget(budgets, total):
    if set(budgets) != set(BUDGET) or any(not math.isfinite(v) or v < 0 for v in budgets.values()):
        raise ValueError('invalid phase budget')
    if not math.isfinite(total) or total <= 0 or sum(budgets.values()) > total + 1e-9 or total > TOTAL_SECONDS:
        raise ValueError('phase budgets exceed the frozen campaign cap')


def charged_seconds(reservations):
    """An orphaned child conservatively consumes its full reserved deadline."""
    charges = defaultdict(float)
    for row in reservations:
        elapsed = row.get('elapsed_s')
        charge = row['reserved_s'] if elapsed is None else elapsed
        if not math.isfinite(charge) or charge < 0:
            raise ValueError('invalid reservation accounting')
        charges[row['phase']] += charge
    return charges


def available_seconds(phase, reservations, budgets=BUDGET, total=TOTAL_SECONDS):
    validate_budget(budgets, total)
    used = charged_seconds(reservations)
    return max(0., min(budgets[phase] - used[phase], total - sum(used.values())))


def pause_unchanged():
    if base.read(LB_PARENT / 'PAUSED_USER_REQUEST.json').get('status') != 'paused_user_request':
        raise RuntimeError('prior full503 pause changed; this diagnostic cannot resume it')


def committed_sources():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = [p for p in tracked if p and (p.startswith(('src/', 'scripts/', 'experiments/', 'tests/'))
             and Path(p).suffix in ('.py', '.json', '.yaml', '.yml', '.toml') or p in (SPEC, RECOVERY_SPEC))]
    if not set(REQUIRED) <= set(files):
        raise RuntimeError('commit the specification and every diagnostic implementation/test before prepare')
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).splitlines()
    if set(files) & set(changed):
        raise RuntimeError('tracked execution source differs from the committed source')
    return files, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def native_contract():
    config, gen = base.read(base.MODEL / 'config.json'), base.read(base.MODEL / 'generation_config.json')
    return dict(canvas=config['canvas_length'], denoise_steps=gen['max_denoising_steps'],
        confidence_threshold=gen['confidence_threshold'], entropy_bound=gen['sampler_config']['entropy_bound'],
        stability_threshold=gen['stability_threshold'], temperature=[gen['t_max'], gen['t_min']], seed_base=31,
        thinking_policy='unchanged per-input manifests', output_limits='unchanged per-input native budgets')


def prepare(study, recovery_parent=None):
    if study.exists():
        raise RuntimeError('preserve existing attempt; select a new directory')
    files, commit = committed_sources(); pause_unchanged()
    parents = dict(longbench_v2=LB_PARENT, ruler128k=RULER_PARENT)
    source_cells = {s: base.read(p / f'private/schedules/{s}/cells.json') for s, p in parents.items()}
    rp = base.read(RULER_PARENT / 'protocol.json')
    ruler_meta = base.read(RULER_PARENT / 'private/pools/ruler128k' /
                           (rp['suites']['ruler128k']['dataset'] + '_rowinfo.json'))['rows']
    cohorts = choose_cohorts(source_cells['longbench_v2'], source_cells['ruler128k'], ruler_meta)
    runtime = base.runtime_identity(); native = native_contract()
    study.mkdir(parents=True); execution = study / 'private/execution'
    for rel in files:
        dest = execution / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    inventory = LB_PARENT / 'model_source_fingerprints.json'
    shutil.copyfile(inventory, study / 'model_source_fingerprints.json')
    for item in base.read(inventory)['files']:
        p = base.MODEL / item['file']
        if not p.is_file() or p.stat().st_size != item['bytes']:
            raise RuntimeError('model inventory size mismatch')
        if not item['file'].endswith('.safetensors') and base.sha(p) != item['sha256']:
            raise RuntimeError('model metadata fingerprint mismatch')
    suites, fingerprints = {}, {}
    for suite, parent in parents.items():
        suites[suite] = base.read(parent / 'protocol.json')['suites'][suite]
        dest = study / f'private/pools/{suite}'
        shutil.copytree(parent / f'private/pools/{suite}', dest)
        for p in dest.rglob('*'):
            if p.is_file():
                p.chmod(0o444); fingerprints[str(p.relative_to(study))] = base.sha(p)
    scorer = study / 'private/scorer_location.json'
    shutil.copyfile(RULER_PARENT / 'private/scorer_location.json', scorer)
    fingerprints[str(scorer.relative_to(study))] = base.sha(scorer)
    ruler_checkout = official.ruler_verify_checkout(base.read(scorer)['ruler'])
    plans, expected = [], []

    def add(phase, suite, family, pass_name, repeat, cells, arms):
        engine = len(plans) + 1
        leaf = study / phase / f'engine{engine:03d}_{suite}_{family}_{pass_name}' / 'attempt001'
        base.write(leaf / 'private/cells.json', cells, True)
        base.write(leaf / 'private/arms.json', arms, True)
        base.check_inputs(study / f'private/pools/{suite}', leaf / 'private/cells.json')
        measured = sum(not arm['warmup'] for arm in arms[1:])
        plan = dict(engine=engine, phase=phase, suite=suite, family=family, pass_name=pass_name, repeat=repeat,
            folder=str(leaf.relative_to(study)), request_count=len(cells), measured_count=measured,
            cells_sha256=base.sha(leaf / 'private/cells.json'), arms_sha256=base.sha(leaf / 'private/arms.json'))
        plans.append(plan)
        if phase == 'factorial':
            unique = {(cell['index'], cell['seed']) for cell, arm in zip(cells, arms[1:]) if not arm['warmup']}
            expected.extend(dict(suite=suite, index=index, panel_seed=seed, engine=engine) for index, seed in sorted(unique))

    add('qualification', 'longbench_v2', 'shared', 'audit', 0,
        *factorial_schedule([cohorts['qualification']], 0, (42,)))
    for family in ('shared', 'dense_full_fix51994', 'dense_piecewise', 'allkept_fa4'):
        add('headroom', 'longbench_v2', family, 'clean', 0, *headroom_schedule(cohorts['headroom'], family, True))
    for family in ('shared', 'dense_full_fix51994', 'allkept_fa4'):
        add('headroom', 'longbench_v2', family, 'profile', 0, *headroom_schedule(cohorts['headroom'], family, False))
    # Alternate suites by engine repetition so short RULER exposure is visible.
    for repeat in range(2):
        for suite in ('longbench_v2', 'ruler128k'):
            add('factorial', suite, 'shared', 'audit', repeat, *factorial_schedule(cohorts[suite], repeat))
    public = lambda c: dict(dataset=c['dataset'], index=c['index'], prompt_tokens=token_length(c), metadata_sha256=metadata_key(c))
    protocol = dict(schema='v31_refresh_diagnostic_v1', status='frozen_bounded_diagnostic', source_commit=commit,
        spec_sha256=base.sha(ROOT / SPEC), runtime=runtime, suites=suites, frozen_native=native,
        model_inventory_sha256=base.sha(study / 'model_source_fingerprints.json'),
        model_identity='google/diffusiongemma-26B-A4B-it@' + base.MODEL.name,
        execution_fingerprints={str(p.relative_to(execution)): base.sha(p) for p in execution.rglob('*') if p.is_file()},
        input_fingerprints=fingerprints, parent_protocols={s: base.sha(p / 'protocol.json') for s, p in parents.items()},
        budgets_seconds=BUDGET, total_gpu_seconds=TOTAL_SECONDS, minimum_engine_start_seconds=MIN_START_SECONDS,
        plans=plans, expected_cells=expected, panel_seeds=[1, 42], measured_factorial_requests=288,
        cohorts={s: [public(c) | (dict(task_base=ruler_meta[c['index']]['task_base']) if s == 'ruler128k' else {})
                     for c in cohorts[s]] for s in ('longbench_v2', 'ruler128k')},
        qualification_input=public(cohorts['qualification']), headroom_inputs=[public(c) for c in cohorts['headroom']],
        selection='metadata hash only, two per processed-length bin; four distinct RULER task_base families; previously examined inputs',
        ruler_checkout=ruler_checkout, reserve_policy='fresh source and successor after root review for fixes; no automatic reserve spending or phase borrowing',
        timing='only headroom clean records permit latency claims; profiling and factorial are audit only',
        failure_policy='preserve completed rows, stop on structural failure; missing or mismatched causal prefixes remain outcomes')
    if recovery_parent is not None:
        bind_successor(study, protocol, recovery_parent)
    base.write(study / 'protocol.json', protocol, True)
    base.write(study / 'receipts/source_fingerprints.json', dict(source_commit=commit,
        protocol_sha256=base.sha(study / 'protocol.json'), execution_fingerprints=protocol['execution_fingerprints']), True)
    print(json.dumps(dict(event='prepared', study=str(study.relative_to(ROOT)), engines=len(plans),
                         measured_factorial_requests=288, source_commit=commit)), flush=True)
    return protocol


def _contained(root, relative):
    """References and marker members must stay inside their declared directory."""
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise RuntimeError('unsafe artifact reference')
    path = (root / rel).resolve()
    if root.resolve() not in path.parents:
        raise RuntimeError('artifact reference escapes its root')
    return path


def plan_artifact_leaf(study, plan):
    reference = plan.get('source_reference')
    if reference is None:
        return _contained(study, plan['folder'])
    attempt = reference['attempt']
    if Path(attempt).name != attempt or attempt in ('.', '..', study.name):
        raise RuntimeError('reference must name a distinct sibling attempt')
    parent = _contained(study.parent, attempt)
    return _contained(parent, reference['folder'])


def validate_completion(leaf, plan, protocol_hash, source_commit, marker_hash=None):
    marker_path = leaf / 'attempt_complete.json'
    if marker_hash is not None and base.sha(marker_path) != marker_hash:
        raise RuntimeError('referenced completion marker changed')
    marker = base.read(marker_path)
    checks = dict(status='complete', errors=[], returncode=0, timed_out=False,
                  protocol_sha256=protocol_hash, source_commit=source_commit,
                  records=plan['request_count'], private_records=plan['request_count'],
                  measured_records=plan['measured_count'], planned_measured=plan['measured_count'])
    if any(marker.get(key) != value for key, value in checks.items()):
        raise RuntimeError('completed engine identity or coverage mismatch')
    required = {'records.jsonl', 'private/generation.private.jsonl', 'scored.json', 'config.json', 'receipt.json'}
    if plan['phase'] == 'qualification':
        required.add('qualification.json')
    members = marker.get('files_sha256', {})
    if not required <= set(members):
        raise RuntimeError('completion marker omits required artifact hashes')
    for name, digest in members.items():
        if base.sha(_contained(leaf, name)) != digest:
            raise RuntimeError('completed engine content changed')
    config = base.read(leaf / 'config.json')
    if config.get('protocol_sha256') != protocol_hash or config.get('source_commit') != source_commit:
        raise RuntimeError('completed config identity mismatch')
    for name in ('cells', 'arms'):
        if base.sha(leaf / f'private/{name}.json') != plan[name + '_sha256']:
            raise RuntimeError('completed engine schedule drift')
    return marker


def _recovery_parent(study, protocol):
    recovery = protocol['recovery']
    attempt = recovery['parent_attempt']
    if Path(attempt).name != attempt or attempt in ('.', '..', study.name):
        raise RuntimeError('recovery requires a distinct sibling parent')
    parent = _contained(study.parent, attempt)
    if base.sha(parent / 'protocol.json') != recovery['parent_protocol_sha256']:
        raise RuntimeError('parent protocol changed')
    if base.sha(parent / 'state.json') != recovery['parent_state_sha256']:
        raise RuntimeError('parent terminal state changed')
    original, state = base.read(parent / 'protocol.json'), base.read(parent / 'state.json')
    if original.get('recovery') or state.get('status') != 'failed' or set(state.get('completed_engines', [])) != set(REFERENCE_ENGINES):
        raise RuntimeError('recovery requires the preserved failed parent with engines 1 through 5 complete')
    if (state.get('failure') or {}).get('engine') != 6:
        raise RuntimeError('recovery is limited to the failed profiling engine 6')
    if state.get('protocol_sha256') != recovery['parent_protocol_sha256'] or original['source_commit'] != recovery['parent_source_commit']:
        raise RuntimeError('parent source/protocol identity mismatch')
    if recovery.get('referenced_engines') != list(REFERENCE_ENGINES):
        raise RuntimeError('unexpected recovery reference set')
    ledger = base.read(study.parent / 'gpu_budget.json')
    parent_reservations = [row for row in ledger['reservations'] if row.get('attempt') == attempt]
    if (len(parent_reservations) != 6 or {row.get('engine') for row in parent_reservations} != set(range(1, 7))
            or any(row.get('elapsed_s') is None for row in parent_reservations)
            or digest_metadata(parent_reservations) != recovery['parent_reservations_sha256']):
        raise RuntimeError('parent GPU reservations changed or are incomplete')
    if not math.isclose(sum(charged_seconds(parent_reservations).values()), state.get('cumulative_gpu_seconds', -1), abs_tol=1e-6):
        raise RuntimeError('parent failed work is missing from campaign accounting')
    if protocol['budgets_seconds'] != RECOVERY_BUDGET or protocol['total_gpu_seconds'] != TOTAL_SECONDS:
        raise RuntimeError('recovery budget differs from the frozen addendum')
    if protocol['execution_fingerprints'].get(RECOVERY_SPEC) != recovery['addendum_sha256']:
        raise RuntimeError('recovery addendum fingerprint mismatch')
    same_fields = ('cohorts', 'expected_cells', 'frozen_native', 'headroom_inputs', 'qualification_input',
                   'input_fingerprints', 'model_identity', 'model_inventory_sha256', 'panel_seeds',
                   'parent_protocols', 'ruler_checkout', 'runtime', 'selection', 'spec_sha256', 'suites',
                   'measured_factorial_requests', 'minimum_engine_start_seconds', 'total_gpu_seconds')
    if any(protocol.get(key) != original.get(key) for key in same_fields):
        raise RuntimeError('recovery changed native, runtime, cohort, or input identity')
    old_source, new_source = original['execution_fingerprints'], protocol['execution_fingerprints']
    changed = {key for key in set(old_source) | set(new_source) if old_source.get(key) != new_source.get(key)}
    if changed - RECOVERY_SOURCE_ALLOWLIST:
        raise RuntimeError('recovery changed protected execution source: ' + ', '.join(sorted(changed - RECOVERY_SOURCE_ALLOWLIST)))
    if len(original['plans']) != len(protocol['plans']):
        raise RuntimeError('recovery changed plan inventory')
    for old, new in zip(original['plans'], protocol['plans']):
        if old != {key: value for key, value in new.items() if key != 'source_reference'}:
            raise RuntimeError('recovery changed a frozen plan or schedule')
    return parent, original


def validate_references(study, protocol):
    referenced = [plan for plan in protocol['plans'] if 'source_reference' in plan]
    if 'recovery' not in protocol:
        if referenced:
            raise RuntimeError('artifact references require a frozen recovery contract')
        return []
    parent, original = _recovery_parent(study, protocol)
    for rel, digest in original['execution_fingerprints'].items():
        if base.sha(_contained(parent / 'private/execution', rel)) != digest:
            raise RuntimeError('parent frozen source changed: ' + rel)
    if [plan['engine'] for plan in referenced] != list(REFERENCE_ENGINES):
        raise RuntimeError('only completed parent engines 1 through 5 may be referenced')
    for plan in referenced:
        reference = plan['source_reference']
        expected = dict(attempt=parent.name, folder=plan['folder'],
                        protocol_sha256=protocol['recovery']['parent_protocol_sha256'],
                        source_commit=original['source_commit'])
        if any(reference.get(key) != value for key, value in expected.items()) or not reference.get('marker_sha256'):
            raise RuntimeError('source reference identity mismatch')
        validate_completion(plan_artifact_leaf(study, plan), plan, reference['protocol_sha256'],
                            reference['source_commit'], reference['marker_sha256'])
        for name in ('cells', 'arms'):
            if base.sha(study / plan['folder'] / f'private/{name}.json') != plan[name + '_sha256']:
                raise RuntimeError('successor reference schedule drift')
    return [plan['engine'] for plan in referenced]


def bind_successor(study, protocol, parent):
    parent = parent.resolve()
    if parent.parent != study.resolve().parent or parent == study.resolve():
        raise RuntimeError('successor must share the parent campaign directory and budget ledger')
    original = base.read(parent / 'protocol.json')
    ledger = base.read(study.parent / 'gpu_budget.json')
    parent_reservations = [row for row in ledger['reservations'] if row.get('attempt') == parent.name]
    protocol['budgets_seconds'] = dict(RECOVERY_BUDGET)
    protocol['recovery'] = dict(parent_attempt=parent.name, parent_protocol_sha256=base.sha(parent / 'protocol.json'),
        parent_state_sha256=base.sha(parent / 'state.json'), parent_source_commit=original['source_commit'],
        parent_reservations_sha256=digest_metadata(parent_reservations),
        referenced_engines=list(REFERENCE_ENGINES), addendum_sha256=protocol['execution_fingerprints'][RECOVERY_SPEC])
    _recovery_parent(study, protocol)
    for plan in protocol['plans']:
        if plan['engine'] in REFERENCE_ENGINES:
            plan['source_reference'] = dict(attempt=parent.name, folder=plan['folder'],
                protocol_sha256=protocol['recovery']['parent_protocol_sha256'], source_commit=original['source_commit'],
                marker_sha256=base.sha(parent / plan['folder'] / 'attempt_complete.json'))
    validate_references(study, protocol)


def verify(study, protocol, *, inputs=True):
    pause_unchanged()
    # Host-side orchestration/scoring must match the same frozen source too.
    for rel in ('scripts/v31_refresh_diagnostic_study.py', 'scripts/v31_v2_local_threshold_sweep.py',
                'scripts/v31_official.py', 'scripts/v31_vllm_paired_bench.py',
                'experiments/diffusion_gemma_aime26_modes/protocol.py'):
        if base.sha(ROOT / rel) != protocol['execution_fingerprints'][rel]:
            raise RuntimeError('host coordinator/scorer source drift: ' + rel)
    for rel, digest in protocol['execution_fingerprints'].items():
        if base.sha(study / 'private/execution' / rel) != digest:
            raise RuntimeError('frozen source drift: ' + rel)
    if inputs:
        for rel, digest in protocol['input_fingerprints'].items():
            if base.sha(study / rel) != digest:
                raise RuntimeError('frozen input drift')
    if base.sha(study / 'model_source_fingerprints.json') != protocol['model_inventory_sha256']:
        raise RuntimeError('model inventory drift')
    for item in base.read(study / 'model_source_fingerprints.json')['files']:
        p = base.MODEL / item['file']
        if not p.is_file() or p.stat().st_size != item['bytes']:
            raise RuntimeError('model inventory drift')
        if not item['file'].endswith('.safetensors') and base.sha(p) != item['sha256']:
            raise RuntimeError('model metadata drift')
    if base.runtime_identity() != protocol['runtime']:
        raise RuntimeError('runtime drift')
    return validate_references(study, protocol)


def require_idle():
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied; no worker was launched')


def validate_row(row, cell, arm, ordinal, plan, settings):
    errors = []
    adapter = (row.get('receipts') or {}).get('adapter') or {}
    checks = [(row.get('dataset') == cell['dataset'] and row.get('index') == cell['index'] and
               row.get('panel_seed') == cell['seed'] and row.get('rng_seed') == request_seed(31, cell, 0), 'input/seed identity'),
        (row.get('cuda_graph_captures') == 0, 'new CUDA graph capture'),
        (row.get('fix_51994') is True and row.get('fa4_local_fix') is True, 'dense correction identity'),
        (row.get('value_clean_timing') == (plan['pass_name'] == 'clean'), 'timing identity'),
        (row.get('timer_policy') == ('request_and_prefill_boundaries' if plan['pass_name'] == 'clean' else 'synchronized_each_engine_step'), 'timer policy'),
        (row.get('cudagraph_mode') == ('default' if plan['family'] == 'dense_full_fix51994' else 'PIECEWISE'), 'graph mode identity'),
        (row.get('sampler_calls') == row.get('denoise_forwards', -100) + row.get('canvases', -100), 'native sampler count'),
        (row.get('engine_steps') == row.get('prefill_steps', -100) + row.get('sampler_calls', -100), 'native engine count'),
        (row.get('block_size') == settings['block'] and row.get('chunk') == settings['chunk'] and
         row.get('max_model_len') == settings['max_len'], 'native geometry')]
    if plan['family'] in ('shared', 'allkept_fa4'):
        checks += [(adapter.get('refresh_label') == arm['label'] and adapter.get('refresh_policy') == arm['policy'] and
                    adapter.get('refresh_request_index') == ordinal + 1, 'refresh schedule identity')]
    if plan['pass_name'] != 'clean':
        checks += [(adapter.get('native_local_skipped_tiles') == 0, 'LOCAL scope'),
                   (adapter.get('global_calls') == 5 * row.get('denoise_forwards', -1), 'GLOBAL layer count'),
                   (adapter.get('native_local_calls_audited') == 25 * row.get('denoise_forwards', -1), 'LOCAL layer count'),
                   (adapter.get('global_eligible_tiles', -1) >= adapter.get('global_kept_tiles', -1) >= 0, 'physical work accounting')]
    if plan['family'] == 'shared':
        checks += [(row.get('mage_k') == settings['mage_k'], 'fixed prefix budget'),
                   (row.get('local_kv_budget') is None and row.get('value_selector') is None, 'unexpected selector or LOCAL sparsity')]
        diagnostic = adapter.get('refresh_diagnostic')
        checks.append((isinstance(diagnostic, dict), 'missing refresh diagnostic'))
        if isinstance(diagnostic, dict):
            checks += [(diagnostic.get('policy') == arm['policy'], 'refresh policy'),
                       (diagnostic.get('scope') == [5, 11, 17, 23, 29], 'refresh GLOBAL scope'),
                       (diagnostic.get('complete_scope') is True and diagnostic.get('errors') == [], 'refresh episode structure'),
                       (diagnostic.get('added_model_forwards') == 0 and diagnostic.get('added_sampler_calls') == 0, 'added native calls')]
            if diagnostic.get('exposed') and arm['policy'] in FACTORIAL:
                checks.append((isinstance(diagnostic.get('prefix_signature'), str), 'missing exposed prefix proof'))
            for event in diagnostic.get('events', []):
                checks.append((event.get('rng_unchanged') is True, 'diagnostic RNG changed'))
    errors.extend(name for ok, name in checks if not ok)
    return errors


def score_completed(study, protocol, plan, rows, private, cells, arms):
    """Bind/score only complete paired rows; incomplete group coverage is retained."""
    suite = plan['suite']; pool = study / f'private/pools/{suite}'
    pools = official.Pools(pool, fields=('thinking', 'task', 'task_base', 'prefilled'))
    final, _ = official.load_final_response(); gold = base.read(pool / (protocol['suites'][suite]['dataset'] + '_gold.json'))
    metrics = task_bases = None
    if suite == 'ruler128k':
        ruler = base.read(study / 'private/scorer_location.json')['ruler']
        official.ruler_verify_checkout(ruler)
        metrics, task_bases = official.ruler_metrics(ruler), official.ruler_task_bases(ruler)
    errors, scored = [], []
    if len(rows) > len(cells) or len(private) > len(cells):
        errors.append('extra generated requests')
    for ordinal, (row, raw, cell, arm) in enumerate(zip(rows, private, cells, arms[1:])):
        errors.extend(validate_row(row, cell, arm, ordinal, plan, protocol['suites'][suite]))
        meta, binding, bound = official.bind(raw, pools)
        if any(key not in raw or key not in row or raw[key] != row[key] for key in official.COMPARE_FIELDS):
            errors.append('public/private binding mismatch')
        if not bound or raw['dataset'] != row['dataset'] or raw['index'] != cell['index'] or raw['panel_seed'] != cell['seed'] or raw['rng_seed'] != row['rng_seed']:
            errors.append('private record identity')
        if suite == 'longbench_v2':
            score = float(official.lb_extract_answer(final(raw['completion'], bool(meta['thinking']))) == gold[raw['id']])
        else:
            answer = official.cut_at_end(raw['completion']) if meta.get('prefilled') else final(raw['completion'], bool(meta.get('thinking')))
            fn = metrics[meta.get('task_base') or task_bases[meta['task']]]
            score = official.ruler_sample_score(fn, official.ruler_postprocess(answer), gold[raw['id']]) / 100.
        if arm['warmup']:
            continue
        adapter = (row.get('receipts') or {}).get('adapter') or {}
        result = dict(suite=suite, dataset=row['dataset'], index=row['index'], panel_seed=row['panel_seed'],
            engine=plan['engine'], phase=plan['phase'], pass_name=plan['pass_name'], family=plan['family'],
            engine_repeat=plan['repeat'], sample_repeat=int(arm['group_id'].rsplit(':', 1)[1]) if plan['phase'] == 'headroom' else 0,
            group_id=arm['group_id'], policy=arm['policy'], label=arm['label'], score=score,
            score_scale='0 to 1; official RULER percent divided by 100', diagnostic_receipt=adapter.get('refresh_diagnostic'),
            refresh_post_event_states=adapter.get('refresh_post_event_states'), native_receipt=adapter,
            protocol_sha256=base.sha(study / 'protocol.json'), source_commit=protocol['source_commit'])
        for key in ('denoise_forwards', 'canvases', 'output_tokens', 'finish_reason', 'output_hash', 'decode_s', 'wall_s',
                    'prefill_s', 'prompt_tokens', 'rng_seed', 'cuda_graph_captures', 'peak_cuda_allocated_bytes', 'value_clean_timing'):
            result[key] = row.get(key)
        scored.append(result)
    return sorted(set(errors)), scored


def qualification_summary(rows, structural_errors):
    by_label = {row['label']: row for row in rows}
    needed = set(FACTORIAL) | {'control_before', 'control_after'}
    complete = set(by_label) == needed
    receipts = [by_label[label].get('diagnostic_receipt') or {} for label in FACTORIAL if label in by_label]
    exposed = len(receipts) == 4 and all(r.get('exposed') is True for r in receipts)
    signatures = [r.get('prefix_signature') for r in receipts]
    verified = len(signatures) == 4 and all(isinstance(s, str) and s for s in signatures)
    prefix = 'matched' if verified and len(set(signatures)) == 1 else 'mismatched' if verified else 'unverified'
    parity = None
    if complete:
        fields = ('output_hash', 'denoise_forwards', 'canvases', 'output_tokens')
        parity = all(by_label['dense_fresh'][k] == by_label['control_before'][k] == by_label['control_after'][k] for k in fields)
    return dict(structural_status='passed' if complete and not structural_errors else 'failed',
                all_four_exposed=exposed, factorial_prefix_status=prefix, dense_fresh_and_aa_exact_parity=parity,
                causal_status='qualified' if complete and not structural_errors and exposed and prefix == 'matched' and parity else 'unresolved',
                interpretation='prefix mismatch, no exposure and no-op discrepancies remain outcomes; root reviews before continuation')


def worker_command(study, plan, protocol):
    execution = study / 'private/execution'; leaf = study / plan['folder']; clean = plan['pass_name'] == 'clean'
    family = plan['family']; arm = 'mage' if family == 'shared' else 'allkept' if family == 'allkept_fa4' else 'dense'
    cg = 'default' if family == 'dense_full_fix51994' else 'PIECEWISE'
    base.SUITES.update(protocol['suites'])
    env = base.env_for(study, plan['suite'], 'current_v31_control')
    for key in list(env):
        if key.startswith(('REFRESH_', 'HEADROOM_')):
            env.pop(key)
    env.update(STUDY_ARM='refresh_' + family, PYTHONHASHSEED='0',
        FROZEN_SOURCE_SHA256=base.sha(study / 'receipts/source_fingerprints.json'),
        REFRESH_SOURCE_COMMIT=protocol['source_commit'], REFRESH_PROTOCOL_SHA256=base.sha(study / 'protocol.json'),
        REFRESH_SCHEDULE=str(leaf / 'private/arms.json'),
        REFRESH_BASE_WORKER=str(execution / 'scripts/v31_vllm_paired_bench.py'),
        REFRESH_AUDIT=str(int(plan['phase'] in ('qualification', 'factorial'))),
        HEADROOM_PROFILE=str(int(plan['pass_name'] == 'profile')),
        HEADROOM_OUTPUT=str(leaf / 'private/headroom'), HEADROOM_PROFILE_INDEX='5' if family == 'shared' else '4',
        VALUE_AUDIT=str(int(not clean)), VALUE_CLEAN_TIMING=str(int(clean)), VALUE_BOUNDARY_TIMING=str(int(clean)))
    cmd = [base.PYTHON, str(execution / 'scripts/v31_refresh_diagnostic_worker.py'), str(base.MODEL),
        str(study / f"private/pools/{plan['suite']}"), str(leaf / 'private/cells.json'), str(leaf / 'records.jsonl'),
        str(leaf / 'private/generation.private.jsonl'), arm, cg]
    return cmd, env


def complete_jsonl(path, *, allow_truncated=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_truncated and i == len(lines) - 1:
                break
            raise
    return rows


def stop_owned_process(proc):
    """Only signal the new session created by this coordinator."""
    if proc.poll() is not None:
        return
    try:
        if os.getpgid(proc.pid) != proc.pid:
            raise RuntimeError('child no longer owns its recorded process group')
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=2.)
    except ProcessLookupError:
        proc.wait()


def run(study, stop_after=None):
    protocol = base.read(study / 'protocol.json')
    validate_budget(protocol['budgets_seconds'], protocol['total_gpu_seconds'])
    previous = base.read(study / 'state.json') if (study / 'state.json').exists() else None
    if previous is not None and previous.get('status') != 'phase_stop':
        raise RuntimeError('only a deliberately stopped intact phase can continue; failures need a new attempt')
    if previous is not None and previous.get('protocol_sha256') != base.sha(study / 'protocol.json'):
        raise RuntimeError('stopped attempt protocol changed')
    # The shared local lock spans successors; external work is additionally
    # protected by the immediate nvidia-smi idle check before every launch.
    lock_path = ROOT / 'results/.v31_refresh_h100.lock'
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another refresh diagnostic owns the GPU lock') from error
        ledger_path = study.parent / 'gpu_budget.json'
        ledger = base.read(ledger_path) if ledger_path.exists() else dict(schema='refresh_gpu_reservations_v1', reservations=[])
        reservations = ledger['reservations']; total = protocol['total_gpu_seconds']; budgets = protocol['budgets_seconds']
        remaining = total - sum(charged_seconds(reservations).values())
        if remaining <= 0:
            raise RuntimeError('campaign GPU budget exhausted')
        campaign_deadline = time.monotonic() + remaining
        referenced = verify(study, protocol)
        completed = list(previous.get('completed_engines', [])) if previous else []
        skipped = list(previous.get('skipped', [])) if previous else []
        failure, stopped = None, False
        base.write(study / 'state.json', dict(status='running', source_commit=protocol['source_commit'], referenced_engines=referenced))
        phases = ('qualification', 'headroom', 'factorial')
        for plan in protocol['plans']:
            phase = plan['phase']; leaf = study / plan['folder']
            if plan['engine'] in referenced:
                if plan['engine'] not in completed:
                    completed.append(plan['engine'])
                continue
            if plan['engine'] in completed:
                validate_completion(leaf, plan, base.sha(study / 'protocol.json'), protocol['source_commit'])
                continue
            if stop_after is not None and phases.index(phase) > phases.index(stop_after):
                stopped = True; break
            if phase == 'factorial':
                qualification_plan = next(p for p in protocol['plans'] if p['phase'] == 'qualification')
                qpath = plan_artifact_leaf(study, qualification_plan) / 'qualification.json'
                if not qpath.exists() or base.read(qpath).get('causal_status') != 'qualified':
                    skipped.append(dict(engine=plan['engine'], phase=phase, reason='causal_gate_unresolved'))
                    continue
            seconds = min(available_seconds(phase, reservations, budgets, total), campaign_deadline - time.monotonic())
            if seconds < protocol['minimum_engine_start_seconds']:
                skipped.append(dict(engine=plan['engine'], phase=phase, reason='insufficient frozen phase/campaign budget'))
                continue
            if any((leaf / name).exists() for name in ('receipt.json', 'records.jsonl', 'private/worker.log')):
                raise RuntimeError('preserve prior worker output')
            for name in ('cells', 'arms'):
                if base.sha(leaf / f'private/{name}.json') != plan[name + '_sha256']:
                    raise RuntimeError('frozen schedule drift')
            verify(study, protocol, inputs=False); require_idle()
            seconds = min(seconds, campaign_deadline - time.monotonic())
            if seconds < protocol['minimum_engine_start_seconds']:
                skipped.append(dict(engine=plan['engine'], phase=phase, reason='verification consumed remaining budget'))
                continue
            cmd, env = worker_command(study, plan, protocol)
            base.write(leaf / 'config.json', dict(plan, protocol_sha256=base.sha(study / 'protocol.json'),
                source_commit=protocol['source_commit'], maximum_reservation_s=seconds), True)
            reservation = dict(attempt=str(study.relative_to(study.parent)), engine=plan['engine'], phase=phase,
                               reserved_s=seconds, elapsed_s=None, status='started')
            reservations.append(reservation); base.write(ledger_path, ledger)
            started = time.monotonic(); proc = None; timed_out = False; launch_error = None
            print(json.dumps(dict(event='launch', engine=plan['engine'], phase=phase, suite=plan['suite'],
                                  measured=plan['measured_count'], deadline_seconds=round(seconds, 2))), flush=True)
            try:
                with (leaf / 'private/worker.log').open('x') as log:
                    proc = subprocess.Popen(cmd, cwd=study / 'private/execution', env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    reservation['pid'] = proc.pid; base.write(ledger_path, ledger)
                    while proc.poll() is None:
                        left = seconds - (time.monotonic() - started) - 4.
                        if left <= 0:
                            timed_out = True; stop_owned_process(proc); break
                        try:
                            proc.wait(timeout=min(30., left))
                        except subprocess.TimeoutExpired:
                            pass
            except BaseException as error:
                if proc is not None:
                    stop_owned_process(proc)
                launch_error = type(error).__name__
            finally:
                elapsed = time.monotonic() - started
                reservation.update(elapsed_s=elapsed, status='deadline' if timed_out else 'finished',
                                   returncode=None if proc is None else proc.returncode)
                base.write(ledger_path, ledger)
            cells, arms = base.read(leaf / 'private/cells.json'), base.read(leaf / 'private/arms.json')
            rows, private, errors, scored = [], [], [], []
            try:
                rows = complete_jsonl(leaf / 'records.jsonl', allow_truncated=timed_out)
                private = complete_jsonl(leaf / 'private/generation.private.jsonl', allow_truncated=timed_out)
                errors, scored = score_completed(study, protocol, plan, rows, private, cells, arms)
            except Exception as error:
                errors.append('receipt/scoring failure: ' + type(error).__name__)
            full = len(rows) == len(private) == len(cells)
            if launch_error:
                errors.append('launch/interruption: ' + launch_error)
            if not timed_out and (proc is None or proc.returncode or not full):
                errors.append('worker failure or incomplete coverage without deadline')
            status = 'failed' if errors else 'budget_limited_partial' if timed_out or not full else 'complete'
            base.write(leaf / 'scored.json', scored, True)
            if phase == 'qualification':
                base.write(leaf / 'qualification.json', qualification_summary(scored, errors), True)
            receipt = dict(status=status, errors=sorted(set(errors)), records=len(rows), private_records=len(private),
                measured_records=len(scored), planned_measured=plan['measured_count'], gpu_reservation_s=elapsed,
                timed_out=timed_out, returncode=None if proc is None else proc.returncode,
                source_commit=protocol['source_commit'], protocol_sha256=base.sha(study / 'protocol.json'))
            base.write(leaf / 'receipt.json', receipt, True)
            if status == 'complete':
                base.write(leaf / 'attempt_complete.json', dict(receipt, files_sha256={name: base.sha(leaf / name)
                    for name in ('records.jsonl', 'private/generation.private.jsonl', 'scored.json', 'config.json', 'receipt.json')
                    + (('qualification.json',) if phase == 'qualification' else ())}), True)
                completed.append(plan['engine'])
            print(json.dumps(dict(event=status, engine=plan['engine'], measured_records=len(scored),
                                  gpu_seconds=round(elapsed, 2), errors=receipt['errors'])), flush=True)
            if errors or (phase == 'qualification' and status != 'complete'):
                failure = dict(engine=plan['engine'], status=status, errors=receipt['errors']); break
        final_status = ('failed' if failure else 'complete_diagnostic' if len(completed) == len(protocol['plans'])
                        else 'phase_stop' if stopped else 'causal_gate_unresolved'
                        if any(row['reason'] == 'causal_gate_unresolved' for row in skipped) else 'budget_limited_partial')
        state = dict(status=final_status, completed_engines=completed, referenced_engines=referenced, skipped=skipped, failure=failure,
            cumulative_gpu_seconds=sum(charged_seconds(reservations).values()),
            phase_gpu_seconds=dict(charged_seconds(reservations)), parent_full503='paused',
            protocol_sha256=base.sha(study / 'protocol.json'))
        base.write(study / 'state.json', state)
        if final_status == 'complete_diagnostic':
            base.write(study / 'final_complete.json', state, True)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'successor', 'run', 'verify'))
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--stop-after', choices=('qualification', 'headroom', 'factorial'))
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args(); study = args.study.resolve()
    if ROOT / 'results' not in study.parents:
        raise RuntimeError('study must be inside canonical results root')
    if args.action == 'prepare':
        if args.parent is not None:
            raise RuntimeError('use successor to bind a parent attempt')
        prepare(study)
    elif args.action == 'successor':
        if args.parent is None:
            raise RuntimeError('successor requires --parent')
        prepare(study, args.parent)
    elif args.action == 'verify':
        verify(study, base.read(study / 'protocol.json'))
    else:
        run(study, args.stop_after)


if __name__ == '__main__':
    main()
