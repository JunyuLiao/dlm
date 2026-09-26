"""CPU-only secondary v18 design plan: 70% arms and T60 allocation controls.

This freezes question selection and logical block order. Allocation controls
remain non-executable until their runtime identity is explicitly qualified.
No answers, steps, timing, or evaluation density enter threshold selection.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from scripts.v13_seed_runs import arm_config_hash, execution_key, plan_schedule
from scripts.v18_frontier import best_point, config_for, measured_density, midpoint_policy, next_policy
from scripts.v18_protocol import REVISION, SEEDS, arm_specs, read_rows, sha, strip_gold

CORE = ('U70', 'T70')
CONTROLS = ('T60_shuffled', 'T60_uniform')
CALIBRATION_ARMS = CORE + CONTROLS
TARGET = {'U70': .70, 'T70': .70, 'T60_shuffled': .60, 'T60_uniform': .60}
CALIBRATION_SEED = 42
MAX_THRESHOLD_PAIRS = 5
ALLOCATION_SEED = 424242


def _balanced(rows, expected_each: int, expected_total: int) -> list[dict]:
    rows = strip_gold(rows)
    tasks = Counter(r.get('task') for r in rows)
    if len(rows) != expected_total or len(tasks) != 13 or set(tasks.values()) != {expected_each}:
        raise ValueError('expected balanced 13-task frozen RULER questions')
    return rows


def mechanism_subset(ruler_rows: list[dict]) -> list[str]:
    """Independent deterministic two-per-task selection; no outcome reads."""
    rows = _balanced(ruler_rows, 10, 130)
    return [r['id'] for task in sorted({r['task'] for r in rows})
            for r in sorted((r for r in rows if r['task'] == task),
                            key=lambda r: sha('v18/mechanism/' + r['id']))[:2]]


def initial_policies(old_scope_policies: dict, primary_frozen: dict) -> dict:
    """Starting thresholds only; all four points need independent new-scope calibration."""
    if 'T60' not in primary_frozen or primary_frozen['T60'].get('attained') is not True:
        raise ValueError('attained primary T60 required as control calibration starting point')
    base = primary_frozen['T60']['policy']
    result = {arm: midpoint_policy(old_scope_policies, arm) for arm in CORE}
    result.update({arm: base for arm in CONTROLS})
    return {arm: validate_policy(policy) for arm, policy in result.items()}


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or set(policy) != {'local', 'global'}:
        raise ValueError('one LOCAL and one GLOBAL threshold required')
    result = {}
    for kind in ('local', 'global'):
        entry = policy[kind]
        if not isinstance(entry, dict) or set(entry) != {'log_threshold'}:
            raise ValueError('threshold pair only')
        value = entry['log_threshold']
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('finite threshold required')
        result[kind] = {'log_threshold': float(value)}
    return result


def update_density_history(arm: str, history: list[dict], policy: dict, actual: dict) -> dict:
    """Advance only from all-26 measured density; stop at ±2pp or five pairs."""
    if arm not in TARGET or len(history) >= MAX_THRESHOLD_PAIRS:
        raise ValueError('unknown arm or threshold-pair ceiling reached')
    if not isinstance(actual, dict) or set(actual) != {'whole', 'local', 'global'}:
        raise ValueError('all density strata required')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in actual.values()):
        raise ValueError('invalid density')
    clean = validate_policy(policy)
    entries = list(history)
    if any(set(item) != {'point', 'policy', 'actual'} or item['point'] != i
           for i, item in enumerate(entries)):
        raise ValueError('history must contain density-only consecutive points')
    entries.append(dict(point=len(entries), policy=clean, actual=dict(actual)))
    target = TARGET[arm]
    chosen = best_point(entries, target)
    attained = all(abs(chosen['actual'][kind] - target) <= .02 for kind in ('whole', 'local', 'global'))
    done = attained or len(entries) == MAX_THRESHOLD_PAIRS
    return dict(arm=arm, target=target, history=entries, done=done, attained=attained,
                best_point=chosen['point'], selected_policy=chosen['policy'],
                next_policy=None if done else validate_policy(next_policy(entries, target)))


def calibration_schedule(calibration_ids: list[str], policies: dict, point: int) -> dict:
    """One independent complete 26-question threshold pair per secondary arm."""
    if type(point) is not int or not 0 <= point < MAX_THRESHOLD_PAIRS:
        raise ValueError('calibration point outside five-pair ceiling')
    if len(calibration_ids) != 26 or len(set(calibration_ids)) != 26 or set(policies) != set(CALIBRATION_ARMS):
        raise ValueError('same frozen 26 questions and four secondary policies required')
    clean = {arm: validate_policy(policies[arm]) for arm in CALIBRATION_ARMS}
    specs = {arm: dict(arm=arm, target=TARGET[arm], policy=clean[arm],
                       allocation=('shuffle' if arm.endswith('shuffled') else
                                   'uniform' if arm.endswith('uniform') else 'normal'),
                       allocation_seed=ALLOCATION_SEED if arm in CONTROLS else 42,
                       rank=32, projection_seed=1729)
             for arm in CALIBRATION_ARMS}
    phase = 'v18_secondary_calibration_p' + str(point) + '_' + sha(json.dumps(
        [calibration_ids, specs, CALIBRATION_SEED], sort_keys=True))[:12]
    hashes = {arm: sha(json.dumps(spec, sort_keys=True)) for arm, spec in specs.items()}
    schedule = plan_schedule(phase, REVISION, hashes, calibration_ids, [CALIBRATION_SEED],
                             2026092606 + point, warm_repeats=0)
    return dict(schema='v18_secondary_calibration_design_v1', status='calibration_pending',
                point=point, ids=list(calibration_ids), seed=CALIBRATION_SEED,
                arms=specs, protocol_id=phase, schedule=schedule,
                planned_executions=104, max_threshold_pairs_per_arm=MAX_THRESHOLD_PAIRS,
                selection='density_only_whole_local_global')


def freeze_calibration_spec(input_path: Path, out: Path) -> dict:
    """Derive immutable same-26 calibration inputs from frozen sources, CPU only."""
    from experiments.numerical_qk_reuse.runner import _rows
    from scripts.v18_evaluate import logical_protocol_digest, write_immutable_json

    source = json.loads(input_path.read_text())
    group = source['group']
    arms = CORE if group == 'core70' else CONTROLS if group == 'controls' else None
    if arms is None:
        raise ValueError('unknown calibration pair')
    primary = json.loads(Path(source['primary_protocol']).read_text())
    aime = json.loads(Path(source['aime_protocol']).read_text())
    if any(p.get('status') != 'frozen' or logical_protocol_digest(p) != p.get('logical_protocol_sha256')
           for p in (primary, aime)) or primary['stage'] != 'ruler4k_primary' or aime['stage'] != 'aime26_primary':
        raise ValueError('frozen primary RULER/AIME protocol identity required')
    if (str(Path(source['primary_calibration']).resolve()) != primary['calibration_file'] or
            sha(Path(source['primary_calibration']).read_bytes()) != primary['calibration_sha256']):
        raise ValueError('primary calibrated T60 source drift')
    old_path = Path(source['old_scope_policies'])
    calibration_path = Path(source['manifest'])
    ids = [row['id'] for row in _balanced(_rows(calibration_path, allow_task_budgets=True,
                                                allow_thinking_off=True), 2, 26)]
    proposals = initial_policies(json.loads(old_path.read_text()),
                                 json.loads(Path(source['primary_calibration']).read_text()))
    prior = {'ruler4k_primary': dict(protocol=source['primary_protocol'],
                                    ledgers=source['primary_ledgers']),
             'aime26_primary': dict(protocol=source['aime_protocol'],
                                   ledgers=source['aime_ledgers'])}
    if group == 'controls':
        if not source.get('preceding_secondary') or not source.get('preceding_ledgers'):
            raise ValueError('frozen 70% protocol/ledgers required for controls')
        core = json.loads(Path(source['preceding_secondary']).read_text())
        if (core.get('stage') != 'ruler_secondary70' or core.get('status') != 'frozen' or
                logical_protocol_digest(core) != core.get('logical_protocol_sha256')):
            raise ValueError('frozen 70% stage identity required for controls')
        prior['ruler_secondary70'] = dict(protocol=source['preceding_secondary'],
                                          ledgers=source['preceding_ledgers'])
    assigned = (source['calibration_host'], source['calibration_gpu_uuid'])
    if not any((h['host'], h['gpu_uuid']) == assigned for h in primary['assigned_hosts']):
        raise ValueError('calibration host/GPU absent from frozen primary assignment')
    spec = dict(schema='v18_secondary_calibration_spec_v1', group=group,
                ids=ids, initial_policies={arm: proposals[arm] for arm in arms},
                manifest=str(calibration_path.resolve()), manifest_sha256=sha(calibration_path.read_bytes()),
                old_scope_policy_sha256=sha(old_path.read_bytes()),
                primary_calibration_sha256=primary['calibration_sha256'],
                prerequisite_protocol_ids={stage: json.loads(Path(item['protocol']).read_text())['protocol_id']
                                           for stage, item in prior.items()},
                prerequisites=prior,
                primary_summaries={'ruler4k_primary': source['ruler_summary'],
                                   'aime26_primary': source['aime_summary']},
                calibration_host=assigned[0], calibration_gpu_uuid=assigned[1],
                policy_file=source['policy_file'], model=source['model'],
                library=source['library'], torch_library=source['torch_library'],
                out=source['out'], lock=source['lock'], authorization=source['authorization'])
    write_immutable_json(out, spec)
    return spec


def freeze_calibration_point(manifest: Path, expected_ids: list[str], policy_file: Path, model: Path,
                             library: Path, torch_library: Path, out: Path, policies: dict,
                             point: int, authorization: str, *, prerequisite_protocol_ids: dict,
                             calibration_host: str, calibration_gpu_uuid: str) -> dict:
    """Write a v18_frontier.run_calibration-compatible bounded batch, CPU only."""
    from experiments.numerical_qk_reuse.runner import _fingerprint, _rows

    if type(point) is not int or not 0 <= point < MAX_THRESHOLD_PAIRS or not policies or set(policies) - set(CALIBRATION_ARMS):
        raise ValueError('invalid secondary threshold-pair point/arms')
    required = {'ruler4k_primary', 'aime26_primary'}
    if any(arm in CONTROLS for arm in policies):
        required.add('ruler_secondary70')
    if set(prerequisite_protocol_ids) != required or not calibration_host or not calibration_gpu_uuid:
        raise ValueError('frozen primary prerequisites and calibration host/GPU required')
    rows = _balanced(_rows(manifest, allow_task_budgets=True, allow_thinking_off=True), 2, 26)
    if [r['id'] for r in rows] != expected_ids:
        raise ValueError('calibration batch must use the exact independent frozen 26 IDs in order')
    clean = {arm: validate_policy(policy) for arm, policy in policies.items()}
    phase = 'v18_secondary_calibration_p' + str(point) + '_' + sha(json.dumps(
        [sha(manifest.read_bytes()), clean, sorted(policies), point], sort_keys=True))[:12]
    helper_path = Path(__file__).resolve()
    configs = {}
    for arm, policy in clean.items():
        base_arm = 'T60' if arm in CONTROLS else arm
        base = config_for(base_arm, manifest, model, policy_file, library, torch_library, phase, policy)
        if arm in CONTROLS:
            config = bind_control_config(base, arm, policy)
        else:
            config = dict(base)
            config['source_hashes'] = {**base['source_hashes'], str(helper_path): sha(helper_path.read_bytes())}
            config.pop('fingerprint', None)
            config['fingerprint'] = _fingerprint(config)
        configs[arm] = config
    hashes = {arm: arm_config_hash(config) for arm, config in configs.items()}
    schedule = plan_schedule(phase, REVISION, hashes, expected_ids, [CALIBRATION_SEED],
                             2026092606 + point, warm_repeats=0)
    protocol = dict(schema='v18_secondary_calibration_protocol_v1', protocol_id=phase,
                    status='frozen_for_explicit_GPU_run', authorization=authorization,
                    point_index=point, max_points_per_method_point=MAX_THRESHOLD_PAIRS,
                    selection_metric='minimum maximum absolute density error over whole/local/global only',
                    source_scope='native_legal_all_layers', ids=list(expected_ids), seeds=[CALIBRATION_SEED],
                    prerequisite_protocol_ids=prerequisite_protocol_ids,
                    calibration_host=calibration_host, calibration_gpu_uuid=calibration_gpu_uuid,
                    manifest=str(manifest.resolve()), manifest_sha256=sha(manifest.read_bytes()),
                    policy_file=str(policy_file.resolve()), model_path=str(model.resolve()),
                    model_revision=REVISION, arm_hashes=hashes,
                    arms={arm: dict(method=configs[arm]['method'], target=configs[arm]['target'],
                                    policy=clean[arm], allocation=configs[arm].get('allocation', 'normal'),
                                    allocation_seed=configs[arm].get('allocation_seed', 42))
                          for arm in configs},
                    schedule=schedule, planned_executions=len(schedule),
                    ceilings=dict(executions=len(schedule), single_gpu=True))
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f'secondary_calibration_p{point}_protocol.json'
    payload = json.dumps(protocol, indent=2, sort_keys=True) + '\n'
    if destination.exists() and destination.read_text() != payload:
        raise ValueError('secondary calibration protocol identity changed')
    destination.write_text(payload)
    for arm, config in configs.items():
        path = out / 'configs' / f'p{point}' / f'{arm}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        value = json.dumps(config, indent=2, sort_keys=True) + '\n'
        if path.exists() and path.read_text() != value:
            raise ValueError('secondary calibration config identity changed')
        path.write_text(value)
    return protocol


def advance_calibration_point(protocol_path: Path, ledger_path: Path, history_path: Path) -> dict:
    """Read routing counts from the existing batch driver, then select by density."""
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('schema') != 'v18_secondary_calibration_protocol_v1':
        raise ValueError('secondary calibration protocol required')
    actual, _counts = measured_density(protocol, ledger_path)
    history = json.loads(history_path.read_text()) if history_path.exists() else {arm: [] for arm in CALIBRATION_ARMS}
    updated, next_policies, chosen = dict(history), {}, {}
    point = protocol['point_index']
    for arm, spec in protocol['arms'].items():
        previous = history.get(arm, [])
        if len(previous) != point:
            raise ValueError('secondary calibration history point mismatch')
        state = update_density_history(arm, previous, spec['policy'], actual[arm])
        updated[arm] = state['history']
        chosen[arm] = dict(policy=state['selected_policy'], actual=state['history'][state['best_point']]['actual'],
                           points=len(state['history']), attained=state['attained'],
                           calibration_ids=protocol['ids'])
        if not state['done']:
            next_policies[arm] = state['next_policy']
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(updated, indent=2, sort_keys=True) + '\n'
    temp = history_path.with_suffix('.tmp')
    temp.write_text(payload)
    temp.replace(history_path)
    return dict(next_policies=next_policies, selected=chosen,
                finished=not next_policies, point_index=point)


def finalize_calibration(history_path: Path, calibration_ids: list[str], out: Path,
                         *, arms: tuple[str, ...] = CALIBRATION_ARMS) -> dict:
    """Freeze best measured density point for all four arms, attainable or not."""
    history = json.loads(history_path.read_text())
    if (not arms or set(arms) - set(CALIBRATION_ARMS) or
            len(calibration_ids) != 26 or len(set(calibration_ids)) != 26):
        raise ValueError('complete secondary history and same 26 IDs required')
    result = {}
    for arm in arms:
        entries = history[arm]
        if not 1 <= len(entries) <= MAX_THRESHOLD_PAIRS or any(
                set(item) != {'point', 'policy', 'actual'} or item['point'] != i or
                set(item['actual']) != {'whole', 'local', 'global'}
                for i, item in enumerate(entries)):
            raise ValueError('missing or nonconsecutive secondary density points')
        best = best_point(entries, TARGET[arm])
        attained = all(abs(best['actual'][kind] - TARGET[arm]) <= .02 for kind in ('whole', 'local', 'global'))
        if not attained and len(entries) < MAX_THRESHOLD_PAIRS:
            raise ValueError('unfinished secondary calibration point')
        result[arm] = dict(policy=best['policy'], actual=best['actual'], attained=attained,
                           points=len(entries), selected_point=best['point'],
                           calibration_ids=list(calibration_ids))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if out.exists() and out.read_text() != payload:
        raise ValueError('secondary frozen calibration identity changed')
    out.write_text(payload)
    return result


def bind_control_config(base_t60: dict, arm: str, calibrated_policy: dict) -> dict:
    """Bind a freshly generated T60 config to the frozen allocation semantics.

    The caller must generate the base with current code/source hashes (for
    example config_for('T60', ...)); stale primary configs are rejected.
    """
    from experiments.numerical_qk_reuse.runner import _fingerprint

    if arm not in CONTROLS or (base_t60.get('condition') != 'native_legal_all_layers' or
                               base_t60.get('frontier_arm') != 'T60' or base_t60.get('method') != 'T' or
                               base_t60.get('target') != 60 or base_t60.get('fast_t') is not True):
        raise ValueError('fresh named T60 base configuration required')
    from experiments.value_direction_hopper import frontier_scope
    if frontier_scope.CONTROL_SEED != ALLOCATION_SEED:
        raise ValueError('allocation seed differs from runtime')
    source_hashes = base_t60.get('source_hashes')
    scope_path = Path(frontier_scope.__file__).resolve()
    if not isinstance(source_hashes, dict) or source_hashes.get(str(scope_path)) != sha(scope_path.read_bytes()):
        raise ValueError('fresh control-capable scope source hash required')
    config = dict(base_t60)
    config['source_hashes'] = {**source_hashes, str(Path(__file__).resolve()): sha(Path(__file__).read_bytes())}
    config.update(frontier_arm=arm, allocation='shuffle' if arm.endswith('shuffled') else 'uniform',
                  allocation_seed=ALLOCATION_SEED, policy=validate_policy(calibrated_policy))
    config.pop('fingerprint', None)
    config['fingerprint'] = _fingerprint(config)
    return config


def secondary_plan(ruler_rows: list[dict], calibration_rows: list[dict], primary_draft: dict) -> dict:
    """Freeze logical question/seed/block order; no runnable configs are implied."""
    ruler = _balanced(ruler_rows, 10, 130)
    calibration = _balanced(calibration_rows, 2, 26)
    ids = [r['id'] for r in ruler]
    cal_ids = [r['id'] for r in calibration]
    if set(ids) & set(cal_ids) or {r['prompt_hash'] for r in ruler} & {r['prompt_hash'] for r in calibration}:
        raise ValueError('independent calibration/evaluation questions required')
    if primary_draft.get('dataset') != 'ruler4k' or primary_draft.get('ids') != ids or tuple(primary_draft.get('seeds', ())) != SEEDS:
        raise ValueError('secondary plan must extend the exact frozen primary RULER panel')
    warm_ids = primary_draft.get('warm_ids')
    if len(warm_ids) != 26 or len(set(warm_ids)) != 26 or not set(warm_ids) <= set(ids):
        raise ValueError('frozen 26-question timing subset required')
    mechanism_ids = mechanism_subset(ruler)
    core_specs = arm_specs(CORE)
    controls = {arm: dict(frontier_arm=arm, method='T', target=60, fast_t=True,
                          allocation='shuffle' if arm.endswith('shuffled') else 'uniform',
                          support_geometry='native_legal', rank=32, projection_seed=1729,
                          allocation_rng_seed=ALLOCATION_SEED,
                          allocation_semantics=('permute_within_physical_Q128' if arm.endswith('shuffled')
                                                else 'entire_current_canvas_mean_broadcast'),
                          canvas=256, max_denoising_steps=48)
                for arm in CONTROLS}
    def logical_hashes(specs):
        return {arm: sha(json.dumps(spec, sort_keys=True)) for arm, spec in specs.items()}
    phase70 = 'v18_ruler_secondary70_' + sha(json.dumps([ids, warm_ids, core_specs], sort_keys=True))[:12]
    schedule70 = [e for e in plan_schedule(phase70, REVISION, logical_hashes(core_specs), ids,
                                            list(SEEDS), 2026092604)
                  if e['role'] == 'attempt0' or e['id'] in warm_ids]
    for i, row in enumerate(schedule70):
        row['index'] = i
    phase_controls = 'v18_ruler_secondary_allocation_' + sha(json.dumps([mechanism_ids, controls], sort_keys=True))[:12]
    control_schedule = plan_schedule(phase_controls, REVISION, logical_hashes(controls), mechanism_ids,
                                     list(SEEDS), 2026092605, warm_repeats=0)
    assert len(schedule70) == 936 and len(control_schedule) == 156
    return dict(schema='v18_secondary_design_plan_v1', status='independent_calibration_pending',
                stage_order=['ruler_primary_initial', 'aime_primary', 'ruler_primary_remainder',
                             'ruler_secondary70', 'ruler_secondary_allocation'],
                generation_seeds=list(SEEDS), calibration_seed=CALIBRATION_SEED,
                calibration_ids=cal_ids, max_threshold_pairs_per_arm=MAX_THRESHOLD_PAIRS,
                calibration_selection='minimum maximum absolute error across whole/local/global achieved density; no quality/steps/timing',
                core70=dict(arms=core_specs, ids=ids, warm_ids=warm_ids, protocol_id=phase70,
                            schedule=schedule70, first_executions=780, warm_executions=156,
                            planned_executions=936),
                allocation=dict(arms=controls, ids=mechanism_ids, warm_ids=[], protocol_id=phase_controls,
                                schedule=control_schedule, first_executions=156, warm_executions=0,
                                planned_executions=156,
                                runnable=False,
                runtime_binding_available=True,
                blocker='independent same-26 density calibration not yet measured'))


def freeze_secondary_eval(stage: str, draft_path: Path, primary_protocol_path: Path,
                          aime_protocol_path: Path, calibration_manifest_path: Path,
                          secondary_calibration_path: Path, policy_file: Path, model: Path,
                          library: Path, torch_library: Path, out: Path,
                          preceding_secondary_path: Path | None = None) -> dict:
    """CPU-only secondary protocol/config freeze on one qualified host.

    The run wrapper below requires complete primary RULER and AIME evidence
    before touching a GPU. This protocol never modifies primary arm configs.
    """
    from experiments.numerical_qk_reuse.runner import _rows
    from scripts.v18_evaluate import logical_protocol_digest

    if stage not in ('ruler_secondary70', 'ruler_secondary_allocation'):
        raise ValueError('unknown secondary stage')
    draft = json.loads(draft_path.read_text())
    primary = json.loads(primary_protocol_path.read_text())
    aime = json.loads(aime_protocol_path.read_text())
    if (draft.get('dataset') != 'ruler4k' or primary.get('stage') != 'ruler4k_primary' or
            primary['ids'] != draft['ids'] or primary['model_revision'] != REVISION or
            logical_protocol_digest(primary) != primary['logical_protocol_sha256']):
        raise ValueError('qualified frozen primary RULER identity required')
    if aime.get('stage') != 'aime26_primary' or logical_protocol_digest(aime) != aime['logical_protocol_sha256']:
        raise ValueError('qualified frozen primary AIME identity required')
    preceding = None
    if stage == 'ruler_secondary_allocation':
        if preceding_secondary_path is None:
            raise ValueError('frozen 70% stage required before allocation stage')
        preceding = json.loads(preceding_secondary_path.read_text())
        if preceding.get('stage') != 'ruler_secondary70' or logical_protocol_digest(preceding) != preceding['logical_protocol_sha256']:
            raise ValueError('frozen 70% stage identity invalid')
    manifest = Path(draft['generation_manifest_path'])
    if (sha(manifest.read_bytes()) != draft['generation_manifest_sha256'] or
            draft['generation_manifest_sha256'] != primary['manifest_sha256']):
        raise ValueError('secondary generation manifest byte drift')
    rows = _balanced(_rows(manifest, allow_task_budgets=True, allow_thinking_off=True), 10, 130)
    cal_rows = _balanced(_rows(calibration_manifest_path, allow_task_budgets=True,
                               allow_thinking_off=True), 2, 26)
    plan = secondary_plan(rows, cal_rows, draft)
    return freeze_secondary_eval_with_plan(stage, draft, primary, aime, preceding, plan,
                                           secondary_calibration_path, policy_file, model,
                                           library, torch_library, out)


def freeze_secondary_eval_with_plan(stage: str, draft: dict, primary: dict, aime: dict,
                                    preceding: dict | None, plan: dict,
                                    secondary_calibration_path: Path, policy_file: Path, model: Path,
                                    library: Path, torch_library: Path, out: Path) -> dict:
    """Freeze a secondary stage using the already checked real-manifest design plan."""
    from experiments.numerical_qk_reuse.runner import _fingerprint
    from scripts.v18_bridge import inventory_identity
    from scripts.v18_evaluate import canonical_arm_hash, eval_config, logical_protocol_digest

    if stage not in ('ruler_secondary70', 'ruler_secondary_allocation') or plan['schema'] != 'v18_secondary_design_plan_v1':
        raise ValueError('frozen secondary design plan required')
    branch = plan['core70'] if stage == 'ruler_secondary70' else plan['allocation']
    arms = tuple(branch['arms'])
    frozen = json.loads(secondary_calibration_path.read_text())
    if any(arm not in frozen or not 1 <= frozen[arm]['points'] <= MAX_THRESHOLD_PAIRS or
           frozen[arm]['calibration_ids'] != plan['calibration_ids'] for arm in arms):
        raise ValueError('independent same-26 calibration required for all stage arms')
    if set(plan['calibration_ids']) & set(branch['ids']):
        raise ValueError('calibration/evaluation overlap')
    inventory = inventory_identity(primary['tensor_inventory']['path'])
    if inventory['tensor_identity_sha256'] != primary['model_tensor_identity_sha256']:
        raise ValueError('primary model tensor identity drift')
    if (str(model.resolve()) != primary['model_path'] or str(policy_file.resolve()) != primary['policy_file'] or
            sha(library.read_bytes()) != primary['binary_hashes']['kernel'] or
            sha(torch_library.read_bytes()) != primary['binary_hashes']['bridge']):
        raise ValueError('secondary host model/policy/binary identity differs from qualified primary')
    manifest = Path(draft['generation_manifest_path'])
    phase = 'v18_' + stage + '_' + sha(json.dumps([draft['generation_manifest_sha256'],
            sha(secondary_calibration_path.read_bytes()), primary['logical_protocol_sha256'],
            branch['protocol_id']], sort_keys=True))[:12]
    helper = Path(__file__).resolve()
    configs = {}
    primary_frozen = json.loads(Path(primary['calibration_file']).read_text())
    for arm in arms:
        if arm in CORE:
            config = eval_config(arm, draft, manifest, policy_file, model, library, torch_library, frozen, phase)
            config['source_hashes'] = {**config['source_hashes'], str(helper): sha(helper.read_bytes())}
            config.pop('fingerprint', None)
            config['fingerprint'] = _fingerprint(config)
        else:
            base = eval_config('T60', draft, manifest, policy_file, model, library, torch_library,
                               primary_frozen, phase)
            config = bind_control_config(base, arm, frozen[arm]['policy'])
        configs[arm] = config
    binaries = primary['binary_hashes']
    canonical = {arm: sha(json.dumps([canonical_arm_hash(configs[arm], binaries,
                       inventory['tensor_identity_sha256']), configs[arm].get('allocation', 'normal'),
                       configs[arm].get('allocation_seed', 42)], sort_keys=True)) for arm in arms}
    execution = {arm: arm_config_hash(configs[arm]) for arm in arms}
    schedule = plan_schedule(phase, REVISION, canonical, branch['ids'], list(SEEDS),
                             2026092604 if stage == 'ruler_secondary70' else 2026092605,
                             warm_repeats=1 if stage == 'ruler_secondary70' else 0)
    if stage == 'ruler_secondary70':
        schedule = [row for row in schedule if row['role'] == 'attempt0' or row['id'] in branch['warm_ids']]
    for i, row in enumerate(schedule):
        row['index'] = i
    if len(schedule) != branch['planned_executions']:
        raise ValueError('secondary execution count drift')
    hosts = primary['assigned_hosts']
    blocks = len({row['block'] for row in schedule})
    assignments = inherit_block_assignments(primary, schedule)
    if len(assignments) != blocks:
        raise ValueError('secondary block assignment incomplete')
    protocol = dict(schema='v18_secondary_eval_protocol_v1', status='frozen', protocol_id=phase,
                    stage=stage, priority=4 if stage == 'ruler_secondary70' else 5,
                    prerequisite_stages=['ruler4k_primary', 'aime26_primary'],
                    prerequisite_protocol_ids={'ruler4k_primary': primary['protocol_id'],
                                               'aime26_primary': aime['protocol_id'],
                                               **({'ruler_secondary70': preceding['protocol_id']}
                                                  if preceding is not None else {})},
                    stage_order=plan['stage_order'], authorization=primary['authorization'],
                    ids=branch['ids'], seeds=list(SEEDS), arms=list(arms), arm_hashes=canonical,
                    execution_config_hashes=execution, model_path=str(model.resolve()),
                    model_revision=REVISION, tensor_inventory=inventory,
                    model_tensor_identity_sha256=inventory['tensor_identity_sha256'],
                    manifest=str(manifest.resolve()), manifest_sha256=sha(manifest.read_bytes()),
                    prompt_hashes={r['id']: r['prompt_hash'] for r in _balanced(read_rows(manifest), 10, 130)},
                    source_manifest_sha256=draft['source_manifest_sha256'],
                    calibration_file=str(secondary_calibration_path.resolve()),
                    calibration_sha256=sha(secondary_calibration_path.read_bytes()),
                    calibrated_points={arm: frozen[arm] for arm in arms},
                    policy_file=str(policy_file.resolve()), policy_file_sha256=sha(policy_file.read_bytes()),
                    source_hashes=configs[arms[0]]['source_hashes'], binary_hashes=binaries,
                    assigned_hosts=hosts, bridge_receipt_file=primary['bridge_receipt_file'],
                    bridge_receipt_sha256=primary['bridge_receipt_sha256'],
                    block_assignments=assignments,
                    block_assignment_rule='inherit exact host/GPU of matching frozen primary RULER (id,seed) block',
                    initial_prefix_blocks=blocks, warm_ids=branch['warm_ids'],
                    schedule=schedule, planned_executions=len(schedule), planned_blocks=blocks,
                    first_executions=branch['first_executions'], warm_executions=branch['warm_executions'],
                    timing='strict paired warm only on 26 frozen subset' if stage == 'ruler_secondary70'
                           else 'first quality outputs only; no warm latency claim')
    protocol['logical_protocol_sha256'] = logical_protocol_digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f'{stage}_protocol.json'
    payload = json.dumps(protocol, indent=2, sort_keys=True) + '\n'
    if destination.exists() and destination.read_text() != payload:
        raise ValueError('secondary eval protocol identity changed')
    destination.write_text(payload)
    for arm, config in configs.items():
        path = out / 'configs' / stage / f'{arm}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(config, indent=2, sort_keys=True) + '\n'
        if path.exists() and path.read_text() != data:
            raise ValueError('secondary eval config identity changed')
        path.write_text(data)
    return protocol


def inherit_block_assignments(primary: dict, schedule: list[dict]) -> dict:
    """Keep every secondary (id,seed) on its exact primary host/GPU."""
    primary_hosts = {}
    for entry in primary['schedule']:
        key = (entry['id'], entry['seed'])
        assigned = primary['block_assignments'][str(entry['block'])]
        if key in primary_hosts and primary_hosts[key] != assigned:
            raise ValueError('primary paired block spans different host/GPU assignments')
        primary_hosts[key] = assigned
    assignments = {}
    for entry in schedule:
        key = (entry['id'], entry['seed'])
        if key not in primary_hosts:
            raise ValueError('secondary cell absent from primary RULER pairing')
        assigned = primary_hosts[key]
        block = str(entry['block'])
        if block in assignments and assignments[block] != assigned:
            raise ValueError('secondary paired block spans different host/GPU assignments')
        assignments[block] = assigned
    return assignments


def stage_completion(protocol: dict, ledger_paths: list[Path]) -> dict:
    """Require every scheduled execution, including failed first outputs, once."""
    from scripts.v18_evaluate import logical_protocol_digest

    if (protocol.get('status') != 'frozen' or
            logical_protocol_digest(protocol) != protocol.get('logical_protocol_sha256')):
        raise ValueError('prerequisite frozen logical protocol drift')
    expected = {execution_key(entry): entry for entry in protocol['schedule']}
    if len(expected) != len(protocol['schedule']):
        raise ValueError('duplicate prerequisite schedule key')
    seen = set()
    for path in ledger_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') != 'run':
                continue
            key = event.get('execution_key')
            if key not in expected or key in seen:
                raise ValueError('foreign or duplicate prerequisite execution')
            planned = expected[key]
            if any(event.get(field) != planned[field] for field in ('arm', 'id', 'seed', 'role', 'repeat', 'cell_id')):
                raise ValueError('prerequisite run identity mismatch')
            assigned = protocol['block_assignments'][str(planned['block'])]
            if (event.get('host'), event.get('gpu_uuid')) != (assigned['host'], assigned['gpu_uuid']):
                raise ValueError('prerequisite host/GPU assignment mismatch')
            seen.add(key)
    missing = [dict(arm=e['arm'], id=e['id'], seed=e['seed'], role=e['role'],
                    repeat=e['repeat'], cell_id=e['cell_id'])
               for e in protocol['schedule'] if execution_key(e) not in seen]
    return dict(stage=protocol['stage'], protocol_id=protocol['protocol_id'],
                planned=len(expected), completed=len(seen), missing=missing, complete=not missing)


def prerequisite_gate(secondary: dict, primary_ruler: tuple[dict, list[Path]],
                      primary_aime: tuple[dict, list[Path]],
                      secondary70: tuple[dict, list[Path]] | None = None) -> dict:
    """Pure CPU hard gate used before a secondary worker can load a GPU model."""
    if secondary['stage'] not in ('ruler_secondary70', 'ruler_secondary_allocation'):
        raise ValueError('secondary stage required')
    if primary_ruler[0].get('stage') != 'ruler4k_primary' or primary_aime[0].get('stage') != 'aime26_primary':
        raise ValueError('frozen primary RULER and AIME protocols required')
    expected = secondary.get('prerequisite_protocol_ids', {})
    if (expected.get('ruler4k_primary') != primary_ruler[0].get('protocol_id') or
            expected.get('aime26_primary') != primary_aime[0].get('protocol_id')):
        raise ValueError('primary prerequisite protocol ID differs from frozen secondary stage')
    checks = [stage_completion(*primary_ruler), stage_completion(*primary_aime)]
    if secondary['stage'] == 'ruler_secondary_allocation':
        if secondary70 is None or secondary70[0].get('stage') != 'ruler_secondary70':
            raise ValueError('70% secondary stage required before allocation controls')
        if expected.get('ruler_secondary70') != secondary70[0].get('protocol_id'):
            raise ValueError('70% prerequisite protocol ID differs from frozen allocation stage')
        checks.append(stage_completion(*secondary70))
    return dict(ready=all(item['complete'] for item in checks), checks=checks,
                missing=[dict(stage=item['stage'], **row) for item in checks for row in item['missing']])


def run_secondary_calibration(protocol_path: Path, prerequisite_paths: dict,
                              private: Path, ledger: Path, lock_path: Path, *,
                              max_executions: int, deadline_epoch: float, timeout: int) -> str:
    """Use the existing v18 batch driver only after complete primary ledgers."""
    import socket
    import subprocess
    from scripts.v18_evaluate import logical_protocol_digest

    protocol = json.loads(protocol_path.read_text())
    if protocol.get('schema') != 'v18_secondary_calibration_protocol_v1':
        raise ValueError('secondary calibration protocol required')
    expected = protocol['prerequisite_protocol_ids']
    if set(prerequisite_paths) != set(expected):
        raise ValueError('complete primary prerequisite paths required')
    checks = []
    for stage, (path, ledgers) in prerequisite_paths.items():
        primary = json.loads(Path(path).read_text())
        if primary.get('stage') != stage or primary.get('protocol_id') != expected[stage]:
            raise ValueError('secondary calibration prerequisite protocol identity drift')
        if logical_protocol_digest(primary) != primary['logical_protocol_sha256']:
            raise ValueError('secondary calibration prerequisite logical drift')
        checks.append(stage_completion(primary, [Path(x) for x in ledgers]))
    gate = dict(ready=all(item['complete'] for item in checks), checks=checks,
                missing=[dict(stage=item['stage'], **row) for item in checks for row in item['missing']])
    private.mkdir(parents=True, exist_ok=True)
    (private / 'prerequisite_gate.json').write_text(json.dumps(gate, indent=2, sort_keys=True) + '\n')
    if not gate['ready']:
        return 'prerequisite_incomplete'
    if socket.gethostname() != protocol['calibration_host']:
        raise ValueError('secondary calibration assigned to a different host')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).splitlines()[0].strip()
    if uuid != protocol['calibration_gpu_uuid']:
        raise ValueError('secondary calibration GPU UUID drift')
    from scripts.v18_frontier import run_calibration
    return run_calibration(protocol_path, private, ledger, lock_path,
                           max_executions, deadline_epoch, timeout)


def run_secondary_eval(protocol_path: Path, prerequisite_paths: dict,
                       private: Path, ledger: Path, lock_path: Path, *, max_blocks: int,
                       deadline_epoch: float, gpu_budget_s: float, remaining_requests: int,
                       block_guard_s: float, timeout: int) -> str:
    """Explicit resumable secondary worker; gate all primary work before GPU load."""
    import os
    import signal
    import socket
    import subprocess
    import time
    import fcntl

    from scripts.v18_evaluate import (canonical_arm_hash, logical_protocol_digest,
                                      native_phase_evidence, run_complete_blocks, strict_warm)
    from scripts.v13_seed_runs import (is_device_error, receipt_path, run_schedule,
                                       validate_resume)

    protocol = json.loads(protocol_path.read_text())
    if (protocol.get('schema') != 'v18_secondary_eval_protocol_v1' or
            protocol.get('status') != 'frozen' or
            logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']):
        raise ValueError('secondary protocol identity drift')
    required = protocol['prerequisite_protocol_ids']
    if set(prerequisite_paths) != set(required):
        raise ValueError('all frozen prerequisite protocol/ledger sets required')
    prior = {}
    for stage, (path, ledger_paths) in prerequisite_paths.items():
        item = json.loads(Path(path).read_text())
        if item.get('stage') != stage or item.get('protocol_id') != required[stage]:
            raise ValueError('prerequisite protocol identity mismatch')
        if logical_protocol_digest(item) != item['logical_protocol_sha256']:
            raise ValueError('prerequisite logical protocol drift')
        prior[stage] = (item, [Path(p) for p in ledger_paths])
    gate = prerequisite_gate(protocol, prior['ruler4k_primary'], prior['aime26_primary'],
                             prior.get('ruler_secondary70'))
    private.mkdir(parents=True, exist_ok=True)
    gate_path = private / 'prerequisite_gate.json'
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + '\n')
    if not gate['ready']:
        return 'prerequisite_incomplete'

    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _atomic, _one, _rows
    from scripts.v15_seed_runs import mapped_shared_objects
    from scripts.v18_bridge import inventory_identity
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import import_identity
    from triton.runtime.jit import JITFunction

    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).splitlines()[0].strip()
    worker = dict(host=socket.gethostname(), gpu_uuid=uuid)
    if worker not in protocol['assigned_hosts']:
        raise ValueError('host/GPU not assigned to frozen secondary stage')
    configs = {arm: json.loads((protocol_path.parent / 'configs' / protocol['stage'] / f'{arm}.json').read_text())
               for arm in protocol['arms']}
    if {arm: arm_config_hash(config) for arm, config in configs.items()} != protocol['execution_config_hashes']:
        raise ValueError('secondary execution config hash drift')
    inventory = inventory_identity(protocol['tensor_inventory']['path'])
    if inventory != protocol['tensor_inventory']:
        raise ValueError('secondary model tensor inventory drift')
    canonical = {arm: sha(json.dumps([canonical_arm_hash(config, protocol['binary_hashes'],
                       inventory['tensor_identity_sha256']), config.get('allocation', 'normal'),
                       config.get('allocation_seed', 42)], sort_keys=True)) for arm, config in configs.items()}
    if canonical != protocol['arm_hashes']:
        raise ValueError('secondary mathematical arm identity drift')
    for filename, expected in ((protocol['manifest'], protocol['manifest_sha256']),
                               (protocol['calibration_file'], protocol['calibration_sha256']),
                               (protocol['policy_file'], protocol['policy_file_sha256'])):
        if sha(Path(filename).read_bytes()) != expected:
            raise ValueError('secondary manifest/calibration/policy byte drift')
    if protocol['bridge_receipt_file'] and sha(Path(protocol['bridge_receipt_file']).read_bytes()) != protocol['bridge_receipt_sha256']:
        raise ValueError('secondary cross-host bridge receipt drift')
    for config in configs.values():
        if any(not Path(path).is_file() or sha(Path(path).read_bytes()) != expected
               for path, expected in config['source_hashes'].items()):
            raise ValueError('secondary source/binary hash drift')
        if any(sha((Path(config['model']) / name).read_bytes()) != expected
               for name, expected in config['model_metadata_hashes'].items()):
            raise ValueError('secondary model metadata drift')
    manifest = Path(protocol['manifest'])
    rows = _rows(manifest, allow_task_budgets=True, allow_thinking_off=True)
    if len(rows) != 130 or any(k in row for row in rows for k in ('outputs', 'answer', 'expected',
                                                                  'expected_answer', 'gold')):
        raise ValueError('gold-free frozen RULER panel required')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != 130 or not set(protocol['ids']) <= set(by_id):
        raise ValueError('secondary question coverage drift')
    if any(sha(row['prompt']) != protocol['prompt_hashes'][row['id']] for row in rows):
        raise ValueError('secondary prompt hash drift')
    identity = dict(protocol_id=protocol['protocol_id'], model_revision=REVISION,
                    arm_hashes=protocol['arm_hashes'], source_hashes=configs[protocol['arms'][0]]['source_hashes'],
                    private_root=str(private.resolve()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        assigned = [entry for entry in protocol['schedule']
                    if protocol['block_assignments'][str(entry['block'])] == worker]
        done = validate_resume(events, identity, assigned)
        for entry in assigned:
            if execution_key(entry) not in done and receipt_path(private, entry).exists():
                raise ValueError('orphan secondary receipt; preserve first output')
        if len(done) == len(assigned):
            return 'complete_prefix'
        append(ledger, dict(event='start', pid=os.getpid(), pgid=os.getpgid(0), when=time.time(),
                            host=socket.gethostname(), gpu_uuid=uuid, import_identity=import_identity(), **identity))
        started = time.perf_counter()
        adapter = create_adapter('diffusion_gemma', protocol['model_path'], device='cuda',
                                 precision='bfloat16', revision=REVISION).load()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        compiles = []
        JITFunction.cache_hook = lambda **kw: compiles.append(time.perf_counter()) or False

        class Timeout(Exception):
            pass
        def alarm(*_):
            raise Timeout()
        signal.signal(signal.SIGALRM, alarm)
        first = {e['cell_id']: e for e in events if e.get('event') == 'run' and e.get('role') == 'attempt0'}

        def execute_one(row, seed, config, entry):
            n, cache_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
            receipt, error = None, None
            outer = time.perf_counter()
            signal.alarm(timeout)
            try:
                receipt = _one(adapter, row, seed, config)
            except Timeout:
                error = f'timeout>{timeout}s'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:500]
            finally:
                signal.alarm(0)
            record = dict(ok=receipt is not None, error=error, generation_seed=seed,
                          arm_config_hash=protocol['arm_hashes'][entry['arm']],
                          triton_misses=len(compiles) - n,
                          triton_disk_entries_added=disk_cache_entries() - cache_before,
                          new_shared_objects=sorted(mapped_shared_objects() - so_before),
                          outer_wall_s=time.perf_counter() - outer, host=socket.gethostname(), gpu_uuid=uuid)
            if receipt is not None:
                if receipt['seed'] != seed:
                    raise AssertionError('secondary generation seed mismatch')
                record.update(redacted(receipt))
                record['phase_evidence'] = native_phase_evidence(receipt, sparse=True)
            if entry['role'] == 'warm':
                record['acceptance'] = strict_warm(first.get(entry['cell_id']), record)
                if receipt is not None:
                    path = receipt_path(private, entry)
                    _atomic(path, receipt)
                    record['private_receipt'] = str(path)
            else:
                first[entry['cell_id']] = record
            return dict(record=record, receipt=receipt, fatal=bool(error and is_device_error(error)))

        try:
            blocks = list(dict.fromkeys(entry['block'] for entry in assigned))
            if max_blocks is not None:
                blocks = blocks[:max_blocks]
            selected = [entry for entry in assigned if entry['block'] in set(blocks)]
            def execute_block(entries):
                return run_schedule(entries, done=done, execute_one=execute_one,
                                    rows=by_id, configs=configs,
                                    ledger_append=lambda record: append(ledger, record),
                                    private=private, save_receipt=_atomic)
            status = run_complete_blocks(selected, done, deadline_epoch=deadline_epoch,
                                         gpu_budget_s=gpu_budget_s, remaining_requests=remaining_requests,
                                         block_guard_s=block_guard_s, process_started=started,
                                         execute_block=execute_block)
        finally:
            JITFunction.cache_hook = None
        append(ledger, dict(event='worker_end', status=status, when=time.time(),
                            host=socket.gethostname(), gpu_uuid=uuid,
                            gpu_process_seconds=time.perf_counter() - started))
    return status


def secondary_scorer_identity(protocol: dict, ruler_root: Path) -> dict:
    """Lock the same pinned official RULER scorer for a secondary protocol."""
    from scripts.v18_summarize import scorer_source_identity

    if protocol.get('stage') not in ('ruler_secondary70', 'ruler_secondary_allocation'):
        raise ValueError('secondary RULER protocol required')
    result = scorer_source_identity({**protocol, 'stage': 'ruler4k_primary'}, ruler_root)
    result['secondary_adapter_sha256'] = sha(Path(__file__).read_bytes())
    return result


def freeze_secondary_scorer_lock(protocol_path: Path, ruler_root: Path, out: Path) -> dict:
    from scripts.v18_evaluate import logical_protocol_digest, write_immutable_json

    protocol = json.loads(protocol_path.read_text())
    if logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']:
        raise ValueError('secondary scorer protocol identity drift')
    identity = secondary_scorer_identity(protocol, ruler_root)
    write_immutable_json(out, identity)
    return identity


def secondary_cells(protocol: dict, ledger_paths: list[Path], gold_by_id: dict,
                    ruler_scorers, private_roots: dict[str, Path]) -> tuple[dict, set[int], int]:
    """Score private first receipts; retain redacted cells only in memory/output."""
    from scripts.v18_evaluate import strict_warm
    from scripts.v18_summarize import load_records, routing_counts, score_one

    records = load_records(protocol, ledger_paths)
    unfinished = {e['block'] for e in protocol['schedule'] if execution_key(e) not in records}
    cells = {}
    for e in protocol['schedule']:
        key = (e['arm'], e['id'], e['seed'])
        cell = cells.setdefault(key, dict(arm=e['arm'], id=e['id'], seed=e['seed'], block=e['block'],
                                          attempt0=None, warm=None, quality=None, warm_s=None, work=None))
        cell['attempt0' if e['role'] == 'attempt0' else 'warm'] = records.get(execution_key(e))
    for cell in cells.values():
        first, warm = cell['attempt0'], cell['warm']
        if first is not None:
            if first.get('ok'):
                path = Path(first['private_receipt'])
                if first['host'] in private_roots:
                    if path.name != 'attempt00.json':
                        raise ValueError('first secondary receipt basename drift')
                    path = private_roots[first['host']] / 'cells' / first['cell_id'] / path.name
                receipt = json.loads(path.read_text())
                if (receipt['id'] != cell['id'] or receipt['seed'] != cell['seed'] or
                        receipt['prompt_token_hash'] != first['prompt_token_hash'] or
                        sha(json.dumps(receipt['completion_tokens'], separators=(',', ':'))) != first['completion_token_hash'] or
                        receipt['termination_reason'] != first['termination'] or
                        [c['decoder_calls'] for c in receipt['per_canvas']] != first['per_canvas_calls']):
                    raise ValueError('secondary receipt/ledger identity mismatch')
                cell['quality'] = score_one('ruler4k', gold_by_id[cell['id']], receipt, first,
                                            ruler_scorers=ruler_scorers)
                cell['work'] = dict(decoder_calls=receipt['total_decoder_calls'],
                                    canvases=len(receipt['per_canvas']),
                                    native_stop_canvases=sum(c['native_stop_final_call'] for c in receipt['per_canvas']),
                                    routing=routing_counts(receipt))
            else:
                cell['quality'] = dict(score=0., correct=False, parsed=False, capped=False, eos=False, failed=True)
        if warm is not None:
            acceptance = strict_warm(first, warm)
            if warm.get('acceptance') != acceptance:
                raise ValueError('secondary stored warm acceptance drift')
            if acceptance['accepted']:
                if type(warm.get('api_wall_s')) not in (int, float) or warm['api_wall_s'] <= 0:
                    raise ValueError('secondary accepted warm missing request wall')
                cell['warm_s'] = warm['api_wall_s']
    return cells, unfinished, len(records)


def summarize_secondary(protocol_path: Path, ledger_paths: list[Path], primary_protocol_path: Path,
                        primary_ledgers: list[Path], gold_path: Path, scorer_lock: Path,
                        ruler_root: Path, *, private_roots: dict[str, Path],
                        primary_private_roots: dict[str, Path], resamples: int = 10000) -> dict:
    """Offline gold-separated secondary summary and paired complete-block comparisons."""
    from collections import Counter
    from statistics import mean
    from dllm.evaluation.ruler import official
    from scripts.v18_evaluate import logical_protocol_digest
    from scripts.v18_summarize import bootstrap_question_clusters, read_rows

    protocol = json.loads(protocol_path.read_text())
    primary = json.loads(primary_protocol_path.read_text())
    if (logical_protocol_digest(protocol) != protocol['logical_protocol_sha256'] or
            logical_protocol_digest(primary) != primary['logical_protocol_sha256'] or
            primary['protocol_id'] != protocol['prerequisite_protocol_ids']['ruler4k_primary']):
        raise ValueError('frozen primary/secondary scoring identity drift')
    if not stage_completion(primary, primary_ledgers)['complete']:
        raise ValueError('primary RULER prerequisite incomplete for secondary scoring')
    if json.loads(scorer_lock.read_text()) != secondary_scorer_identity(protocol, ruler_root):
        raise ValueError('secondary pinned scorer source lock drift')
    if sha(gold_path.read_bytes()) != protocol['source_manifest_sha256']:
        raise ValueError('secondary scorer-only gold source byte drift')
    gold = read_rows(gold_path)
    by_id = {row['id']: row for row in gold}
    if len(by_id) != len(gold) or set(primary['ids']) - set(by_id):
        raise ValueError('secondary scorer gold ID coverage mismatch')
    ruler_scorers = official.load_scorers(ruler_root)
    cells, unfinished, recorded = secondary_cells(protocol, ledger_paths, by_id,
                                                  ruler_scorers, private_roots)
    primary_cells, primary_unfinished, _ = secondary_cells(primary, primary_ledgers, by_id,
                                                           ruler_scorers, primary_private_roots)
    scored = {key: cell for key, cell in cells.items() if cell['block'] not in unfinished}
    scored.update({key: cell for key, cell in primary_cells.items()
                   if cell['block'] not in primary_unfinished and key[0] in ('T60', 'U60')})
    task_by_id = {rid: by_id[rid]['task'] for rid in protocol['ids']}
    comparisons = ((('T70', 'U70'), ('T70', 'T60'), ('U70', 'U60'))
                   if protocol['stage'] == 'ruler_secondary70' else
                   (('T60_shuffled', 'T60'), ('T60_uniform', 'T60'),
                    ('T60_shuffled', 'T60_uniform')))
    pairs = {f'{a}/{b}': bootstrap_question_clusters(scored, a, b, task_by_id=task_by_id,
                                                      resamples=resamples)
             for a, b in comparisons}
    arms = {}
    for arm in protocol['arms']:
        group = [cell for cell in cells.values() if cell['arm'] == arm]
        routing = {kind: Counter() for kind in ('whole', 'local', 'global')}
        for cell in group:
            if cell['work'] and cell['work']['routing']:
                for kind, counts in cell['work']['routing'].items():
                    routing[kind].update(counts)
        scores = [cell['quality']['score'] for cell in group if cell['quality'] is not None]
        arms[arm] = dict(scheduled_cells=len(group), attempted=sum(cell['attempt0'] is not None for cell in group),
                         failures=sum(cell['attempt0'] is not None and not cell['attempt0'].get('ok') for cell in group),
                         mean_score=mean(scores) if scores else None,
                         correct=sum(bool(cell['quality'] and cell['quality']['correct']) for cell in group),
                         decoder_calls=sum(cell['work']['decoder_calls'] for cell in group if cell['work']),
                         strict_warm_accepted=sum(cell['warm_s'] is not None for cell in group),
                         pv_bitmap={kind: dict(counts) for kind, counts in routing.items()})
    return dict(schema='v18_secondary_redacted_summary_v1', stage=protocol['stage'],
                protocol_id=protocol['protocol_id'], scorer_source_identity=secondary_scorer_identity(protocol, ruler_root),
                planned_executions=protocol['planned_executions'], recorded_executions=recorded,
                complete=recorded == protocol['planned_executions'],
                unfinished_blocks=sorted(unfinished), arms=arms, pairs=pairs,
                timing_note=('U70/T70 strict warm is same-stage; T70/T60 and U70/U60 compare same-GPU '
                             'receipts across primary and secondary stages separated by AIME/time, so '
                             'these are descriptive with possible temporal drift. Allocation controls '
                             'have no warm latency claim.'),
                pairing_note='Quality/calls pairs use complete logical blocks and inherited primary host/GPU; bootstrap by question/task')


def main() -> None:
    """Explicit GPU runner entrypoint; importing this module remains CPU only."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run-eval', 'run-calibration'))
    for name in ('protocol', 'prerequisites', 'private', 'ledger', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--max-executions', type=int)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--gpu-budget-s', type=float)
    parser.add_argument('--remaining-requests', type=int)
    parser.add_argument('--block-guard-s', type=float)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    raw = json.loads(args.prerequisites.read_text())
    paths = {stage: (Path(value['protocol']), [Path(x) for x in value['ledgers']])
             for stage, value in raw.items()}
    if args.command == 'run-calibration':
        if args.max_executions is None:
            parser.error('--max-executions required for calibration')
        status = run_secondary_calibration(args.protocol, paths, args.private, args.ledger,
                                           args.lock, max_executions=args.max_executions,
                                           deadline_epoch=args.deadline_epoch, timeout=args.timeout)
    else:
        if None in (args.max_blocks, args.gpu_budget_s, args.remaining_requests, args.block_guard_s):
            parser.error('eval budget/blocks arguments required')
        status = run_secondary_eval(args.protocol, paths, args.private, args.ledger, args.lock,
                                    max_blocks=args.max_blocks, deadline_epoch=args.deadline_epoch,
                                    gpu_budget_s=args.gpu_budget_s,
                                    remaining_requests=args.remaining_requests,
                                    block_guard_s=args.block_guard_s, timeout=args.timeout)
    print(json.dumps({'status': status}))


def authoring_main() -> None:
    """CPU-only freeze and offline scoring commands for the stage supervisor."""
    import argparse
    import sys
    from scripts.v18_evaluate import write_immutable_json

    command = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    if command == 'freeze-calibration-spec':
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--out', type=Path, required=True)
        a = parser.parse_args()
        result = freeze_calibration_spec(a.input, a.out)
        print(json.dumps(dict(group=result['group'], ids=len(result['ids']),
                              spec_sha256=sha(a.out.read_bytes()))))
    elif command == 'freeze-eval':
        parser.add_argument('--stage', choices=('ruler_secondary70', 'ruler_secondary_allocation'), required=True)
        for name in ('draft', 'primary-protocol', 'aime-protocol', 'calibration-manifest',
                     'secondary-calibration', 'policy-file', 'model', 'library',
                     'torch-library', 'out'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--preceding-secondary', type=Path)
        a = parser.parse_args()
        result = freeze_secondary_eval(a.stage, a.draft, a.primary_protocol,
                                       a.aime_protocol, a.calibration_manifest,
                                       a.secondary_calibration, a.policy_file, a.model,
                                       a.library, a.torch_library, a.out,
                                       preceding_secondary_path=a.preceding_secondary)
        print(json.dumps(dict(stage=result['stage'], protocol_id=result['protocol_id'],
                              planned_executions=result['planned_executions'])))
    elif command == 'freeze-scorer-lock':
        for name in ('protocol', 'ruler-root', 'out'):
            parser.add_argument('--' + name, type=Path, required=True)
        a = parser.parse_args()
        identity = freeze_secondary_scorer_lock(a.protocol, a.ruler_root, a.out)
        print(json.dumps(dict(scorer_lock_sha256=sha(a.out.read_bytes()),
                              stage=identity.get('stage'))))
    elif command == 'score':
        for name in ('protocol', 'primary-protocol', 'gold', 'scorer-lock', 'ruler-root', 'out'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--ledger', type=Path, action='append', required=True)
        parser.add_argument('--primary-ledger', type=Path, action='append', required=True)
        parser.add_argument('--private-root', action='append', default=[], metavar='HOST=PATH')
        parser.add_argument('--primary-private-root', action='append', default=[], metavar='HOST=PATH')
        parser.add_argument('--resamples', type=int, default=10000)
        a = parser.parse_args()
        def roots(values):
            result = {}
            for item in values:
                host, sep, path = item.partition('=')
                if not sep or not host or not path or host in result:
                    raise ValueError('private roots must be unique HOST=PATH')
                result[host] = Path(path)
            return result
        summary = summarize_secondary(a.protocol, a.ledger, a.primary_protocol,
                                      a.primary_ledger, a.gold, a.scorer_lock,
                                      a.ruler_root, private_roots=roots(a.private_root),
                                      primary_private_roots=roots(a.primary_private_root),
                                      resamples=a.resamples)
        write_immutable_json(a.out, summary)
        print(json.dumps(dict(stage=summary['stage'], complete=summary['complete'],
                              recorded_executions=summary['recorded_executions'])))
    else:
        parser.error('unknown CPU authoring command')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ('freeze-calibration-spec', 'freeze-eval',
                                            'freeze-scorer-lock', 'score'):
        authoring_main()
    else:
        main()
