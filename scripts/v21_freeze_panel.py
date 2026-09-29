"""CPU-only freeze and host binding for a conditional v21 panel.

No prompt is constructed, no model is loaded, and this module never launches a
generation worker. Freeze uses the original v20 ordered development IDs and
gold-free manifest rows; host bind rehashes the old P0 runtime on that host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.numerical_qk_reuse import v20
from scripts import v20_bind as old
from scripts.v13_seed_runs import cell_id
from scripts.v21_run import validate_protocol


NUMERIC_ARMS = ('D_native', 'D_matched_legacy', 'D_matched_new', 'T_scope',
                'M3_R3_A8_legacy', 'M3_R3_A8_new', 'B_A8_matched_new')
LAYOUT_ARMS = ('M3_R3_A8_legacy', 'M3_R3_A8_layout')
# v23 Track P: six frozen arms; every method arm shares the selected output
# contract and the exact grouped-Q observation producer (bitwise-qualified).
BOOTSTRAP_ARMS = ('D_native', 'T_scope', 'M3_R3_A8_incumbent',
                  'M1_native_bootstrap2_observe1', 'M3_native_bootstrap2_observe1',
                  'B_native_bootstrap2_observe1')
BOOTSTRAP = 'native_bootstrap2_observe1'
PRODUCER = "grouped_q"

SCOPE = v20.GLOBAL_ONLY_NATIVE_LOCAL


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                       ensure_ascii=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


PILOT_ARMS = ('D_native', 'T_scope', 'M3_boot_logical', 'M1_boot_aligned16',
              'M3_boot_aligned16', 'B_boot_aligned16')


def freeze_pilot(v20_protocol_path, v20_binding_path, manifests_dir, out_dir):
    """v25 aligned16 pilot: inputs chosen by sha256(id) order from the existing
    frozen inventories before any v25 output; host counterbalanced by question
    and seed; RULER seed 101 with warm on the first selected task only."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    original_bytes, binding_bytes = Path(v20_protocol_path).read_bytes(), Path(v20_binding_path).read_bytes()
    original, binding = json.loads(original_bytes), json.loads(binding_bytes)
    if binding.get('panel_protocol_sha256') != _sha(original_bytes) or binding.get('status') != 'frozen':
        raise ValueError('old frozen v20 protocol/binding identity drift')
    host_uuids = {}
    for entry in original['block_assignments'].values():
        host_uuids.setdefault(entry['host'], set()).add(entry['gpu_uuid'])
    host_ids = sorted(host_uuids)
    order = lambda values: sorted(values, key=lambda v: _sha(v.encode()))
    ids = {'longbench_v2': order(original['ids']['longbench_v2'][:6])[:2],
           'aime26': order(original['ids']['aime26'][:4])[:1],
           'ruler4k': order(original['ids']['ruler4k'])[:2]}
    seeds_by_dataset = {'longbench_v2': [101, 202], 'aime26': [101, 202], 'ruler4k': [101]}
    manifests, hashes = {}, {}
    for dataset in ids:
        raw = (Path(manifests_dir) / f'{dataset}_generation_manifest.json').read_bytes()
        if _sha(raw) != original['generation_manifest_sha256'][dataset]:
            raise ValueError(f'old frozen generation manifest drift: {dataset}')
        rows = {r['id']: r for r in json.loads(raw)}
        clean = []
        for id_ in ids[dataset]:
            row = rows[id_]
            if set(row) & old.GOLD or _sha(row['prompt'].encode()) != row.get('prompt_hash'):
                raise ValueError(f'gold/prompt drift: {dataset}')
            clean.append(row)
        manifests[dataset] = _bytes(clean)
        hashes[dataset] = _sha(manifests[dataset])
    updated = ('fp32_scores_bf16_pv', 'model_major')
    def method(parent, storage):
        contract = dict(kind='v21_method', parent_v20_arm=parent, scope=SCOPE,
                        output_score_precision=updated[0], output_layout=updated[1],
                        observation_producer=PRODUCER, bootstrap_policy=BOOTSTRAP)
        if storage != 'logical':
            contract['route_storage'] = storage
        return contract
    contracts = {'D_native': dict(kind='native'),
                 'T_scope': dict(kind='v20_legacy', parent_v20_arm='T_scope', scope=SCOPE),
                 'M3_boot_logical': method('M3_R3_A8_current_output', 'logical'),
                 'M1_boot_aligned16': method('M1_R1_A8_current_output', 'aligned16'),
                 'M3_boot_aligned16': method('M3_R3_A8_current_output', 'aligned16'),
                 'B_boot_aligned16': method('B_A8_matched', 'aligned16')}
    arms = list(PILOT_ARMS)
    protocol_id = 'v25_aligned6_pilot_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes),
        ids=ids, arms=contracts, seeds=seeds_by_dataset)))[:16]
    assignments, schedule, warm_blocks, stages = {}, [], [], {}
    block = 0
    for dataset in ('longbench_v2', 'aime26', 'ruler4k'):
        stages[dataset] = []
        for item_index, id_ in enumerate(ids[dataset]):
            for seed_index, seed in enumerate(seeds_by_dataset[dataset]):
                host = host_ids[(item_index + seed_index) % 2]
                assignment = dict(dataset=dataset, id=id_, seed=seed, host=host,
                                  gpu_uuid=next(iter(host_uuids[host])))
                assignments[str(block)] = assignment
                rotation = block % len(arms)
                first = arms[rotation:] + arms[:rotation]
                warm = dataset != 'ruler4k' or item_index == 0
                if warm:
                    warm_blocks.append(block)
                roles = (('attempt0', first), ('warm', first[::-1])) if warm else (('attempt0', first),)
                for role, arm_order in roles:
                    for arm in arm_order:
                        schedule.append(dict(index=len(schedule), block=block, arm=arm,
                            cell_id=cell_id(protocol_id, old.REVISION, old.sha_json(contracts[arm]), id_, seed),
                            role=role, repeat=0 if role == 'attempt0' else 1, **assignment))
                stages[dataset].append(block)
                block += 1
    protocol = dict(schema='v21_conditional_panel_v1', status='frozen', execution_ready=True,
                    panel_kind='aligned6_pilot', protocol_id=protocol_id, model_revision=old.REVISION,
                    v20_protocol_sha256=_sha(original_bytes), v20_binding_sha256=_sha(binding_bytes),
                    selected_scope=SCOPE, policy_point='P0', policy_sha256=binding['policy_sha256'],
                    output_layout='model_major', ids=ids, seeds=[101, 202],
                    seeds_by_dataset=seeds_by_dataset, warm_blocks=warm_blocks,
                    arms=arms, arm_contracts=contracts, block_assignments=assignments,
                    schedule=schedule, planned_executions=len(schedule),
                    generation_manifest_sha256=hashes, stages=stages,
                    selection_rule='sha256(id) ascending over the existing frozen inventories '
                                   '(LB first 6, AIME first 4, all 13 RULER); no scores or lengths used',
                    host_assignment_rule='host = (item_index + seed_index) mod 2 within each task',
                    quality_eligible=True, timing_eligible=True)
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=False)
    for dataset, content in manifests.items():
        _new(out_dir / f'{dataset}_generation_manifest.json', content)
    _new(out_dir / 'protocol.json', _bytes(protocol))
    return protocol


BRIDGE_ARMS = ('M3_boot_logical', 'M3_boot_aligned16')
BRIDGE_IDS = ('longbench_v2/66f9625fbb02136c067c5456',   # v25b-predeclared odd-K (K%16=7) profile input
              'longbench_v2/66f8c6b4bb02136c067c4480')   # control: highest KDIV (8) in the frozen LB six


def freeze_bridge(v20_protocol_path, v20_binding_path, manifests_dir, out_dir):
    """v25b CP2: logical vs aligned16 bootstrapped M3 on one odd-K and one
    aligned-class LB input x seeds 101/202 x first/warm (16). No frozen LB input
    has K%16==0; the control is the highest-KDIV input (ties by sha256)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    original_bytes, binding_bytes = Path(v20_protocol_path).read_bytes(), Path(v20_binding_path).read_bytes()
    original, binding = json.loads(original_bytes), json.loads(binding_bytes)
    if binding.get('panel_protocol_sha256') != _sha(original_bytes) or binding.get('status') != 'frozen':
        raise ValueError('old frozen v20 protocol/binding identity drift')
    host_uuids = {}
    for entry in original['block_assignments'].values():
        host_uuids.setdefault(entry['host'], set()).add(entry['gpu_uuid'])
    host_ids = sorted(host_uuids)
    ids = {'longbench_v2': list(BRIDGE_IDS)}
    if any(i not in original['ids']['longbench_v2'][:6] for i in BRIDGE_IDS):
        raise ValueError('bridge inputs must come from the frozen LB six')
    raw = (Path(manifests_dir) / 'longbench_v2_generation_manifest.json').read_bytes()
    if _sha(raw) != original['generation_manifest_sha256']['longbench_v2']:
        raise ValueError('old frozen generation manifest drift')
    rows = {r['id']: r for r in json.loads(raw)}
    clean = [rows[i] for i in BRIDGE_IDS]
    if any(set(r) & old.GOLD or _sha(r['prompt'].encode()) != r.get('prompt_hash') for r in clean):
        raise ValueError('gold/prompt drift')
    manifest = _bytes(clean)

    def method(storage):
        contract = dict(kind='v21_method', parent_v20_arm='M3_R3_A8_current_output', scope=SCOPE,
                        output_score_precision='fp32_scores_bf16_pv', output_layout='model_major',
                        observation_producer=PRODUCER, bootstrap_policy=BOOTSTRAP)
        if storage != 'logical':
            contract['route_storage'] = storage
        return contract
    contracts = {'M3_boot_logical': method('logical'), 'M3_boot_aligned16': method('aligned16')}
    arms = list(BRIDGE_ARMS)
    protocol_id = 'v25b_aligned_bridge_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes), ids=ids,
                                                             arms=contracts)))[:16]
    assignments, schedule, block = {}, [], 0
    for item_index, id_ in enumerate(ids['longbench_v2']):
        for seed_index, seed in enumerate((101, 202)):
            host = host_ids[(item_index + seed_index) % 2]
            assignment = dict(dataset='longbench_v2', id=id_, seed=seed, host=host,
                              gpu_uuid=next(iter(host_uuids[host])))
            assignments[str(block)] = assignment
            first = arms[block % 2:] + arms[:block % 2]
            for role, order in (('attempt0', first), ('warm', first[::-1])):
                for arm in order:
                    schedule.append(dict(index=len(schedule), block=block, arm=arm,
                        cell_id=cell_id(protocol_id, old.REVISION, old.sha_json(contracts[arm]), id_, seed),
                        role=role, repeat=0 if role == 'attempt0' else 1, **assignment))
            block += 1
    protocol = dict(schema='v21_conditional_panel_v1', status='frozen', execution_ready=True,
                    panel_kind='aligned_bridge', protocol_id=protocol_id, model_revision=old.REVISION,
                    v20_protocol_sha256=_sha(original_bytes), v20_binding_sha256=_sha(binding_bytes),
                    selected_scope=SCOPE, policy_point='P0', policy_sha256=binding['policy_sha256'],
                    output_layout='model_major', ids=ids, seeds=[101, 202], arms=arms,
                    arm_contracts=contracts, block_assignments=assignments, schedule=schedule,
                    planned_executions=len(schedule),
                    generation_manifest_sha256={'longbench_v2': _sha(manifest)},
                    stages={'bridge': list(range(block))},
                    selection_rule='v25b-predeclared odd-K profile input + highest-KDIV frozen LB control',
                    host_assignment_rule='host = (item_index + seed_index) mod 2',
                    quality_eligible=True, timing_eligible=True)
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=False)
    _new(out_dir / 'longbench_v2_generation_manifest.json', manifest)
    _new(out_dir / 'protocol.json', _bytes(protocol))
    return protocol


SEVEN_ARMS = ('D_native', 'T_scope', 'M1_R1_A8', 'M2_pool_R1_A8', 'M3_R3_A8', 'M3_R3_A16', 'B_A8')


def freeze_seven(v20_protocol_path, v20_binding_path, manifests_dir, out_dir):
    """v26 quick seven-arm panel (84): LB odd-K + KDIV-8, AIME first two by sha256,
    RULER first two tasks by sha256; seed 101; first + warm; hosts alternate by block.
    Inputs follow fixed ID rules chosen before any v26 output."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    original_bytes, binding_bytes = Path(v20_protocol_path).read_bytes(), Path(v20_binding_path).read_bytes()
    original, binding = json.loads(original_bytes), json.loads(binding_bytes)
    if binding.get('panel_protocol_sha256') != _sha(original_bytes) or binding.get('status') != 'frozen':
        raise ValueError('old frozen v20 protocol/binding identity drift')
    host_uuids = {}
    for entry in original['block_assignments'].values():
        host_uuids.setdefault(entry['host'], set()).add(entry['gpu_uuid'])
    host_ids = sorted(host_uuids)
    order = lambda values: sorted(values, key=lambda v: _sha(v.encode()))
    ids = {'longbench_v2': list(BRIDGE_IDS),
           'aime26': order(original['ids']['aime26'][:4])[:2],
           'ruler4k': order(original['ids']['ruler4k'])[:2]}
    manifests, hashes = {}, {}
    for dataset in ids:
        raw = (Path(manifests_dir) / f'{dataset}_generation_manifest.json').read_bytes()
        if _sha(raw) != original['generation_manifest_sha256'][dataset]:
            raise ValueError(f'old frozen generation manifest drift: {dataset}')
        rows = {r['id']: r for r in json.loads(raw)}
        clean = [rows[i] for i in ids[dataset]]
        if any(set(r) & old.GOLD or _sha(r['prompt'].encode()) != r.get('prompt_hash') for r in clean):
            raise ValueError(f'gold/prompt drift: {dataset}')
        manifests[dataset] = _bytes(clean)
        hashes[dataset] = _sha(manifests[dataset])

    def method(parent, **extra):
        contract = dict(kind='v21_method', parent_v20_arm=parent, scope=SCOPE,
                        output_score_precision='fp32_scores_bf16_pv', output_layout='model_major',
                        observation_producer=PRODUCER, bootstrap_policy=BOOTSTRAP,
                        route_storage='aligned16_odd')
        contract.update(extra)
        return contract
    contracts = {'D_native': dict(kind='native'),
                 'T_scope': dict(kind='v20_legacy', parent_v20_arm='T_scope', scope=SCOPE),
                 'M1_R1_A8': method('M1_R1_A8_current_output'),
                 'M2_pool_R1_A8': method('M1_R1_A8_current_output', mu_mode='pooled'),
                 'M3_R3_A8': method('M3_R3_A8_current_output'),
                 'M3_R3_A16': method('M3_R3_A8_current_output', score_period=16),
                 'B_A8': method('B_A8_matched')}
    arms = list(SEVEN_ARMS)
    protocol_id = 'v26_seven_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes), ids=ids,
                                                   arms=contracts)))[:16]
    assignments, schedule, block, stages = {}, [], 0, {}
    for dataset in ('longbench_v2', 'aime26', 'ruler4k'):
        stages[dataset] = []
        for id_ in ids[dataset]:
            host = host_ids[block % 2]
            assignment = dict(dataset=dataset, id=id_, seed=101, host=host,
                              gpu_uuid=next(iter(host_uuids[host])))
            assignments[str(block)] = assignment
            first = arms[block % len(arms):] + arms[:block % len(arms)]
            for role, arm_order in (('attempt0', first), ('warm', first[::-1])):
                for arm in arm_order:
                    schedule.append(dict(index=len(schedule), block=block, arm=arm,
                        cell_id=cell_id(protocol_id, old.REVISION, old.sha_json(contracts[arm]), id_, 101),
                        role=role, repeat=0 if role == 'attempt0' else 1, **assignment))
            stages[dataset].append(block)
            block += 1
    protocol = dict(schema='v21_conditional_panel_v1', status='frozen', execution_ready=True,
                    panel_kind='v26_seven', protocol_id=protocol_id, model_revision=old.REVISION,
                    v20_protocol_sha256=_sha(original_bytes), v20_binding_sha256=_sha(binding_bytes),
                    selected_scope=SCOPE, policy_point='P0', policy_sha256=binding['policy_sha256'],
                    output_layout='model_major', ids=ids, seeds=[101],
                    arms=arms, arm_contracts=contracts, block_assignments=assignments,
                    schedule=schedule, planned_executions=len(schedule),
                    generation_manifest_sha256=hashes, stages=stages,
                    selection_rule='LB: only odd-K frozen input + highest-KDIV (sha tie); AIME/RULER: first two by sha256(id); no outcomes used',
                    host_assignment_rule='host = block mod 2 (each host one LB, one AIME, one RULER)',
                    quality_eligible=True, timing_eligible=True)
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=False)
    for dataset, content in manifests.items():
        _new(out_dir / f'{dataset}_generation_manifest.json', content)
    _new(out_dir / 'protocol.json', _bytes(protocol))
    return protocol


V27_EXTRA_KEYS = ('mu_mode', 'score_period', 'decision_interval', 'hold_only', 'threshold_shift',
                  'min_route_keys', 'share_layers', 'route_layers', 'consumer64', 'memory_caps', 'fused_observe',
                  'fresh_fused', 'route_pipeline')


def freeze_v27(spec_path, v20_protocol_path, v20_binding_path, pool_dir, out_dir):
    """v27 generic frozen panel from a committed spec (ids, seeds, named arms).

    spec: {"name", "ids": {dataset: [...]}, "seeds": [...], "warm": bool,
           "arms": {name: {"kind": "native"|"fresh_T"|"method", "parent": v20 arm,
                           "extra": {v21 optional keys}}}}
    Pool rows must carry a consistent prompt hash and no gold; any id also in the
    frozen v20 manifest must be byte-identical to it. Host = (question index + seed
    index) mod 2 within each dataset; every arm of a question-seed block on one GPU.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    original_bytes, binding_bytes = Path(v20_protocol_path).read_bytes(), Path(v20_binding_path).read_bytes()
    original, binding = json.loads(original_bytes), json.loads(binding_bytes)
    if binding.get('panel_protocol_sha256') != _sha(original_bytes) or binding.get('status') != 'frozen':
        raise ValueError('old frozen v20 protocol/binding identity drift')
    host_uuids = {}
    for entry in original['block_assignments'].values():
        host_uuids.setdefault(entry['host'], set()).add(entry['gpu_uuid'])
    host_ids = sorted(host_uuids)
    ids, seeds = spec['ids'], list(spec['seeds'])
    if not ids or not set(ids) <= {'longbench_v2', 'aime26', 'ruler4k', 'ruler32k', 'ruler64k'} or not seeds or \
            not set(seeds) <= {101, 202, 303} or len(set(seeds)) != len(seeds):
        raise ValueError('v27 spec ids/seeds outside the allowed tasks/seeds')
    manifests, hashes = {}, {}
    for dataset, wanted in ids.items():
        pool = {r['id']: r for r in json.loads((Path(pool_dir) / f'{dataset}_pool_manifest.json').read_bytes())}
        frozen_path = Path(pool_dir).parent / 'v20_frozen' / f'{dataset}_generation_manifest.json'
        frozen = {r['id']: r for r in json.loads(frozen_path.read_bytes())} if frozen_path.is_file() else {}
        if len(set(wanted)) != len(wanted) or any(i not in pool for i in wanted):
            raise ValueError(f'v27 ids missing from pool or duplicated: {dataset}')
        clean = [pool[i] for i in wanted]
        if any(set(r) & old.GOLD or _sha(r['prompt'].encode()) != r.get('prompt_hash') for r in clean):
            raise ValueError(f'gold/prompt drift: {dataset}')
        if any(r['id'] in frozen and r != frozen[r['id']] for r in clean):
            raise ValueError(f'pool row differs from the frozen v20 row: {dataset}')
        manifests[dataset] = _bytes(clean)
        hashes[dataset] = _sha(manifests[dataset])

    def method(parent, extra):
        if set(extra) - set(V27_EXTRA_KEYS):
            raise ValueError(f'unknown v27 arm key: {sorted(set(extra) - set(V27_EXTRA_KEYS))}')
        contract = dict(kind='v21_method', parent_v20_arm=parent, scope=SCOPE,
                        output_score_precision='fp32_scores_bf16_pv', output_layout='model_major',
                        observation_producer=PRODUCER, bootstrap_policy=BOOTSTRAP,
                        route_storage='aligned16_odd')
        contract.update(extra)
        return contract
    contracts = {}
    for name, arm in spec['arms'].items():
        if arm['kind'] == 'native':
            if name != 'D_native':
                raise ValueError('native arm must be D_native')
            contracts[name] = dict(kind='native')
        elif arm['kind'] == 'dense_c64':
            # strongest dense: native model with GLOBAL attention through the 64-row kernel (all kept)
            contracts[name] = dict(kind='v27_dense', control='D_c64', scope=SCOPE)
        elif arm['kind'] == 'dense_matched':
            # Same Triton consumer, FP32 scores, model-major output, every legal tile kept.
            contracts[name] = dict(kind='v21_control', parent_v20_arm='D_matched', scope=SCOPE,
                                   output_score_precision='fp32_scores_bf16_pv', output_layout='model_major')
        elif arm['kind'] == 'fresh_T':
            contracts[name] = dict(kind='v20_legacy', parent_v20_arm='T_scope', scope=SCOPE)
        elif arm['kind'] == 'method':
            contracts[name] = method(arm['parent'], arm.get('extra', {}))
        else:
            raise ValueError(f'unknown v27 arm kind: {arm["kind"]}')
    arms = list(spec['arms'])
    protocol_id = 'v27_' + spec['name'] + '_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes), ids=ids,
                                                              seeds=seeds, arms=contracts)))[:16]
    assignments, schedule, block, stages = {}, [], 0, {}
    for dataset in ('longbench_v2', 'aime26', 'ruler4k', 'ruler32k', 'ruler64k'):
        if dataset not in ids:
            continue
        stages[dataset] = []
        for q_index, id_ in enumerate(ids[dataset]):
            for s_index, seed in enumerate(seeds):
                host = host_ids[(q_index + s_index) % 2]
                assignment = dict(dataset=dataset, id=id_, seed=seed, host=host,
                                  gpu_uuid=next(iter(host_uuids[host])))
                assignments[str(block)] = assignment
                first = arms[block % len(arms):] + arms[:block % len(arms)]
                roles = (('attempt0', first), ('warm', first[::-1])) if spec.get('warm', True) else \
                    (('attempt0', first),)
                for role, arm_order in roles:
                    for arm in arm_order:
                        schedule.append(dict(index=len(schedule), block=block, arm=arm,
                            cell_id=cell_id(protocol_id, old.REVISION, old.sha_json(contracts[arm]), id_, seed),
                            role=role, repeat=0 if role == 'attempt0' else 1, **assignment))
                stages[dataset].append(block)
                block += 1
    protocol = dict(schema='v21_conditional_panel_v1', status='frozen', execution_ready=True,
                    panel_kind='v27_panel', panel_name=spec['name'], protocol_id=protocol_id,
                    model_revision=old.REVISION, spec_sha256=_sha(spec_bytes),
                    v20_protocol_sha256=_sha(original_bytes), v20_binding_sha256=_sha(binding_bytes),
                    selected_scope=SCOPE, policy_point='P0', policy_sha256=binding['policy_sha256'],
                    output_layout='model_major', ids=ids, seeds=seeds, warm=bool(spec.get('warm', True)),
                    arms=arms, arm_contracts=contracts, block_assignments=assignments,
                    schedule=schedule, planned_executions=len(schedule),
                    generation_manifest_sha256=hashes, stages=stages,
                    selection_rule=spec.get('selection_rule', ''),
                    host_assignment_rule='host = (question index + seed index) mod 2 within each dataset',
                    extra_gold_sha256=spec.get('extra_gold_sha256', {}),
                    long_context_execution=spec.get('long_context_execution'),
                    quality_eligible=True, timing_eligible=bool(spec.get('warm', True)))
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=False)
    for dataset, content in manifests.items():
        _new(out_dir / f'{dataset}_generation_manifest.json', content)
    _new(out_dir / 'protocol.json', _bytes(protocol))
    return protocol


def _contracts(mode, output_layout):
    legacy = ('legacy_bf16_scores', 'head_major')
    updated = ('fp32_scores_bf16_pv', output_layout)
    def control(precision, layout):
        return dict(kind='v21_control', parent_v20_arm='D_matched', scope=SCOPE,
                    output_score_precision=precision, output_layout=layout)
    def method(parent, precision, layout):
        return dict(kind='v21_method', parent_v20_arm=parent, scope=SCOPE,
                    output_score_precision=precision, output_layout=layout)
    if mode in ('bootstrap6', 'bootstrap6_ruler'):
        updated = ('fp32_scores_bf16_pv', output_layout)
        def v23(parent, bootstrap):
            contract = dict(method(parent, *updated), observation_producer=PRODUCER)
            if bootstrap:
                contract['bootstrap_policy'] = BOOTSTRAP
            return contract
        return {
            'D_native': dict(kind='native'),
            'T_scope': dict(kind='v20_legacy', parent_v20_arm='T_scope', scope=SCOPE),
            'M3_R3_A8_incumbent': v23('M3_R3_A8_current_output', False),
            'M1_native_bootstrap2_observe1': v23('M1_R1_A8_current_output', True),
            'M3_native_bootstrap2_observe1': v23('M3_R3_A8_current_output', True),
            'B_native_bootstrap2_observe1': v23('B_A8_matched', True),
        }
    if mode == 'numeric7':
        return {
            'D_native': dict(kind='native'),
            'D_matched_legacy': control(*legacy),
            'D_matched_new': control(*updated),
            'T_scope': dict(kind='v20_legacy', parent_v20_arm='T_scope', scope=SCOPE),
            'M3_R3_A8_legacy': method('M3_R3_A8_current_output', *legacy),
            'M3_R3_A8_new': method('M3_R3_A8_current_output', *updated),
            'B_A8_matched_new': method('B_A8_matched', *updated),
        }
    return {'M3_R3_A8_legacy': method('M3_R3_A8_current_output', *legacy),
            'M3_R3_A8_layout': method('M3_R3_A8_current_output',
                                     'legacy_bf16_scores', 'model_major')}


def freeze(mode, v20_protocol_path, v20_binding_path, manifests_dir, output_layout,
           out_dir):
    if mode not in ('numeric7', 'layout_pair', 'bootstrap6', 'bootstrap6_ruler') or output_layout not in ('head_major', 'model_major'):
        raise ValueError('explicit numeric7/layout_pair and qualified output layout required')
    if mode == 'layout_pair' and output_layout != 'model_major':
        raise ValueError('layout_pair requires model_major successor')
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    original_bytes, binding_bytes = Path(v20_protocol_path).read_bytes(), Path(v20_binding_path).read_bytes()
    original, binding = json.loads(original_bytes), json.loads(binding_bytes)
    if (original.get('schema') != 'v20_fan_panel_v1' or original.get('planned_executions') != 700 or
            original.get('model_revision') != old.REVISION or
            binding.get('status') != 'frozen' or binding.get('policy_point') != 'P0' or
            binding.get('scope') != SCOPE or binding.get('panel_protocol_sha256') != _sha(original_bytes)):
        raise ValueError('old frozen v20 protocol/binding identity drift')
    if set(binding.get('host_configs', {})) != set(binding.get('host_models', {})):
        raise ValueError('old host config/model assignment differs')
    host_uuids = {}
    for entry in original['block_assignments'].values():
        host_uuids.setdefault(entry['host'], set()).add(entry['gpu_uuid'])
    if set(host_uuids) != set(binding['host_models']) or any(len(v) != 1 for v in host_uuids.values()) or len(host_uuids) != 2:
        raise ValueError('expected two pinned old host/GPU identities')
    host_ids = sorted(host_uuids)
    # bootstrap6: first six LB and first four AIME IDs in frozen v20 order,
    # chosen before any v23 output; RULER is a separate later stage.
    ids = {dataset: list(original['ids'][dataset][:count]) for dataset, count in
           (('ruler4k', 4 if mode == 'numeric7' else 0),
            ('aime26', 4 if mode == 'bootstrap6' else 6), ('longbench_v2', 6))}
    if mode == 'bootstrap6':
        ids.pop('ruler4k')
    if mode == 'bootstrap6_ruler':
        # v24: the RULER stage omitted by bootstrap6; all 13 frozen task-balanced IDs.
        ids = {'ruler4k': list(original['ids']['ruler4k'])}
    if mode in ('numeric7', 'bootstrap6_ruler'):
        ruler_rows = _json(Path(manifests_dir) / 'ruler4k_generation_manifest.json')
        tasks = [next(row['task'] for row in ruler_rows if row['id'] == id_)
                 for id_ in ids['ruler4k']]
        if len(set(tasks)) != len(ids['ruler4k']):
            raise ValueError('ordered RULER IDs are not distinct tasks')
    manifests, hashes = {}, {}
    for dataset in [d for d in old.DATASETS if d in ids]:
        source = Path(manifests_dir) / f'{dataset}_generation_manifest.json'
        raw = source.read_bytes()
        if _sha(raw) != original['generation_manifest_sha256'][dataset]:
            raise ValueError(f'old frozen generation manifest drift: {dataset}')
        rows = json.loads(raw)
        if {r['id'] for r in rows} != set(original['ids'][dataset]) or len(rows) != len(original['ids'][dataset]):
            raise ValueError(f'old frozen ID inventory drift: {dataset}')
        chosen = {r['id']: r for r in rows if r['id'] in ids[dataset]}
        if len(chosen) != len(ids[dataset]):
            raise ValueError(f'missing selected input: {dataset}')
        clean = []
        for id_ in ids[dataset]:
            row = chosen[id_]
            if set(row) & old.GOLD or _sha(row['prompt'].encode()) != row.get('prompt_hash'):
                raise ValueError(f'gold/prompt drift: {dataset}')
            if row.get('thinking') is not (dataset != 'ruler4k') or (
                    type(row.get('generation_budget')) is not int or
                    not 1 <= row['generation_budget'] <= 8192):
                raise ValueError(f'task generation contract drift: {dataset}')
            clean.append(row)
        manifests[dataset] = _bytes(clean)
        hashes[dataset] = _sha(manifests[dataset])
    arms = list(NUMERIC_ARMS if mode == 'numeric7' else
                BOOTSTRAP_ARMS if mode in ('bootstrap6', 'bootstrap6_ruler') else LAYOUT_ARMS)
    seeds = [101] if mode == 'bootstrap6_ruler' else [101, 202]
    # Warm timing repeats only on the first two frozen RULER tasks (predeclared).
    warm_blocks = {0, 1} if mode == 'bootstrap6_ruler' else None
    contracts = _contracts(mode, output_layout)
    protocol_id = 'v21_' + mode + '_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes),
        old_binding=_sha(binding_bytes), ids=ids, arms=contracts, output_layout=output_layout)))[:16]
    assignments, schedule = {}, []
    block = 0
    # Whole question-seed blocks alternate old hosts within each task family.
    for dataset in [d for d in ('longbench_v2', 'aime26', 'ruler4k') if d in ids]:
        for item_index, id_ in enumerate(ids[dataset]):
            for seed_index, seed in enumerate(seeds):
                host = host_ids[(item_index * 2 + seed_index) % 2] if len(seeds) == 2 else host_ids[item_index % 2]
                assignment = dict(dataset=dataset, id=id_, seed=seed, host=host,
                                  gpu_uuid=next(iter(host_uuids[host])))
                assignments[str(block)] = assignment
                rotation = block % len(arms)
                first = arms[rotation:] + arms[:rotation]
                roles = (('attempt0', first), ('warm', first[::-1]))
                if warm_blocks is not None and block not in warm_blocks:
                    roles = roles[:1]
                for role, order in roles:
                    for arm in order:
                        schedule.append(dict(index=len(schedule), block=block, arm=arm,
                            cell_id=cell_id(protocol_id, old.REVISION,
                                            old.sha_json(contracts[arm]), id_, seed),
                            role=role, repeat=0 if role == 'attempt0' else 1,
                            **assignment))
                block += 1
    protocol = dict(schema='v21_conditional_panel_v1', status='frozen',
                    execution_ready=True, panel_kind=mode, protocol_id=protocol_id,
                    model_revision=old.REVISION, v20_protocol_sha256=_sha(original_bytes),
                    v20_binding_sha256=_sha(binding_bytes), selected_scope=SCOPE,
                    policy_point='P0', policy_sha256=binding['policy_sha256'],
                    output_layout=output_layout, ids=ids, seeds=seeds,
                    warm_blocks=(sorted(warm_blocks) if warm_blocks is not None else None),
                    arms=arms, arm_contracts=contracts, block_assignments=assignments,
                    schedule=schedule, planned_executions=len(schedule),
                    generation_manifest_sha256=hashes,
                    stages=({'lb_preview': [0, 1, 2, 3], 'lb_rest': list(range(4, 12)),
                             'aime': list(range(12, block))} if mode == 'bootstrap6'
                            else {'ruler': list(range(block))} if mode == 'bootstrap6_ruler'
                            else {'main': list(range(block))}),
                    selection_rule=('ordered first six LB and first four AIME v20 IDs; RULER deferred; no scores'
                                    if mode == 'bootstrap6' else
                                    'all 13 frozen task-balanced v20 RULER IDs, seed 101, warm on first two tasks; no scores'
                                    if mode == 'bootstrap6_ruler' else
                                    'ordered first six LB/AIME and first four distinct-task RULER v20 IDs; no scores'),
                    host_assignment_rule='whole question-seed blocks alternate two old pinned hosts per task',
                    quality_eligible=True, timing_eligible=True)
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=False)
    for dataset, content in manifests.items():
        _new(out_dir / f'{dataset}_generation_manifest.json', content)
    _new(out_dir / 'protocol.json', _bytes(protocol))
    return protocol


def _source(plugin, old_config):
    source = old.source_hashes(plugin, old_config['library'], old_config['torch_library'],
                               old_config.get('support_build'))
    for name in ('v21_freeze_panel.py', 'v21_run.py'):
        path = Path(__file__).resolve().with_name(name)
        source[str(path)] = _sha(path.read_bytes())
    return source


def _old_config(binding, host, dataset, arm):
    item = binding['host_configs'][host][dataset][arm]
    path = Path(item['path'])
    raw = path.read_bytes()
    if _sha(raw) != item['sha256']:
        raise ValueError(f'old host config byte drift: {host}/{dataset}/{arm}')
    return json.loads(raw)


def _base(old_config, *, protocol, dataset, manifest_path, model, source, source_commit):
    base = dict(old_config)
    for key in ('fingerprint', 'condition', 'plugin', 'v20_arm', 'v20_scope',
                'decision_interval', 'score_refresh_period', 'output_mode', 'control'):
        base.pop(key, None)
    base.update(model=str(Path(model).resolve()), revision=old.REVISION,
                source_commit=source_commit, source_hashes=source,
                model_metadata_hashes=old.model_hashes(model),
                manifest=str(Path(manifest_path).resolve()),
                manifest_sha256=protocol['generation_manifest_sha256'][dataset],
                ids=protocol['ids'][dataset], seeds=sorted(set(protocol['seeds']) | {101, 202}),
                phase='v21_generation', policy_name='P0',
                policy_sha256=protocol['policy_sha256'],
                consumer='triton', kernel_variant='generic',
                support='native_mask', support_geometry='native_legal',
                diagnostic=False, timing_events=True)
    return base


def bind_host(old_binding_path, host, source_commit, protocol_path, manifests_dir, out):
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('source commit must be exact 40-hex SHA')
    protocol_raw, old_binding_raw = Path(protocol_path).read_bytes(), Path(old_binding_path).read_bytes()
    protocol, binding = json.loads(protocol_raw), json.loads(old_binding_raw)
    validate_protocol(protocol)
    if (protocol.get('v20_binding_sha256') != _sha(old_binding_raw) or
            binding.get('policy_point') != 'P0' or binding.get('scope') != SCOPE or
            host not in binding.get('host_configs', {})):
        raise ValueError('host old binding differs from frozen v21 panel')
    model = binding['host_models'][host]
    manifest_paths = {d: Path(manifests_dir) / f'{d}_generation_manifest.json'
                      for d in protocol['ids']}
    for dataset, path in manifest_paths.items():
        if _sha(path.read_bytes()) != protocol['generation_manifest_sha256'][dataset]:
            raise ValueError(f'host frozen v21 manifest drift: {dataset}')
    config_dir = Path(out).with_name(Path(out).stem + '_configs')
    if Path(out).exists() or config_dir.exists():
        raise FileExistsError('host binding/config output exists')
    configs = {}
    from experiments.numerical_qk_reuse import v21
    for dataset in protocol['ids']:
        configs[dataset] = {}
        for arm in protocol['arms']:
            contract = protocol['arm_contracts'][arm]
            parent = contract.get('parent_v20_arm')
            source_arm = ('D_native' if contract['kind'] == 'native' else
                          'T_scope' if parent == 'T_scope' else
                          'D_matched' if parent == 'D_matched' else parent)
            inherited = _old_config(binding, host, 'ruler4k' if dataset in ('ruler32k', 'ruler64k') else dataset,
                                    'D_native' if contract['kind'] == 'v27_dense' else source_arm)
            plugin = (old.CONTROLS if source_arm in ('D_native', 'D_matched', 'T_scope')
                      else v20.PLUGIN)
            base = _base(inherited, protocol=protocol, dataset=dataset,
                         manifest_path=manifest_paths[dataset], model=model,
                         source=_source(plugin, inherited), source_commit=source_commit)
            if contract['kind'] == 'native':
                result = old.control_config(base, 'native_dense', SCOPE)
            elif contract['kind'] == 'v27_dense':
                from experiments.numerical_qk_reuse import v27_fast_dense as fast
                result = old.control_config(base, 'native_dense', SCOPE)
                result = {k: v for k, v in result.items() if k not in ('fingerprint', 'condition', 'plugin')}
                result.update(control=contract['control'], plugin=fast.PLUGIN, condition=fast.CONDITION)
                for src in (Path(fast.__file__).resolve(), Path(fast.__file__).resolve().with_name('v27_consumer64.py')):
                    result['source_hashes'][str(src)] = _sha(src.read_bytes())
                # the runner records the config fingerprint in every receipt
                result['fingerprint'] = old.sha_json(result)
            elif contract['kind'] == 'v20_legacy':
                result = old.control_config(dict(base, control='T_scope'), 'v20_fresh_T', SCOPE)
            elif contract['kind'] == 'v21_control':
                result = v21.effective_control_config(base, SCOPE,
                    output_score_precision=contract['output_score_precision'],
                    output_layout=contract['output_layout'])
            else:
                base['selector'] = ('legacy_recompute' if parent == 'B_A8_matched'
                                    else 'prefix_block_summary')
                base['selector_layers'] = 'all'
                result = v21.effective_config(base, parent, SCOPE,
                    output_score_precision=contract['output_score_precision'],
                    output_layout=contract['output_layout'],
                    bootstrap_policy=contract.get('bootstrap_policy'),
                    observation_producer=contract.get('observation_producer', 'repeat_interleave'),
                    route_storage=contract.get('route_storage', 'logical'),
                    mu_mode=contract.get('mu_mode', 'exact'),
                    score_period=contract.get('score_period', 8),
                    decision_interval=contract.get('decision_interval'),
                    hold_only=contract.get('hold_only', False),
                    threshold_shift=contract.get('threshold_shift'),
                    min_route_keys=contract.get('min_route_keys'),
                    share_layers=contract.get('share_layers'), route_layers=contract.get('route_layers'),
                    consumer64=contract.get('consumer64'), memory_caps=contract.get('memory_caps'),
                    fused_observe=contract.get('fused_observe', False),
                    fresh_fused=contract.get('fresh_fused', False),
                    route_pipeline=contract.get('route_pipeline', False))
            path = config_dir / dataset / f'{arm}.json'
            _new(path, _bytes(result))
            configs[dataset][arm] = dict(path=str(path.resolve()), sha256=_sha(path.read_bytes()))
    fragment = dict(schema='v21_conditional_binding_fragment_v1',
                    status='host_fragment', panel_protocol_sha256=_sha(protocol_raw),
                    v20_binding_sha256=_sha(old_binding_raw),
                    source_commit=source_commit, policy_sha256=protocol['policy_sha256'],
                    host_models={host: str(Path(model).resolve())},
                    host_configs={host: configs})
    _new(out, _bytes(fragment))
    return fragment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fr = sub.add_parser('freeze')
    fr.add_argument('--mode', choices=('numeric7', 'layout_pair', 'bootstrap6', 'bootstrap6_ruler',
                                        'aligned6_pilot', 'aligned_bridge', 'v26_seven', 'v27_panel'), required=True)
    fr.add_argument('--v20-protocol', type=Path, required=True)
    fr.add_argument('--spec', type=Path, help='v27_panel: committed arm/id/seed spec')
    fr.add_argument('--v20-binding', type=Path, required=True)
    fr.add_argument('--manifests-dir', type=Path, required=True)
    fr.add_argument('--output-layout', choices=('head_major', 'model_major'), required=True)
    fr.add_argument('--out-dir', type=Path, required=True)
    hb = sub.add_parser('host-bind')
    hb.add_argument('--old-binding', type=Path, required=True)
    hb.add_argument('--host', required=True)
    hb.add_argument('--source-commit', required=True)
    hb.add_argument('--protocol', type=Path, required=True)
    hb.add_argument('--manifests-dir', type=Path, required=True)
    hb.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze' and args.mode == 'v27_panel':
        result = freeze_v27(args.spec, args.v20_protocol, args.v20_binding, args.manifests_dir, args.out_dir)
    elif args.command == 'freeze' and args.mode in ('aligned6_pilot', 'aligned_bridge', 'v26_seven'):
        result = dict(aligned6_pilot=freeze_pilot, aligned_bridge=freeze_bridge,
                      v26_seven=freeze_seven)[args.mode](
            args.v20_protocol, args.v20_binding, args.manifests_dir, args.out_dir)
        print(json.dumps(dict(protocol=str((args.out_dir / 'protocol.json').resolve()),
                              planned_executions=result['planned_executions']), sort_keys=True))
    elif args.command == 'freeze':
        result = freeze(args.mode, args.v20_protocol, args.v20_binding,
                        args.manifests_dir, args.output_layout, args.out_dir)
        print(json.dumps(dict(protocol=str((args.out_dir / 'protocol.json').resolve()),
                              planned_executions=result['planned_executions']), sort_keys=True))
    else:
        bind_host(args.old_binding, args.host, args.source_commit, args.protocol,
                  args.manifests_dir, args.out)
        print(json.dumps(dict(host=args.host, binding_fragment=str(args.out.resolve())),
                         sort_keys=True))


if __name__ == '__main__':
    main()
