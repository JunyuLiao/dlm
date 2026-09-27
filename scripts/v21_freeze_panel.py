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


def _contracts(mode, output_layout):
    legacy = ('legacy_bf16_scores', 'head_major')
    updated = ('fp32_scores_bf16_pv', output_layout)
    def control(precision, layout):
        return dict(kind='v21_control', parent_v20_arm='D_matched', scope=SCOPE,
                    output_score_precision=precision, output_layout=layout)
    def method(parent, precision, layout):
        return dict(kind='v21_method', parent_v20_arm=parent, scope=SCOPE,
                    output_score_precision=precision, output_layout=layout)
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
    if mode not in ('numeric7', 'layout_pair') or output_layout not in ('head_major', 'model_major'):
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
    ids = {dataset: list(original['ids'][dataset][:count]) for dataset, count in
           (('ruler4k', 4 if mode == 'numeric7' else 0),
            ('aime26', 6), ('longbench_v2', 6))}
    if mode == 'numeric7':
        ruler_rows = _json(Path(manifests_dir) / 'ruler4k_generation_manifest.json')
        tasks = [next(row['task'] for row in ruler_rows if row['id'] == id_)
                 for id_ in ids['ruler4k']]
        if len(set(tasks)) != 4:
            raise ValueError('first four ordered RULER IDs are not distinct tasks')
    manifests, hashes = {}, {}
    for dataset in old.DATASETS:
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
    arms = list(NUMERIC_ARMS if mode == 'numeric7' else LAYOUT_ARMS)
    contracts = _contracts(mode, output_layout)
    protocol_id = 'v21_' + mode + '_' + _sha(_bytes(dict(old_protocol=_sha(original_bytes),
        old_binding=_sha(binding_bytes), ids=ids, arms=contracts, output_layout=output_layout)))[:16]
    assignments, schedule = {}, []
    block = 0
    # Whole question-seed blocks alternate old hosts within each task family.
    for dataset in ('longbench_v2', 'aime26', 'ruler4k'):
        for item_index, id_ in enumerate(ids[dataset]):
            for seed_index, seed in enumerate((101, 202)):
                host = host_ids[(item_index * 2 + seed_index) % 2]
                assignment = dict(dataset=dataset, id=id_, seed=seed, host=host,
                                  gpu_uuid=next(iter(host_uuids[host])))
                assignments[str(block)] = assignment
                rotation = block % len(arms)
                first = arms[rotation:] + arms[:rotation]
                for role, order in (('attempt0', first), ('warm', first[::-1])):
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
                    output_layout=output_layout, ids=ids, seeds=[101, 202],
                    arms=arms, arm_contracts=contracts, block_assignments=assignments,
                    schedule=schedule, planned_executions=len(schedule),
                    generation_manifest_sha256=hashes,
                    stages={'main': list(range(block))},
                    selection_rule='ordered first six LB/AIME and first four distinct-task RULER v20 IDs; no scores',
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
                ids=protocol['ids'][dataset], seeds=[101, 202],
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
            inherited = _old_config(binding, host, dataset, source_arm)
            plugin = (old.CONTROLS if source_arm in ('D_native', 'D_matched', 'T_scope')
                      else v20.PLUGIN)
            base = _base(inherited, protocol=protocol, dataset=dataset,
                         manifest_path=manifest_paths[dataset], model=model,
                         source=_source(plugin, inherited), source_commit=source_commit)
            if contract['kind'] == 'native':
                result = old.control_config(base, 'native_dense', SCOPE)
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
                    output_layout=contract['output_layout'])
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
    fr.add_argument('--mode', choices=('numeric7', 'layout_pair'), required=True)
    fr.add_argument('--v20-protocol', type=Path, required=True)
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
    if args.command == 'freeze':
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
