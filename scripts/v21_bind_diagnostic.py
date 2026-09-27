"""CPU-only host-local binder for frozen v21 numerical and direct-cost probes.

The verified v20 D_native config supplies model, P0 policy and host libraries.
This binder checks its frozen protocol/manifest identities, chooses only the
predeclared first (host 0) or second (host 1) question triple per task, and
never loads a model or launches a GPU job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from experiments.numerical_qk_reuse import v20
from scripts import v20_bind as old


SCOPE = v20.GLOBAL_ONLY_NATIVE_LOCAL
EXTRA_SOURCES = ('v21_bind_diagnostic.py', 'v21_numerical_diagnostic.py',
                 'v21_profile.py')


def _source_inventory(plugin, old_config):
    inventory = old.source_hashes(plugin, old_config['library'],
                                  old_config['torch_library'],
                                  old_config.get('support_build'))
    script_dir = Path(__file__).resolve().parent
    for name in EXTRA_SOURCES:
        source = script_dir / name
        inventory[str(source)] = old.sha_bytes(source.read_bytes())
    return inventory


def _manifests(protocol, directory):
    paths, rows = {}, {}
    for dataset in old.DATASETS:
        path = Path(directory) / f'{dataset}_generation_manifest.json'
        raw = path.read_bytes()
        if old.sha_bytes(raw) != protocol['generation_manifest_sha256'][dataset]:
            raise ValueError(f'frozen generation manifest byte drift: {dataset}')
        items = json.loads(raw)
        if {row['id'] for row in items} != set(protocol['ids'][dataset]) or len(items) != len(protocol['ids'][dataset]):
            raise ValueError(f'frozen generation manifest ID drift: {dataset}')
        for row in items:
            if set(row) & old.GOLD or old.sha_bytes(row['prompt'].encode()) != row['prompt_hash']:
                raise ValueError(f'gold/prompt identity drift: {dataset}')
            if row.get('thinking') is not (dataset != 'ruler4k'):
                raise ValueError(f'thinking contract drift: {dataset}')
            budget = row.get('generation_budget')
            if type(budget) is not int or not 1 <= budget <= 8192 or (dataset != 'ruler4k' and budget != 8192):
                raise ValueError(f'generation budget drift: {dataset}')
        paths[dataset] = str(path.resolve())
        rows[dataset] = {row['id']: row for row in items}
    return paths, rows


def _common_base(old_config, *, source_commit, protocol, protocol_path,
                 manifests, rows, targets, metadata, source):
    base = dict(old_config)
    # The old native config is inherited for identities, never reused as an
    # arm-specific request. Task budget/thinking comes from the manifest row.
    for key in ('fingerprint', 'manifest', 'manifest_sha256', 'ids', 'seeds',
                'thinking', 'max_new_tokens', 'condition', 'plugin', 'v20_scope',
                'output_mode'):
        base.pop(key, None)
    selected = {dataset: [t['id'] for t in targets if t['dataset'] == dataset]
                for dataset in old.DATASETS}
    base.update(schema='numerical_qk_native_runner_v1', phase='v21_diagnostic',
                model=str(Path(old_config['model']).resolve()), revision=old_config['revision'],
                source_commit=source_commit, model_metadata_hashes=metadata,
                source_hashes=source, manifests=manifests,
                manifest_sha256_by_dataset=protocol['generation_manifest_sha256'],
                task_contract_by_dataset={dataset: dict(thinking=dataset != 'ruler4k',
                    budgets_by_id={id_: rows[dataset][id_]['generation_budget']
                                   for id_ in dict.fromkeys(selected[dataset])})
                    for dataset in old.DATASETS},
                task_contract_source='per-target frozen manifest row',
                policy_name='P0', policy=old_config['policy'],
                policy_sha256=old.sha_json(old_config['policy']),
                consumer='triton', kernel_variant='generic',
                selector='prefix_block_summary', selector_layers='all',
                support='native_mask', support_geometry='native_legal',
                diagnostic=False, timing_events=False)
    return base


def build(old_config, *, source_commit, gpu_uuid, host_index, manifests_dir,
          protocol_path=None):
    from experiments.numerical_qk_reuse import v21
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('source commit must be exact 40-hex SHA')
    if host_index not in (0, 1) or not gpu_uuid or not gpu_uuid.startswith('GPU-'):
        raise ValueError('host index 0/1 and pinned GPU UUID required')
    if old_config.get('condition') != 'native_dense' or old_config.get('policy_name') != 'P0':
        raise ValueError('verified old P0 D_native config required')
    frozen_old = dict(old_config)
    fingerprint = frozen_old.pop('fingerprint', None)
    if fingerprint is None or fingerprint != old.sha_json(frozen_old):
        raise ValueError('old native config fingerprint drift')
    if old_config.get('revision') != old.REVISION or old_config.get('model') is None:
        raise ValueError('old native model/revision drift')
    if not isinstance(old_config.get('policy'), dict) or set(old_config['policy']) != {'local', 'global'}:
        raise ValueError('old native frozen P0 policy missing')
    if old_config.get('policy_sha256') != old.sha_json(old_config.get('policy')):
        raise ValueError('old native P0 policy identity drift')
    protocol_path = Path(protocol_path or (Path(__file__).resolve().parents[1] / 'results' /
                         'fan_m1_m3_multidataset_20260927' / 'frozen_protocol.json'))
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol.get('schema') != 'v20_fan_panel_v1' or protocol.get('planned_executions') != 700 or
            protocol.get('model_revision') != old.REVISION):
        raise ValueError('wrong frozen v20 protocol')
    protocol_hash = old.sha_bytes(protocol_bytes)
    if old_config.get('panel_protocol_sha256') not in (None, protocol_hash):
        raise ValueError('old native protocol identity drift')
    inherited_manifests = old_config.get('manifest_sha256_by_dataset')
    if inherited_manifests is not None and inherited_manifests != protocol['generation_manifest_sha256']:
        raise ValueError('old native manifest identity drift')
    manifests, rows = _manifests(protocol, manifests_dir)
    # Import the sole source of the eighteen frozen target definitions. This
    # selection uses ID order only, never answers, scores or observed timing.
    from scripts.v21_numerical_diagnostic import frozen_targets, ARM_MODES
    frozen = frozen_targets(protocol)
    targets = [dict(t) for t in frozen if protocol['ids'][t['dataset']].index(t['id']) == host_index]
    if len(targets) != 9 or len({(t['dataset'], t['id']) for t in targets}) != 3:
        raise ValueError('host assignment must contain three frozen target triples')
    metadata = old.model_hashes(old_config['model'])
    if metadata != old_config.get('model_metadata_hashes'):
        raise ValueError('old native model metadata identity drift')
    control_source = _source_inventory(old.CONTROLS, old_config)
    method_source = _source_inventory(v20.PLUGIN, old_config)
    common = dict(source_commit=source_commit, protocol=protocol,
                  protocol_path=protocol_path, manifests=manifests, rows=rows,
                  targets=targets, metadata=metadata)
    native_base = _common_base(old_config, source=control_source, **common)
    method_base = _common_base(old_config, source=method_source, **common)
    native_base['selector'] = 'legacy_recompute'
    native = dict(name='D_native', plugin=None, condition='native_dense',
                  config=old.control_config(native_base, 'native_dense', SCOPE))
    arms = [native]
    for name, (kind, precision, layout) in ARM_MODES.items():
        if kind == 'v20_control':
            config = v21.effective_control_config(native_base, SCOPE,
                output_score_precision=precision, output_layout=layout)
        else:
            config = v21.effective_config(method_base, 'M3_R3_A8_current_output', SCOPE,
                output_score_precision=precision, output_layout=layout)
        arms.append(dict(name=name, plugin=v21.PLUGIN,
                         condition=config['condition'], config=config))
    shared = dict(model=str(Path(old_config['model']).resolve()), revision=old.REVISION,
                  manifests=manifests,
                  manifest_sha256=dict(protocol['generation_manifest_sha256']),
                  panel_protocol_sha256=protocol_hash,
                  seed=101, gpu_uuid=gpu_uuid, host_index=host_index,
                  targets=targets)
    diagnostic = dict(schema='v21_numerical_diagnostic_config_v1', **shared, arms=arms)
    names = {'M3_R3_legacy': 'legacy', 'M3_R3_layout': 'layout_only',
             'M3_R3_new': 'numeric_only', 'M3_R3_combined': 'combined'}
    profile_arms = [native] + [dict(arm, name=names[arm['name']]) for arm in arms[1:]
                               if arm['name'] in names]
    profile = dict(schema='v21_profile_config_v1', **shared, scope=SCOPE,
                   reps=3, blocks=3, warmup=1,
                   boundaries=['model_forward', 'denoising_step'],
                   sequence_lengths=[4, 16], counter_twins=True,
                   operator_probe=False, prepared_support_floor=False,
                   arms=profile_arms)
    from scripts.v21_numerical_diagnostic import validate_config as validate_diagnostic
    from scripts.v21_profile import validate_config as validate_profile
    validate_diagnostic(diagnostic)
    validate_profile(profile)
    return diagnostic, profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-native-config', type=Path, required=True)
    parser.add_argument('--source-commit40', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--host-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--manifests-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out_dir
    diagnostic_path, profile_path = out / 'diagnostic.json', out / 'profile.json'
    if out.exists() or diagnostic_path.exists() or profile_path.exists():
        raise FileExistsError(f'exclusive diagnostic output directory exists: {out}')
    old_config = old.read_json(args.old_native_config)
    diagnostic, profile = build(old_config, source_commit=args.source_commit40,
                                gpu_uuid=args.gpu_uuid, host_index=args.host_index,
                                manifests_dir=args.manifests_dir)
    out.mkdir(parents=True, exist_ok=False)
    old.atomic_new(diagnostic_path, diagnostic)
    old.atomic_new(profile_path, profile)
    print(json.dumps(dict(diagnostic=str(diagnostic_path.resolve()),
                          profile=str(profile_path.resolve()), targets=9,
                          host_index=args.host_index), sort_keys=True))


if __name__ == '__main__':
    main()
