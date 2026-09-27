"""CPU-only, host-local v20 screening and later generation config binder.

Screening binds the first six frozen panel blocks (two development inputs per
dataset) to P0/P1 and both scopes. Generation emits seven configs for ONE
selected scope/policy as a host fragment; the coordinator combines host
fragments and freezes the final binding only after numeric/cost qualification.
No model is loaded, no answers are read, and no GPU work is launched here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from experiments.numerical_qk_reuse import v20


DATASETS = ('ruler4k', 'aime26', 'longbench_v2')
POLICY_POINTS = {'P0': 'T50', 'P1': 'T60'}
CONTROLS = 'experiments.numerical_qk_reuse.v20_controls:install'
V20_METHODS = ('M1_R1_A8_current_output', 'M3_R2_A8_current_output',
               'M3_R3_A8_current_output', 'B_A8_matched')
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
M_REF, BETA, GAMMA = 14.258454322814941, 3., .5
GOLD = frozenset(('expected', 'expected_answer', 'answer', 'reference_solution',
                  'solution', 'gold', 'target'))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_frozen(protocol_path, manifests_dir, calibration_path):
    protocol = read_json(protocol_path)
    if protocol.get('schema') != 'v20_fan_panel_v1' or protocol.get('planned_executions') != 700:
        raise ValueError('wrong frozen panel')
    if protocol.get('model_revision') != REVISION:
        raise ValueError('model revision drift')
    calibration = read_json(calibration_path)
    policies = {point: calibration[name]['policy'] for point, name in POLICY_POINTS.items()}
    if any(set(policy) != {'local', 'global'} for policy in policies.values()):
        raise ValueError('frozen T policy must have local/global thresholds')
    rows = {}
    paths = {}
    for dataset in DATASETS:
        path = Path(manifests_dir) / f'{dataset}_generation_manifest.json'
        raw = path.read_bytes()
        if sha_bytes(raw) != protocol['generation_manifest_sha256'][dataset]:
            raise ValueError(f'generation manifest hash drift: {dataset}')
        items = json.loads(raw)
        if {r['id'] for r in items} != set(protocol['ids'][dataset]) or len(items) != len(protocol['ids'][dataset]):
            raise ValueError(f'manifest ID drift: {dataset}')
        for row in items:
            if set(row) & GOLD or sha_bytes(row['prompt'].encode()) != row.get('prompt_hash'):
                raise ValueError('gold or prompt identity drift')
            if row.get('thinking') is not (dataset != 'ruler4k'):
                raise ValueError('thinking contract drift')
            if type(row.get('generation_budget')) is not int or not 1 <= row['generation_budget'] <= 8192:
                raise ValueError('generation budget drift')
            if dataset != 'ruler4k' and row['generation_budget'] != 8192:
                raise ValueError('AIME/LB output budget drift')
        rows[dataset] = {row['id']: row for row in items}
        paths[dataset] = path
    first = [protocol['block_assignments'][str(i)] for i in range(6)]
    chosen = {dataset: [r['id'] for r in first if r['dataset'] == dataset] for dataset in DATASETS}
    if any(len(set(ids)) != 2 or len(ids) != 2 for ids in chosen.values()) or any(r['seed'] != 101 for r in first):
        raise ValueError('first84 must freeze two distinct seed101 IDs per dataset')
    return protocol, policies, rows, paths, chosen


def model_hashes(model):
    path = Path(model)
    if not path.is_dir():
        raise FileNotFoundError(f'model snapshot absent: {path}')
    names = ('config.json', 'generation_config.json', 'tokenizer_config.json',
             'preprocessor_config.json', 'model.safetensors.index.json')
    result = {name: sha_bytes((path/name).read_bytes()) for name in names if (path/name).is_file()}
    if not {'config.json', 'generation_config.json'} <= result.keys():
        raise ValueError('model config/generation metadata absent')
    return result


def source_hashes(plugin, library, torch_library, support_build):
    """Use the established source inventory plus explicit support binaries."""
    from experiments.numerical_qk_reuse.runner import _source_hashes
    extras = [Path(__file__), Path(__file__).with_name('v20_profile.py'),
              Path(__file__).with_name('v20_run.py')]
    if support_build:
        identity = read_json(support_build)
        for field in ('kernel', 'bridge'):
            path = Path(identity[field])
            if sha_bytes(path.read_bytes()) != identity[f'{field}_sha256']:
                raise ValueError(f'support {field} identity drift')
            extras.append(path)
        extras.append(Path(support_build))
    return _source_hashes(plugin, library=Path(library) if library else None,
                          torch_library=Path(torch_library) if torch_library else None,
                          extra_sources=extras)


def base_config(dataset, *, protocol, manifest_path, model, metadata, source, policy,
                policy_point, library, torch_library, support_build, consumer, phase,
                selected_ids, timing_events, source_commit):
    if consumer not in ('triton', 'hopper') or (consumer == 'hopper' and not support_build):
        raise ValueError('qualified consumer/support build required')
    if not library or not torch_library:
        raise ValueError('matching Junyu library and torch library required')
    config = dict(schema='numerical_qk_native_runner_v1', phase=phase,
                  ids=list(selected_ids), seeds=[101, 202] if phase == 'v20_generation' else [101],
                  manifest=str(Path(manifest_path).resolve()),
                  manifest_sha256=protocol['generation_manifest_sha256'][dataset],
                  policy_name=policy_point, policy=policy, policy_sha256=sha_json(policy),
                  m_ref=M_REF, beta=BETA, gamma=GAMMA,
                  model=str(Path(model).resolve()), revision=REVISION, dtype='bfloat16',
                  source_commit=source_commit,
                  model_metadata_hashes=metadata, source_hashes=source,
                  max_new_tokens=max(128 if dataset == 'ruler4k' else 8192, 1),
                  thinking=dataset != 'ruler4k', temperature=0., eos_enabled=True,
                  native_adaptive=True, diagnostic=False, timing_events=timing_events,
                  library=str(Path(library).resolve()), torch_library=str(Path(torch_library).resolve()),
                  support_build=str(Path(support_build).resolve()) if support_build else None,
                  consumer=consumer, support_geometry='native_legal',
                  selector='legacy_recompute', selector_layers='all',
                  kernel_variant='generic', telemetry='minimal', guard_mode='fused')
    return config


def control_config(base, condition, scope):
    config = dict(base, condition=condition, plugin=(None if condition == 'native_dense' else CONTROLS),
                  v20_scope=scope if condition != 'native_dense' else None,
                  output_mode=v20.PREQK_MODE if condition != 'native_dense' else None)
    config['fingerprint'] = sha_json(config)
    return config


def method_config(base, method, scope):
    selector = 'legacy_recompute' if method == 'B_A8_matched' else 'prefix_block_summary'
    return v20.effective_config(dict(base, selector=selector, selector_layers='all'), method, scope)


def capture_targets(chosen, rows):
    targets = []
    for dataset in DATASETS:
        for id_ in chosen[dataset]:
            row = rows[dataset][id_]
            # Metadata is recorded without prompt or task answer.
            common = dict(id=id_, dataset=dataset, prompt_hash=row['prompt_hash'],
                          prompt_token_count=row['prompt_token_count'])
            targets.append(dict(common, canvas=0, call_index=0))
            targets.append(dict(common, canvas=(0 if dataset == 'ruler4k' else 8 if dataset == 'aime26' else 4),
                                call_index=2, fallback_call_index=1))
    return targets


def screen_config(protocol, policies, rows, manifests, chosen, host_args, identity):
    if host_args.consumer != 'hopper':
        raise ValueError('screen primary consumer must be Hopper; Triton diagnostics are explicit')
    gpu_uuid = None
    if host_args.host:
        first = [protocol['block_assignments'][str(i)] for i in range(6)]
        chosen = {dataset: [entry['id'] for entry in first
                            if entry['dataset'] == dataset and entry['host'] == host_args.host]
                  for dataset in DATASETS}
        uuids = {entry['gpu_uuid'] for entry in first if entry['host'] == host_args.host}
        if any(len(ids) != 1 for ids in chosen.values()) or len(uuids) != 1:
            raise ValueError('screen host must own exactly one initial input per dataset on one GPU')
        gpu_uuid = next(iter(uuids))
        if host_args.gpu_uuid and host_args.gpu_uuid != gpu_uuid:
            raise ValueError('screen host/GPU assignment drift')
    arms = [dict(name='D_native', plugin=None, condition='native_dense', config={})]
    for scope in v20.SCOPES:
        # Screen uses one arm config per scope/policy with the same thresholds;
        # task metadata remains in each target's manifest row.
        base = base_config('aime26', protocol=protocol, manifest_path=manifests['aime26'],
                           model=host_args.model, metadata=identity['model'], source=identity['control'],
                           policy=policies['P0'], policy_point='P0', library=host_args.library,
                           torch_library=host_args.torch_library, support_build=host_args.support_build,
                           consumer=host_args.consumer, phase='v20_screen', selected_ids=[],
                           timing_events=False, source_commit=host_args.source_commit)
        base.update(manifests={d: str(manifests[d].resolve()) for d in DATASETS},
                    manifest_sha256_by_dataset={d: protocol['generation_manifest_sha256'][d] for d in DATASETS},
                    task_contract_by_dataset={d: dict(thinking=d != 'ruler4k',
                                                      budgets_by_id={id_: rows[d][id_]['generation_budget']
                                                                     for id_ in chosen[d]}) for d in DATASETS},
                    task_contract_source='per-target manifest row')
        base.pop('manifest', None)
        base.pop('manifest_sha256', None)
        base.pop('thinking', None)
        base.pop('max_new_tokens', None)
        for name, condition in (('D_matched', 'v20_dense_consumer'), ('T_scope', 'v20_fresh_T')):
            arms.append(dict(name=f'{scope}_{name}', plugin=CONTROLS, condition=condition,
                             config=control_config(base, condition, scope)))
        if scope == v20.ALL_NATIVE_LEGAL:
            arms.append(dict(name='G75L30_nativeQ128', plugin=CONTROLS,
                             condition='v20_G75L30_nativeQ128',
                             config=control_config(base, 'v20_G75L30_nativeQ128', scope)))
        for point, methods in (('P0', V20_METHODS), ('P1', V20_METHODS)):
            method_base = dict(base, policy=policies[point], policy_name=point,
                               policy_sha256=sha_json(policies[point]),
                               source_hashes=identity['method'])
            for method in methods:
                config = method_config(method_base, method, scope)
                arms.append(dict(name=f'{scope}_{point}_{method}', plugin=v20.PLUGIN,
                                 condition=config['condition'], config=config))
        for method in ('M3_R2_A8_current_output', 'B_A8_matched'):
            triton_base = dict(base, policy=policies['P0'], policy_name='P0',
                               policy_sha256=sha_json(policies['P0']),
                               consumer='triton', support_build=None,
                               source_hashes=identity['method'])
            config = method_config(triton_base, method, scope)
            arms.append(dict(name=f'{scope}_P0_triton_{method}', plugin=v20.PLUGIN,
                             condition=config['condition'], config=config,
                             diagnostic_consumer_alternative=True))
    return dict(schema='v20_profile_config_v1', model=str(Path(host_args.model).resolve()),
                revision=REVISION, manifests={d: str(manifests[d].resolve()) for d in DATASETS},
                host=host_args.host, gpu_uuid=gpu_uuid,
                manifest_sha256={d: protocol['generation_manifest_sha256'][d] for d in DATASETS},
                panel_protocol_sha256=host_args.protocol_sha256,
                calibration_sha256=host_args.calibration_sha256,
                seed=101, reps=3, blocks=3, warmup=1,
                boundaries=['model_forward'], sequence_lengths=[4],
                counter_twins=True, prepared_support_floor=False,
                targets=capture_targets(chosen, rows), arms=arms)


def generation_configs(protocol, policies, manifests, host_args, identity, scope, point):
    if scope not in v20.SCOPES or point not in POLICY_POINTS:
        raise ValueError('explicit qualified scope and policy point required')
    result = {}
    for dataset in DATASETS:
        base = base_config(dataset, protocol=protocol, manifest_path=manifests[dataset],
                           model=host_args.model, metadata=identity['model'], source=identity['control'],
                           policy=policies[point], policy_point=point, library=host_args.library,
                           torch_library=host_args.torch_library, support_build=host_args.support_build,
                           consumer=host_args.consumer, phase='v20_generation',
                           selected_ids=protocol['ids'][dataset], timing_events=True,
                           source_commit=host_args.source_commit)
        family = {'D_native': control_config(base, 'native_dense', scope),
                  'D_matched': control_config(base, 'v20_dense_consumer', scope),
                  'T_scope': control_config(base, 'v20_fresh_T', scope)}
        method_base = dict(base, source_hashes=identity['method'])
        family.update({method: method_config(method_base, method, scope) for method in V20_METHODS})
        historical_base = dict(base, source_hashes=identity['control'])
        family['G75L30_nativeQ128'] = control_config(historical_base,
                                                      'v20_G75L30_nativeQ128',
                                                      v20.ALL_NATIVE_LEGAL)
        result[dataset] = family
    return result


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.writing')
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with tmp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def bind(args):
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        raise ValueError('source-commit must be the exact 40-hex deployed commit')
    protocol, policies, rows, manifests, chosen = read_frozen(args.protocol, args.manifests_dir, args.calibration)
    args.protocol_sha256 = sha_bytes(Path(args.protocol).read_bytes())
    args.calibration_sha256 = sha_bytes(Path(args.calibration).read_bytes())
    metadata = model_hashes(args.model)
    identity = dict(model=metadata,
                    control=source_hashes(CONTROLS, args.library, args.torch_library, args.support_build),
                    method=source_hashes(v20.PLUGIN, args.library, args.torch_library, args.support_build))
    if args.mode == 'screen':
        atomic_new(args.out, screen_config(protocol, policies, rows, manifests, chosen, args, identity))
        return
    if not args.host or not args.scope or not args.policy_point:
        raise ValueError('generation requires host, scope, policy-point')
    host_uuids = {a['gpu_uuid'] for a in protocol['block_assignments'].values()
                  if a['host'] == args.host}
    if len(host_uuids) != 1 or (args.gpu_uuid and args.gpu_uuid not in host_uuids):
        raise ValueError('generation host/GPU assignment drift')
    configs = generation_configs(protocol, policies, manifests, args, identity, args.scope, args.policy_point)
    config_dir = args.out.with_name(args.out.stem + '_configs')
    if args.out.exists() or config_dir.exists():
        raise FileExistsError('generation binding/config output already exists')
    paths = {}
    for dataset, family in configs.items():
        paths[dataset] = {}
        for name, config in family.items():
            path = config_dir / dataset / f'{name}.json'
            atomic_new(path, config)
            paths[dataset][name] = dict(path=str(path.resolve()), sha256=sha_bytes(path.read_bytes()))
    fragment = dict(schema='v20_host_binding_fragment_v1', status='host_fragment',
                    panel_protocol_sha256=args.protocol_sha256, scope=args.scope,
                    policy_point=args.policy_point, policy_sha256=sha_json(policies[args.policy_point]),
                    host_models={args.host: str(Path(args.model).resolve())},
                    host_configs={args.host: paths}, historical_qualified=False,
                    historical_scope=v20.ALL_NATIVE_LEGAL,
                    historical_extension=dict(arm='G75L30_nativeQ128',
                                              scope=v20.ALL_NATIVE_LEGAL,
                                              qualified=False,
                                              note='separate native-Q128 port; outside main seven-arm scope'),
                    calibration_sha256=args.calibration_sha256,
                    source_commit=args.source_commit)
    atomic_new(args.out, fragment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('screen', 'generation'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifests-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--torch-library', type=Path, required=True)
    parser.add_argument('--support-build', type=Path)
    parser.add_argument('--consumer', choices=('triton', 'hopper'), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--host')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--scope', choices=v20.SCOPES)
    parser.add_argument('--policy-point', choices=tuple(POLICY_POINTS))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    bind(args)


if __name__ == '__main__':
    main()
