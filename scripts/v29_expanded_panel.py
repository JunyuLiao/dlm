"""Freeze and run expanded vLLM panels; prompts, gold and bindings stay private.

The GPU loop is v27 unchanged. V28 supplies allocator/KV/JIT instrumentation.
This module never resets per-request RNG and never launches additional workers.
Separate protocols allow LongBench, AIME and HumanEval to use different hosts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import re
import time
from copy import deepcopy
from scripts.v27_vllm_bind import fingerprint

from scripts import v27_vllm_panel_run as panel
from scripts import v27_vllm_panel_summary as old
from scripts import v28_vllm_qualify as qualify
from scripts.v27_vllm_bind import _validate_fingerprints, _method_fields

COUNTS = {'longbench_v2_32k': 24, 'longbench_v2_64k': 24,
          'longbench_v2_96k': 11, 'aime26': 30, 'humaneval': 164}
SUITES = {'longbench': tuple(COUNTS)[:3], 'aime': ('aime26',), 'humaneval': ('humaneval',)}
ARMS = ('dense', 'native', 'allkept', 'method')
METHOD = dict(score_period=64, decision_interval=6, risk_state='dense_prefix',
              carry_first=True, fused_observe=True, async_route=True, consumer='fa4',
              threshold_shift='minus_ln2', scope='GLOBAL_ONLY_NATIVE_LOCAL', sensitivity='temporal_T',
              q_block=128, q_regroup=False, q_carry64=False,
              min_route_keys=0, proj_rank=32, observe_carried=False)
GOLD_KEYS = frozenset(('answer', 'expected', 'expected_answer', 'gold', 'solution',
                       'reference_solution', 'outputs', 'test', 'entry_point'))
SOURCES = ('scripts/v29_expanded_panel.py', 'scripts/v29_expanded_summary.py',
           'scripts/v27_vllm_panel_run.py', 'scripts/v27_vllm_metrics.py',
           'scripts/v27_vllm_panel_summary.py', 'scripts/v27_vllm_bind.py',
           'scripts/v28_vllm_qualify.py', 'scripts/v28_jit_receipts.py',
           'scripts/v15_longbench_task.py', 'scripts/v27_humaneval.py',
           'experiments/numerical_qk_reuse/vllm_adapter.py',
           'experiments/numerical_qk_reuse/v29_paged_copy.py',
           'experiments/numerical_qk_reuse/v29_lse_merge.py',
           'experiments/diffusion_gemma_aime26_modes/protocol.py',
           'experiments/diffusion_gemma_aime30/protocol.py',
           'experiments/diffusion_gemma_solattn_blasst_multibench/runner.py')


def task_contract(dataset):
    if dataset.startswith('longbench_v2_'):
        return dict(task='longbench_v2', metric='strict_correct_eos', budget=8192,
                    scorer='v15-lbv2-nemo-mcq-default-on-final-channel-1')
    if dataset == 'aime26':
        return dict(task='aime26', metric='strict_correct_eos', budget=8192,
                    scorer='existing_v20_final_response_numeric_score', thinking=True)
    if dataset == 'humaneval':
        return dict(task='humaneval', metric='test_pass_at_1_eos_not_required', budget=8192,
                    scorer='v27_complete_function_final_channel_bwrap_official_tests')
    raise ValueError('unsupported dataset')


def build_spec(suite, name, seeds, hosts):
    if suite not in SUITES or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('invalid suite/name')
    if len(seeds) != 8 or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 8:
        raise ValueError('exactly eight distinct nonnegative engine seeds required')
    if not hosts or any(not re.fullmatch(r'[A-Za-z0-9_-]+', host) for host in hosts):
        raise ValueError('host aliases required; never pass addresses')
    datasets = SUITES[suite]
    spec = dict(schema='v29_expanded_panel_v1', name=name, protocol_id='v29_' + name,
                suite=suite, arms=['dense', 'method'], controls=['native', 'allkept'],
                datasets={ds: dict(indices=list(range(COUNTS[ds])), repeats=list(range(8))) for ds in datasets},
                control_indices={ds: list(range(COUNTS[ds])) for ds in datasets},
                blocks=[dict(engine_seed=seed, repeats=[block], host=hosts[block % len(hosts)],
                             arm_order=list(ARMS[block % 4:] + ARMS[:block % 4]))
                        for block, seed in enumerate(seeds)],
                settings=dict(max_model_len=106496 if suite == 'longbench' else 16384,
                              chunk=16384, gpu_memory_utilization=.85, block_size=32, cpu_threads=1),
                arm_settings={arm: dict(compilation_config='default' if arm == 'dense' else 'PIECEWISE',
                                         cudagraph_mode='default' if arm == 'dense' else 'PIECEWISE') for arm in ARMS},
                adapter_settings=dict(lifecycle='request_clear', alias_splits=2, canvas_buffers='legacy',
                                      kv_copy_backend='torch', merge_backend='torch'),
                primary_receipt_method=dict(METHOD), task_contracts={ds: task_contract(ds) for ds in datasets},
                order_seed=2026100209, jit_event_receipts=True,
                qualification='All arms per dataset: longest frozen input, one warm and one timed request; score before formal launch.',
                accuracy_analysis=dict(primary='Observed task difference and question-cluster 95% CI',
                                       noninferiority_margin=None, noninferiority_claim_authorized=False),
                pairing='dataset/index/repeat label; engine seed applies at load, not per request; no identical-trajectory claim',
                warmup='Every frozen item once per fresh engine before any timed requests')
    validate_spec(spec)
    return spec


def validate_spec(spec):
    if spec.get('schema') != 'v29_expanded_panel_v1' or not str(spec.get('protocol_id', '')).startswith('v29_'):
        raise ValueError('a new v29 protocol is required')
    old.expected_inventory(spec)
    if spec.get('suite') not in SUITES or set(spec['datasets']) != set(SUITES[spec['suite']]):
        raise ValueError('suite inventory differs')
    blocks = spec.get('blocks', [])
    if len(blocks) != 8 or len({block['engine_seed'] for block in blocks}) != 8:
        raise ValueError('eight distinct engine seed blocks required')
    for index, block in enumerate(blocks):
        if block.get('repeats') != [index] or not isinstance(block.get('host'), str) or not block['host']:
            raise ValueError('one unique repeat label and a host per block required')
        if sorted(block.get('arm_order', [])) != sorted(ARMS):
            raise ValueError('each block must counterbalance all four arms')
    for ds, value in spec['datasets'].items():
        if value != dict(indices=list(range(COUNTS[ds])), repeats=list(range(8))):
            raise ValueError('full question/seed inventory required; no subsampling')
        if spec['control_indices'][ds] != value['indices']:
            raise ValueError('all controls must cover every question')
    if spec.get('task_contracts') != {ds: task_contract(ds) for ds in spec['datasets']}:
        raise ValueError('task metric/scorer/budget contract drift')
    if spec.get('primary_receipt_method') != METHOD:
        raise ValueError('frozen q128 main receipt contract changed')
    adapter = spec.get('adapter_settings', {})
    if (set(adapter) != {'lifecycle', 'alias_splits', 'canvas_buffers', 'kv_copy_backend', 'merge_backend'} or
            adapter['lifecycle'] != 'request_clear' or adapter['alias_splits'] != 2 or
            adapter['canvas_buffers'] not in ('legacy', 'release_after_invalidate') or
            adapter['kv_copy_backend'] not in ('torch', 'triton') or
            adapter['merge_backend'] not in ('torch', 'triton')):
        raise ValueError('unknown or unmatched adapter standard optimization')
    if spec.get('jit_event_receipts') is not True or type(spec['settings'].get('cpu_threads')) is not int or spec['settings']['cpu_threads'] != 1:
        raise ValueError('known JIT monitor and OMP1 required')
    analysis = spec.get('accuracy_analysis', {})
    if analysis.get('noninferiority_margin') is not None or analysis.get('noninferiority_claim_authorized') is not False:
        raise ValueError('this draft has no user-approved noninferiority decision rule')
    return spec


def execution_settings(spec):
    adapter = spec['adapter_settings']
    value = qualify.execution_settings(128, adapter['lifecycle'], adapter['canvas_buffers'])
    value.update(jit_monitor_events=True, kv_copy_backend=adapter['kv_copy_backend'],
                 merge_backend=adapter['merge_backend'])
    return value


def validate_manifest(rows, dataset, selected_ids, budget):
    if not isinstance(rows, list) or not isinstance(selected_ids, list) or len(selected_ids) != COUNTS[dataset]:
        raise ValueError('manifest/selected full inventory differs')
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in by_id:
            raise ValueError('manifest contains duplicate or invalid identity')
        by_id[row['id']] = row
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(by_id):
        raise ValueError('selected private identities are missing or duplicated')
    clean = []
    for identity in selected_ids:
        row = by_id[identity]
        tokens = row.get('prompt_tokens')
        if (not isinstance(tokens, list) or not tokens or
                any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError('a frozen nonempty tokenized prompt is required')
        if row.get('generation_budget') != budget or row.get('thinking', True) is not True:
            raise ValueError('generation budget/thinking differs from existing task contract')
        clean.append({key: value for key, value in row.items() if key not in GOLD_KEYS})
    if dataset == 'aime26' and set(selected_ids) != {f'aime26/{i}' for i in range(1, 31)}:
        raise ValueError('complete existing AIME30 identity set required')
    if dataset == 'humaneval' and set(selected_ids) != {f'humaneval/{i}' for i in range(164)}:
        raise ValueError('complete official HumanEval164 identity set required')
    return clean


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze_private(spec_path, catalog_path, config_path, model_path, deploy_path, host, gpu_uuid, out_dir):
    """Run only after source/spec commits are deployed; no model loading occurs."""
    spec = validate_spec(panel.read(spec_path))
    deploy, model = Path(deploy_path).resolve(), Path(model_path).absolute()
    config = panel.read(config_path)
    _validate_fingerprints(config)
    defaults = dict(q_block=128, q_regroup=False, q_carry64=False,
                    min_route_keys=0, proj_rank=32, observe_carried=False)
    if any(config.get(key, default) != default for key, default in defaults.items()):
        raise ValueError('q128 main config required')
    sha = (deploy / 'DEPLOY_SHA').read_text().strip()
    if not re.fullmatch('[0-9a-f]{40}', sha) or host not in {block['host'] for block in spec['blocks']}:
        raise ValueError('committed deployment/assigned host required')
    catalog = panel.read(catalog_path)
    if set(catalog) != set(spec['datasets']):
        raise ValueError('private catalog inventory differs')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'spec.json', spec)
    write_new(out / 'config.json', config)
    manifests, golds, gold_sha256, contracts, cells = {}, {}, {}, {}, []
    files = {}
    for dataset, entry in catalog.items():
        rows = panel.read(entry['manifest'])
        ids = entry.get('selected_ids', [row['id'] for row in rows])
        rows = validate_manifest(rows, dataset, ids, spec['task_contracts'][dataset]['budget'])
        if any(len(row['prompt_tokens']) + row['generation_budget'] > spec['settings']['max_model_len'] for row in rows):
            raise ValueError('frozen prompt plus complete output budget exceeds model length')
        path = out / (dataset + '_generation_manifest.json')
        write_new(path, rows)
        manifests[dataset] = str(path.resolve())
        golds[dataset] = str(Path(entry['gold']).resolve())
        # Gold remains scorer-only; not copied into the generation manifest.
        gold_path = Path(golds[dataset])
        expected_gold = entry.get('gold_sha256')
        if gold_path.is_file():
            actual_gold = panel.digest(gold_path)
            if expected_gold is not None and expected_gold != actual_gold:
                raise ValueError('scorer-only gold pin differs')
            expected_gold = actual_gold
        if not isinstance(expected_gold, str) or not re.fullmatch('[0-9a-f]{64}', expected_gold):
            raise ValueError('scorer-only gold byte pin required; never transport gold to a GPU worker')
        gold_sha256[dataset] = expected_gold
        if dataset == 'humaneval':
            contract_path = str(Path(entry['task_contract']).resolve())
            from scripts import v27_humaneval as humaneval
            contract = panel.read(contract_path)
            humaneval.validate_contract(contract)
            if contract['scorer_sha256'] != panel.digest(deploy / 'scripts/v27_humaneval.py'):
                raise ValueError('HumanEval existing scorer bytes changed')
            contracts[dataset] = contract_path
            files[contract_path] = panel.digest(contract_path)
        cells.extend(dict(dataset=dataset, index=index, id=identity) for index, identity in enumerate(ids))
    write_new(out / 'cells.private.json', cells)
    source_files = {relative: panel.digest(deploy / relative) for relative in SOURCES}
    source_file_paths = {relative: str((deploy / relative).resolve(strict=True)) for relative in SOURCES}
    model_config_files = {name: str((model / name).resolve(strict=True))
                          for name in ('config.json', 'generation_config.json')}
    paths = [out / 'spec.json', out / 'config.json', out / 'cells.private.json',
             *map(Path, manifests.values()), *map(Path, model_config_files.values())]
    files.update({str(path.resolve()): panel.digest(path) for path in paths})
    files.update({source_file_paths[relative]: value for relative, value in source_files.items()})
    config_source_files = {}
    def pin_config_sources(value):
        if isinstance(value, list):
            for child in value:pin_config_sources(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key != 'source_hashes':
                    pin_config_sources(child)
                else:
                    if not isinstance(child, dict):
                        raise ValueError('config source hash dictionary required')
                    for original, digest in child.items():
                        path = Path(original).resolve(strict=True)
                        if panel.digest(path) != digest or str(path) in files and files[str(path)] != digest:
                            raise ValueError('config source identity drift')
                        files[str(path)] = digest
                        config_source_files[original] = str(path)
    pin_config_sources(config)
    binding = dict(schema='v29_expanded_binding_v1', protocol_id=spec['protocol_id'], deploy_commit=sha,
                   deploy=str(deploy), host=host, gpu_uuid=gpu_uuid, model=str(model),
                   model_config_files=model_config_files,
                   spec=str((out / 'spec.json').resolve()), config=str((out / 'config.json').resolve()),
                   cells=str((out / 'cells.private.json').resolve()), manifests=manifests,
                   golds=golds, gold_sha256=gold_sha256, task_contracts=contracts, files=files, source_files=source_files,
                   source_file_paths=source_file_paths, config_source_files=config_source_files)
    write_new(out / 'binding.private.json', binding)
    return dict(protocol_id=spec['protocol_id'], question_counts={ds: len(spec['datasets'][ds]['indices']) for ds in spec['datasets']},
                engine_seed_blocks=8, timed_requests=4*8*len(cells), warm_requests=4*8*len(cells))


class FrozenArtifacts:
    """Explicit scorer-only byte mirrors; original binding and identities survive."""
    def __init__(self, binding_path, artifact_map=None, binding_sha256=None):
        self.binding = panel.read(binding_path)
        self.mapping = None
        if artifact_map is not None:
            if not isinstance(binding_sha256, str) or panel.digest(binding_path) != binding_sha256:
                raise ValueError('scorer mirror requires the original binding byte pin')
            expected = set(self.binding['files']) | {str(Path(self.binding['deploy']) / 'DEPLOY_SHA')}
            if not isinstance(artifact_map, dict) or set(artifact_map) != expected:
                raise ValueError('scorer mirror must map every original artifact, and no unknown artifact')
            resolved = [str(Path(value).resolve(strict=True)) for value in artifact_map.values()]
            if len(set(resolved)) != len(resolved):
                raise ValueError('ambiguous scorer artifact aliases')
            self.mapping = dict(zip(artifact_map, resolved))
            for original, digest in self.binding['files'].items():
                if panel.digest(self.local(original)) != digest:
                    raise ValueError('scorer mirror artifact byte drift')

    def local(self, original):
        if self.mapping is None:
            return Path(original)
        try:
            return Path(self.mapping[str(original)])
        except KeyError as exc:
            raise ValueError('unmapped original frozen artifact') from exc

    def read(self, original):
        return panel.read(self.local(original))


def read_frozen(path, artifacts=None):
    artifacts = artifacts or FrozenArtifacts(path)
    binding = artifacts.binding
    if binding.get('schema') != 'v29_expanded_binding_v1':
        raise ValueError('v29 binding required')
    spec = validate_spec(artifacts.read(binding['spec']))
    if binding.get('protocol_id') != spec['protocol_id']:
        raise ValueError('binding protocol differs')
    # Origin paths already came from the generation host. Never resolve that
    # host's symlinks on a CPU scorer which may have no origin filesystem.
    deploy = Path(binding['deploy'])
    if artifacts.local(deploy / 'DEPLOY_SHA').read_text().strip() != binding['deploy_commit']:
        raise ValueError('deployment commit drift')
    pins = {str(Path(path)): digest for path, digest in binding['files'].items()}
    if len(pins) != len(binding['files']) or any(not Path(path).is_absolute() for path in pins):
        raise ValueError('ambiguous or relative canonical frozen artifact path')
    for path, digest in pins.items():
        if panel.digest(artifacts.local(path)) != digest:
            raise ValueError('frozen input/source/gold byte drift')
    required = [binding[key] for key in ('spec', 'config', 'cells')] + list(binding['manifests'].values())
    required += list(binding['task_contracts'].values())
    model_files = binding.get('model_config_files')
    if (not isinstance(model_files, dict) or set(model_files) != {'config.json', 'generation_config.json'} or
            any(not isinstance(value, str) or not Path(value).is_absolute() for value in model_files.values()) or
            len(set(model_files.values())) != 2):
        raise ValueError('exact canonical model config file pair required')
    required += list(model_files.values())
    source_paths = binding.get('source_file_paths')
    if not isinstance(source_paths, dict) or set(source_paths) != set(SOURCES):
        raise ValueError('exact canonical source path set required')
    required += list(source_paths.values())
    if any(str(Path(path)) not in pins for path in required):
        raise ValueError('binding omits a required source/input pin')
    if binding.get('source_files') != {relative: pins[source_paths[relative]] for relative in SOURCES}:
        raise ValueError('relative source identity differs')
    if artifacts.mapping is None:
        for name, canonical in model_files.items():
            origin = Path(binding['model']) / name
            if str(origin.resolve(strict=True)) != canonical or panel.digest(origin) != pins[canonical]:
                raise ValueError('generation model config alias/bytes drift')
        for relative, canonical in source_paths.items():
            if str((deploy / relative).resolve(strict=True)) != canonical:
                raise ValueError('generation source alias drift')
    if set(binding['manifests']) != set(spec['datasets']) or set(binding['golds']) != set(spec['datasets']):
        raise ValueError('binding dataset/gold inventory differs')
    if set(binding.get('gold_sha256', {})) != set(spec['datasets']) or any(
            not re.fullmatch('[0-9a-f]{64}', value) for value in binding['gold_sha256'].values()):
        raise ValueError('scorer-only gold pins required')
    config = artifacts.read(binding['config'])
    _validate_fingerprints(config)
    config_sources = binding.get('config_source_files')
    if not isinstance(config_sources, dict):
        raise ValueError('canonical nested config source paths required')
    used_config_sources = set()
    def require_config_sources(value):
        if isinstance(value, list):
            for child in value:require_config_sources(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key != 'source_hashes':
                    require_config_sources(child)
                else:
                    if not isinstance(child, dict):
                        raise ValueError('config source hash dictionary required')
                    for source, digest in child.items():
                        canonical = config_sources.get(source)
                        if (not Path(source).is_absolute() or not isinstance(canonical, str) or
                                pins.get(canonical) != digest):
                            raise ValueError('binding omits a nested config source pin')
                        if artifacts.mapping is None and str(Path(source).resolve(strict=True)) != canonical:
                            raise ValueError('generation nested config source alias drift')
                        used_config_sources.add(source)
    require_config_sources(config)
    if used_config_sources != set(config_sources):
        raise ValueError('unknown nested config source mapping')
    cells = artifacts.read(binding['cells'])
    expected = {(ds, index) for ds, value in spec['datasets'].items() for index in value['indices']}
    if len(cells) != len(expected) or {(cell['dataset'], cell['index']) for cell in cells} != expected:
        raise ValueError('frozen cell inventory differs')
    for dataset, manifest in binding['manifests'].items():
        identities = [cell['id'] for cell in sorted(cells, key=lambda cell: cell['index']) if cell['dataset'] == dataset]
        validate_manifest(artifacts.read(manifest), dataset, identities, spec['task_contracts'][dataset]['budget'])
    return binding, spec, config


def rebind_config_from_mirror(config, old_root, mirror_root, new_root):
    """Validate old pinned source mirrors and destination bytes; change paths only.

    Run on the destination host. No directory at the old absolute path is needed.
    External source paths are retained and checked locally, never silently remapped.
    """
    _validate_fingerprints(config)
    if not isinstance(config, dict) or 'fingerprint' not in config:
        raise ValueError('fingerprinted original config required')
    old_root = Path(old_root)
    mirror_root, new_root = Path(mirror_root).resolve(strict=True), Path(new_root).resolve(strict=True)
    if not old_root.is_absolute() or not mirror_root.is_dir() or not new_root.is_dir():
        raise ValueError('absolute old root and existing mirror/destination roots required')
    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, child in value.items():
            if key == 'fingerprint':
                continue
            if key != 'source_hashes':
                result[key] = rewrite(child)
                continue
            if not isinstance(child, dict):
                raise ValueError('source hash dictionary required')
            rebound = {}
            for original, digest in child.items():
                source = Path(original)
                if not source.is_absolute() or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    raise ValueError('absolute source path and original byte pin required')
                try:
                    relative = source.relative_to(old_root)
                except ValueError:
                    destination = source
                    candidates = [source]
                else:
                    if '..' in relative.parts:
                        raise ValueError('source escapes original root')
                    destination = (new_root / relative).resolve(strict=True)
                    mirrored = (mirror_root / relative).resolve(strict=True)
                    if not destination.is_relative_to(new_root) or not mirrored.is_relative_to(mirror_root):
                        raise ValueError('source mirror/destination symlink escape')
                    candidates = [mirrored, destination]
                if any(panel.digest(candidate) != digest for candidate in candidates):
                    raise ValueError('original mirror or destination source byte drift')
                if str(destination) in rebound:
                    raise ValueError('source path collision')
                rebound[str(destination)] = digest
            result[key] = rebound
        if 'fingerprint' in value:
            result['fingerprint'] = fingerprint(result)
        return result
    result = rewrite(deepcopy(config))
    if _method_fields(result) != _method_fields(config):
        raise ValueError('rebinding changed non-binding method fields')
    _validate_fingerprints(result)
    return result


def validate_receipt(arm, receipts, n, required, settings, original=panel.validate_receipts):
    qualify.validate_variant_receipt(arm, receipts, n, required, settings, original)
    if arm == 'dense':
        return
    adapter = receipts['adapter']
    if settings['kv_copy_backend'] == 'torch' and any(adapter.get(key, 0) != 0
            for key in ('triton_kv_copy_calls', 'triton_kv_copy_elements')):
        raise ValueError('Triton KV copy ran under frozen torch backend')
    if settings['merge_backend'] == 'triton' and arm in ('method', 'allkept') and (
            type(adapter.get('triton_lse_merge_calls')) is not int or adapter['triton_lse_merge_calls'] <= 0):
        raise ValueError('frozen Triton LSE merge path did not execute')
    if settings['merge_backend'] == 'torch' and adapter.get('triton_lse_merge_calls', 0) != 0:
        raise ValueError('Triton LSE merge ran under frozen torch backend')
    if settings['kv_copy_backend'] == 'triton' and arm in ('method', 'allkept'):
        if any(type(adapter.get(key)) is not int or adapter[key] <= 0
               for key in ('triton_kv_copy_calls', 'triton_kv_copy_elements')):
            raise ValueError('frozen Triton KV copy path did not execute')


@contextmanager
def qualification_cells(binding, dataset):
    """Select the declared qualification dataset in memory; never rewrite a pin."""
    if dataset is None:
        yield
        return
    original = panel.read
    cells_path = Path(binding['cells']).resolve()
    def read(path):
        value = original(path)
        return [cell for cell in value if cell['dataset'] == dataset] if Path(path).resolve() == cells_path else value
    panel.read = read
    try:
        yield
    finally:
        panel.read = original


@contextmanager
def instrument_runner(adapter_module, settings, warm_count):
    original_adapter, original_validate = adapter_module.VllmMethodAdapter, panel.validate_receipts
    qualified = qualify.adapter_variant(original_adapter, settings, warm_count)
    class BackendAdapter(qualified):
        def __init__(self, *args, **kwargs):
            kwargs['kv_copy_backend'] = settings['kv_copy_backend']
            kwargs['merge_backend'] = settings['merge_backend']
            super().__init__(*args, **kwargs)
            if self.kv_copy_backend != settings['kv_copy_backend'] or self.merge_backend != settings['merge_backend']:
                raise ValueError('adapter KV copy backend drift')
    if not {'kv_copy_backend', 'merge_backend'} <= set(inspect.signature(original_adapter.__init__).parameters):
        raise ValueError('deployed adapter lacks explicit KV backend API')
    adapter_module.VllmMethodAdapter = BackendAdapter
    panel.validate_receipts = lambda arm, receipts, n, required: validate_receipt(
        arm, receipts, n, required, settings, original_validate)
    try:
        yield
    finally:
        adapter_module.VllmMethodAdapter, panel.validate_receipts = original_adapter, original_validate


def validate_qualification(proof, spec, binding, torch_version, vllm_version):
    expected = {(dataset, arm) for dataset in spec['datasets'] for arm in ARMS}
    arms = proof.get('arms', [])
    hosts = {block['host'] for block in spec['blocks']}
    if (proof.get('schema') != 'v29_expanded_aggregate_v1' or proof.get('protocol_id') != spec['protocol_id'] or
            proof.get('deployment_commit') != binding['deploy_commit'] or proof.get('qualification_only') is not True or
            proof.get('qualification_ready') is not True or proof.get('task_contracts') != spec['task_contracts'] or
            proof.get('scorer_public_toy_selftests') != {dataset: True for dataset in spec['datasets']} or
            len(arms) != len(expected) or {(row['dataset'], row['arm']) for row in arms} != expected or
            any(row.get('cells') != len(hosts) for row in arms)):
        raise ValueError('scored qualification source/task/full-arm/toy coverage differs')
    environments = proof.get('qualified_environments', [])
    if {value.get('host') for value in environments} != hosts or len(environments) != len(hosts):
        raise ValueError('every assigned host requires its own complete scored qualification')
    matched = [value for value in environments if value.get('host') == binding['host']]
    required = dict(gpu_uuid=binding['gpu_uuid'], deploy_commit=binding['deploy_commit'], cpu_threads=1,
                    torch=torch_version, vllm=vllm_version,
                    method_fingerprint=panel.read(binding['config'])['fingerprint'],
                    adapter_sha256=binding['source_files']['experiments/numerical_qk_reuse/vllm_adapter.py'])
    if len(matched) != 1 or any(matched[0].get(key) != value for key, value in required.items()):
        raise ValueError('this host/GPU/OMP/deployment/software lacks matching scored qualification')


def run_worker(args):
    import os
    binding, spec, config = read_frozen(args.binding)
    if not 0 <= args.block < len(spec['blocks']) or spec['blocks'][args.block]['host'] != binding['host']:
        raise ValueError('block host assignment differs')
    if Path.cwd().resolve() != Path(binding['deploy']).resolve():
        raise ValueError('run from the exact bound deployment')
    if os.environ.get('OMP_NUM_THREADS') != '1':
        raise ValueError('frozen OMP_NUM_THREADS=1 required before loading')
    from experiments.numerical_qk_reuse import v21
    v21.validate_effective(config, config['condition'])
    settings = execution_settings(spec)
    args.preflight = args.mode == 'qualification'
    first_host_block = min(index for index, block in enumerate(spec['blocks']) if block['host'] == binding['host'])
    if args.preflight and args.block != first_host_block:
        raise ValueError('qualification uses the predeclared first engine seed assigned to this host')
    if not args.preflight:
        proof_path = getattr(args, 'qualification_score', None)
        if proof_path is None:
            raise ValueError('all-dataset/all-arm scored qualification is required before formal generation')
        proof = panel.read(proof_path)
        import torch
        import vllm
        validate_qualification(proof, spec, binding, torch.__version__, vllm.__version__)
    dataset = getattr(args, 'dataset', None)
    if args.preflight and dataset is None and len(spec['datasets']) == 1:
        dataset = next(iter(spec['datasets']))
    if args.preflight and dataset not in spec['datasets'] or not args.preflight and dataset is not None:
        raise ValueError('qualification requires a frozen dataset; benchmark cannot subset')
    warm_count = qualify.warm_request_count(args, binding, spec)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    status = dict(protocol_id=spec['protocol_id'], arm=args.arm, block=args.block,
                  run_id=args.run_dir.parent.name + '_' + args.run_dir.name, complete=False,
                  v29_config=settings, mode=args.mode, qualification_dataset=dataset)
    started = time.perf_counter()
    try:
        import experiments.numerical_qk_reuse.vllm_adapter as adapter_module
        with qualification_cells(binding, dataset), qualify.instrument_jit_snapshots(True), instrument_runner(adapter_module, settings, warm_count):
            panel.run(args, binding, spec, status)
        records = list(old._jsonl([args.run_dir / 'records.jsonl']))
        for row in records:
            validate_receipt(args.arm, row['receipts'], row['denoise_forward_count'], METHOD, settings)
        receipt = qualify.summarize_closed_run(records, status, settings, args.preflight)
        receipt['schema'] = 'v29_closed_worker_receipt_v1'
        receipt['jit_monitor_receipt'] = qualify.validate_jit_deltas(records, True)
        write_new(args.run_dir / 'qualification.json', receipt)
        status['complete'] = True
    finally:
        status['gpu_reserved_seconds'] = round(time.perf_counter() - started, 3)
        write_new(args.run_dir / 'terminal.json', status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    draft = sub.add_parser('spec')
    draft.add_argument('--suite', choices=SUITES, required=True)
    draft.add_argument('--name', required=True)
    draft.add_argument('--engine-seeds', type=int, nargs=8, required=True)
    draft.add_argument('--host', action='append', required=True)
    draft.add_argument('--out', type=Path, required=True)
    freeze = sub.add_parser('freeze')
    for name in ('spec', 'catalog', 'config', 'model', 'deploy', 'out-dir'):
        freeze.add_argument('--' + name, type=Path, required=True)
    freeze.add_argument('--host', required=True)
    freeze.add_argument('--gpu-uuid', required=True)
    rebind = sub.add_parser('rebind-mirror')
    for name in ('config', 'old-root', 'mirror-root', 'new-root', 'out'):
        rebind.add_argument('--' + name, type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--binding', type=Path, required=True)
    run.add_argument('--arm', choices=ARMS, required=True)
    run.add_argument('--block', type=int, required=True)
    run.add_argument('--run-dir', type=Path, required=True)
    run.add_argument('--mode', choices=('qualification', 'benchmark'), required=True)
    run.add_argument('--dataset', choices=COUNTS)
    run.add_argument('--qualification-score', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'spec':
        spec = build_spec(args.suite, args.name, args.engine_seeds, args.host)
        write_new(args.out, spec)
        print(json.dumps(dict(protocol_id=spec['protocol_id'], questions=sum(COUNTS[ds] for ds in spec['datasets']),
                              timed_requests=len(old.expected_inventory(spec)))))
    elif args.action == 'freeze':
        print(json.dumps(freeze_private(args.spec, args.catalog, args.config, args.model, args.deploy,
                                       args.host, args.gpu_uuid, args.out_dir)))
    elif args.action == 'rebind-mirror':
        write_new(args.out, rebind_config_from_mirror(panel.read(args.config), args.old_root, args.mirror_root, args.new_root))
    else:
        run_worker(args)


if __name__ == '__main__':
    main()
