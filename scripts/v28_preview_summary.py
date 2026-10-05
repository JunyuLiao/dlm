"""Strict CPU summary of six variants, six questions and four engine seeds.

Run where the private frozen bindings' source/input paths remain readable.
No protocol/arm/seed is relabelled to satisfy a different panel contract. The
unchanged v27 score_panel calls the v15 final-channel NeMo scorer; private ids
are used only to join pinned cells and gold. Artifacts contain aggregate counts
and statistics only, never bindings, source hashes, paths or completion text.

Bootstrap resamples questions within dataset/length strata, keeping all four
engine blocks and both sequential repeat labels together. Matching engine seeds
does not establish identical stochastic trajectories. Six questions constitute
a preview, not a confirmatory accuracy or speed result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
from collections import defaultdict

from scripts import v27_vllm_panel_summary as old
from scripts import v28_vllm_qualify as qualify
from scripts.v27_vllm_bind import _validate_fingerprints


VARIANTS = {
    'dense': ('dense', 128, 'legacy', 'q128_legacy'),
    'native': ('native', 128, 'legacy', 'q128_legacy'),
    'allkept_release': ('allkept', 128, 'release_after_invalidate', 'q128_release'),
    'main_legacy': ('method', 128, 'legacy', 'q128_legacy'),
    'main_release': ('method', 128, 'release_after_invalidate', 'q128_release'),
    'q64_release': ('method', 64, 'release_after_invalidate', 'q64_release'),
}
COMPARISONS = (
    ('q64_release', 'main_release'), ('main_release', 'main_legacy'),
    ('main_legacy', 'dense'), ('main_release', 'dense'), ('q64_release', 'dense'),
    ('allkept_release', 'dense'), ('native', 'dense'), ('main_release', 'allkept_release'),
)
SOURCES = ('scripts/v28_vllm_qualify.py', 'scripts/v27_vllm_panel_run.py',
           'scripts/v27_vllm_metrics.py', 'scripts/v28_jit_receipts.py',
           'experiments/numerical_qk_reuse/vllm_adapter.py')
METRICS = ('W', 'S', 'SN', 'N', 'P')


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError('a required private JSON input is unreadable or invalid') from None


def _rows(path):
    try:
        return list(old._jsonl([path]))
    except (OSError, ValueError):
        raise ValueError('a required worker ledger is unreadable or invalid') from None


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('required private file reference is missing')
    return Path(value).resolve()


def _frozen(entry, variant):
    binding = _read(_path(entry.get('binding')))
    if not isinstance(binding, dict):
        raise ValueError('binding must be an object')
    spec, config = _read(_path(binding.get('spec'))), _read(_path(binding.get('config')))
    if not isinstance(spec, dict) or not isinstance(config, dict):
        raise ValueError('spec/config must be objects')
    if binding.get('protocol_id') != spec.get('protocol_id') or not str(spec.get('protocol_id', '')).startswith('v28_'):
        raise ValueError('binding protocol identity differs')
    deploy = _path(binding.get('deploy'))
    try:
        deployed_sha = (deploy / 'DEPLOY_SHA').read_text(encoding='utf-8').strip()
    except OSError:
        raise ValueError('frozen deployment receipt is unreadable') from None
    if not binding.get('deploy_commit') or deployed_sha != binding['deploy_commit']:
        raise ValueError('frozen deployment commit drift')
    files = binding.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('binding source/input pins are missing')
    pins = {}
    for path, expected in files.items():
        path = _path(path)
        if path in pins or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('duplicate or invalid frozen source/input pin')
        try:
            actual = qualify.panel.digest(path)
        except OSError:
            raise ValueError('frozen source/input bytes are unavailable') from None
        if actual != expected:
            raise ValueError('frozen source/input byte drift')
        pins[path] = expected
    manifests = binding.get('manifests')
    if not isinstance(manifests, dict) or set(manifests) != set(spec.get('datasets', {})):
        raise ValueError('frozen manifest inventory differs')
    model = _path(binding.get('model'))
    model_inputs = [model / 'config.json', model / 'generation_config.json']
    inputs = [binding.get(key) for key in ('spec', 'config', 'cells')] + list(manifests.values())
    if any(_path(path) not in pins for path in inputs) or any(deploy / path not in pins for path in SOURCES):
        raise ValueError('binding must pin every input and required execution source')
    if any(path not in pins for path in model_inputs):
        raise ValueError('binding must pin common model/generation settings')
    _validate_fingerprints(config)
    arm, query_block, canvas, group = VARIANTS[variant]
    settings = qualify.execution_settings(query_block, 'request_clear', canvas)
    settings['jit_monitor_events'] = True
    expected_adapter = dict(lifecycle='request_clear', alias_splits=2)
    if canvas != 'legacy':
        expected_adapter['canvas_buffers'] = canvas
    if spec.get('adapter_settings') != expected_adapter or spec.get('jit_event_receipts') is not True:
        raise ValueError('variant lifecycle/alias/JIT contract differs')
    required = spec.get('primary_receipt_method', {})
    fixed = dict(q_block=query_block, score_period=64, decision_interval=6,
                 risk_state='dense_prefix', carry_first=True, fused_observe=True,
                 async_route=True, consumer='fa4')
    if any(required.get(key) != value for key, value in fixed.items()):
        raise ValueError('variant effective-method contract differs')
    if config.get('q_block', 128) != query_block or config.get('q_regroup', False) or config.get('q_carry64', False):
        raise ValueError('frozen query variant differs')
    inventory = old.expected_inventory(spec)
    if type(spec['settings'].get('cpu_threads')) is not int or spec['settings']['cpu_threads'] != 1:
        raise ValueError('preview requires frozen OMP CPU threads = 1')
    datasets = spec['datasets']
    if len(datasets) != 3 or any(len(ds['indices']) != 2 or len(ds['repeats']) != 8 for ds in datasets.values()):
        raise ValueError('preview must contain three strata and six questions with eight repeats')
    if any(set(spec['control_indices'][ds]) != set(cfg['indices']) for ds, cfg in datasets.items()):
        raise ValueError('preview controls must cover all six questions')
    blocks = spec.get('blocks', [])
    if len(blocks) != 4 or any(len(block['repeats']) != 2 for block in blocks):
        raise ValueError('preview requires four engine blocks with two repeat labels')
    if len({block['engine_seed'] for block in blocks}) != 4:
        raise ValueError('repeat labels cannot masquerade as independent engine seeds')
    selected = {key[:3] for key in inventory if key[3] == arm}
    if len(selected) != 48:
        raise ValueError('variant inventory must contain exactly 48 executions')
    private_cells = _read(_path(binding['cells']))
    items = {}
    reverse = set()
    if not isinstance(private_cells, list):
        raise ValueError('private frozen cells must be a list')
    for cell in private_cells:
        item = (cell.get('dataset'), old._integer(cell.get('index'), 'private cell index'))
        identifier = cell.get('id')
        if item in items or not isinstance(identifier, str) or not identifier or (item[0], identifier) in reverse:
            raise ValueError('private frozen item mapping is duplicate or invalid')
        items[item] = identifier
        reverse.add((item[0], identifier))
    if set(items) != {key[:2] for key in selected}:
        raise ValueError('private frozen question inventory differs')
    source_pins = {str(path.relative_to(deploy)): sha for path, sha in pins.items() if path.is_relative_to(deploy)}
    normalized_config = {key: value for key, value in config.items() if key not in ('q_block', 'fingerprint')}
    signature = dict(datasets=spec['datasets'], blocks=blocks, settings=spec['settings'],
                     arm_settings=spec['arm_settings'], order_seed=spec.get('order_seed'),
                     method_contract={key: value for key, value in required.items() if key != 'q_block'},
                     source_pins=source_pins, deploy_commit=binding['deploy_commit'],
                     host=binding.get('host'), gpu_uuid=binding.get('gpu_uuid'),
                     model=str(model), model_settings=[pins[path] for path in model_inputs],
                     cells=items, config=normalized_config,
                     manifests={ds: pins[_path(path)] for ds, path in manifests.items()})
    return binding, spec, settings, selected, signature, group


def load_family(family_path):
    """Validate all 24 closed workers before reading gold or invoking scoring."""
    try:
        return _load_family(family_path)
    except (KeyError, TypeError, IndexError, AttributeError):
        raise ValueError('missing or invalid frozen family/worker receipt field') from None


def _load_family(family_path):
    family = _read(family_path)
    variants = family.get('variants') if isinstance(family, dict) else None
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError('family must declare exactly the six preview variants')
    cells, completions = defaultdict(dict), {}
    common, protocols, run_ids, directories = None, {}, set(), set()
    for variant, (arm, _, _, _) in VARIANTS.items():
        entry = variants[variant]
        if not isinstance(entry, dict):
            raise ValueError('family variant must be an object')
        binding, spec, settings, expected, signature, group = _frozen(entry, variant)
        if common is None:
            common = signature
        elif signature != common:
            raise ValueError('family question/seed/settings/source/deploy/input drift')
        protocol_id = spec['protocol_id']
        if group in protocols and protocols[group] != protocol_id:
            raise ValueError('same variant family protocol identity differs')
        if group not in protocols and protocol_id in protocols.values():
            raise ValueError('distinct query/lifecycle protocols must retain distinct identities')
        protocols[group] = protocol_id
        run_dirs = entry.get('run_dirs')
        if not isinstance(run_dirs, list) or len(run_dirs) != 4:
            raise ValueError('variant requires exactly four closed worker directories')
        found, found_blocks = set(), set()
        for directory in run_dirs:
            directory = _path(directory)
            if directory in directories:
                raise ValueError('duplicate worker directory')
            directories.add(directory)
            status = _read(directory / 'terminal.json')
            if not isinstance(status, dict) or status.get('complete') is not True or status.get('mode') != 'benchmark':
                raise ValueError('preview worker is open, failed or qualification-only')
            block = status.get('block')
            if type(block) is not int or not 0 <= block < 4 or block in found_blocks:
                raise ValueError('duplicate or invalid worker engine block')
            found_blocks.add(block)
            if status.get('protocol_id') != protocol_id or status.get('arm') != arm or status.get('v28_config') != settings:
                raise ValueError('worker variant/protocol receipt differs')
            rid = status.get('run_id')
            if not isinstance(rid, str) or not rid or rid in run_ids:
                raise ValueError('worker run identity is missing or duplicated')
            run_ids.add(rid)
            records = _rows(directory / 'records.jsonl')
            for field in ('expected_timed', 'completed_timed'):
                if type(status.get(field)) is not int or status[field] != 12 or len(records) != 12:
                    raise ValueError('closed worker count must match twelve actual executions')
            block_spec = spec['blocks'][block]
            block_expected = {key for key in expected if key[2] in block_spec['repeats']}
            block_found = set()
            for row in records:
                key4 = old._key(row)
                key = key4[:3]
                if key4[3] != arm or key in found or key in block_found or key not in block_expected:
                    raise ValueError('duplicate, missing or unexpected variant cell')
                block_found.add(key)
                if row.get('schema') != old.SCHEMA or row.get('protocol_id') != protocol_id:
                    raise ValueError('record protocol identity differs')
                if row.get('run_id') != rid or row.get('qualification_only') is not False:
                    raise ValueError('record worker/qualification identity differs')
                if row.get('seed_applied') is not False or row.get('engine_seed') != block_spec['engine_seed']:
                    raise ValueError('record engine seed differs; repeat labels are not seed pairing')
                if type(row.get('engine_seed')) is not int or row.get('measurement_mode') != 'request_boundary_sync':
                    raise ValueError('record seed/timing boundary contract differs')
                if any(row.get(field) != value for field, value in spec['settings'].items()):
                    raise ValueError('record settings/OMP drift from frozen protocol')
                if type(row.get('cpu_threads')) is not int:
                    raise ValueError('record OMP setting must be the integer one')
                if any(row.get(field) != binding[field] for field in ('deploy_commit', 'host', 'gpu_uuid')):
                    raise ValueError('record deployment/host/GPU drift')
                if any(row.get(field) != value for field, value in spec['arm_settings'][arm].items()):
                    raise ValueError('record predeclared graph mode differs')
                for field in ('torch', 'vllm'):
                    if not isinstance(row.get(field), str) or not row[field]:
                        raise ValueError('record software version is missing')
                expected_adapter = None if arm == 'dense' else binding['files'][str(_path(binding['deploy']) / SOURCES[-1])]
                if row.get('adapter_sha256') != expected_adapter:
                    raise ValueError('record adapter byte identity differs from binding')
                if row.get('method_fingerprint') != (None if arm != 'method' else _read(binding['config'])['fingerprint']):
                    raise ValueError('record effective-config fingerprint differs')
                old._integer(row.get('output_tokens'), 'output tokens')
                if row.get('finish_reason') not in ('stop', 'eos', 'length'):
                    raise ValueError('record finish reason is unsupported')
                for field in ('wall_s', 'decode_span_s', 'prefill_s'):
                    old._positive(row.get(field), field)
                if row['decode_span_s'] > row['wall_s'] or row['prefill_s'] > row['wall_s']:
                    raise ValueError('record phase span exceeds wall time')
                qualify.validate_variant_receipt(arm, row.get('receipts'), row.get('denoise_forward_count'),
                                                 spec['primary_receipt_method'], settings)
                cells[key][variant] = row
            if block_found != block_expected:
                raise ValueError('worker engine-block question inventory is incomplete')
            qualify.summarize_closed_run(records, status, settings, False)
            qualify.validate_jit_deltas(records, True)
            found.update(block_found)
            private = _rows(directory / 'completions.private.jsonl')
            joined = set()
            for row in private:
                key4 = old._key(row)
                key = key4[:3]
                if key4[3] != arm or key not in block_found or key in joined:
                    raise ValueError('private worker completion inventory differs')
                public = cells[key][variant]
                if row.get('run_id') != rid or row.get('finish_reason') != public['finish_reason']:
                    raise ValueError('private/public worker identity differs')
                if row.get('id') != signature['cells'][key[:2]] or not isinstance(row.get('completion'), str):
                    raise ValueError('private completion differs from frozen item identity')
                joined.add(key)
                completions[key + (variant,)] = row
            if joined != block_found:
                raise ValueError('private completion inventory is incomplete')
        if found != expected or found_blocks != set(range(4)):
            raise ValueError('variant engine-block inventory is incomplete')
    environments = set()
    for variants_at_cell in cells.values():
        if set(variants_at_cell) != set(VARIANTS):
            raise ValueError('six-way paired cell inventory is incomplete')
        same = {tuple(row[field] for field in old.COMMON + ('cpu_threads',)) for row in variants_at_cell.values()}
        if len(same) != 1:
            raise ValueError('same-cell host/GPU/seed/settings/software drift')
        environments.update(tuple(row[field] for field in ('deploy_commit', 'host', 'gpu_uuid', 'torch', 'vllm'))
                            for row in variants_at_cell.values())
    if len(environments) != 1:
        raise ValueError('cross-block execution environment drift')
    return dict(cells), completions


def _value(row, metric):
    if metric == 'SN':
        return row['decode_span_s'] / row['denoise_forward_count']
    return row[{'W': 'wall_s', 'S': 'decode_span_s', 'N': 'denoise_forward_count', 'P': 'prefill_s'}[metric]]


def question_ci(groups, geometric=True, reps=4000):
    """Fixed seed 7, question clusters, fixed dataset/length stratum sizes."""
    if type(reps) is not int or reps < 40:
        raise ValueError('bootstrap repetitions must be >= 40')
    strata = defaultdict(list)
    for item in sorted(groups):
        strata[item[0]].append(item)
    if len(strata) == 1:
        return old.cluster_ci(groups, geometric=geometric, reps=reps, seed=7)
    rng = random.Random(7)
    estimate = old._geo if geometric else statistics.mean
    draws = []
    for _ in range(reps):
        sampled = [item for stratum in sorted(strata)
                   for item in rng.choices(strata[stratum], k=len(strata[stratum]))]
        draws.append(estimate([value for item in sampled for value in groups[item]]))
    draws.sort()
    return [draws[int(.025 * reps)], draws[int(.975 * reps) - 1]]


def _absolute(rows):
    result = dict(requests=len(rows), task_correct=sum(row['task_correct'] for row in rows),
                  strict_correct=sum(row['strict_correct'] for row in rows))
    for metric in METRICS:
        values = [_value(row, metric) for row in rows]
        result[metric] = dict(mean=statistics.mean(values), median=statistics.median(values),
                              geomean=old._geo(values), minimum=min(values), maximum=max(values))
    counts = [row['denoise_forward_count'] for row in rows]
    result['N']['sum'] = sum(counts)
    result['N']['distribution'] = [{'N': n, 'requests': counts.count(n)} for n in sorted(set(counts))]
    result['scheduler_N_sum'] = sum(row['scheduler_denoise_forward_count'] for row in rows)
    result['unused_denoising_mean'] = statistics.mean(row['speculative_unused_denoising'] for row in rows)
    return result


def _comparison(cells, candidate, reference, reps):
    result = dict(candidate=candidate, reference=reference, requests=len(cells),
                  question_clusters=len({key[:2] for key in cells}))
    for metric in METRICS:
        groups = defaultdict(list)
        for key, variants in cells.items():
            groups[key[:2]].append(_value(variants[candidate], metric) / _value(variants[reference], metric))
        result[metric] = dict(ratio=old._geo([v for values in groups.values() for v in values]),
                              question_cluster_ci95=question_ci(groups, reps=reps))
    for field in ('task_correct', 'strict_correct'):
        groups = defaultdict(list)
        for key, variants in cells.items():
            groups[key[:2]].append(100 * (variants[candidate][field] - variants[reference][field]))
        result[field] = dict(candidate_count=sum(v[candidate][field] for v in cells.values()),
                             reference_count=sum(v[reference][field] for v in cells.values()),
                             difference_pp=statistics.mean(x for values in groups.values() for x in values),
                             question_cluster_ci95_pp=question_ci(groups, geometric=False, reps=reps))
    return result


def summarize(cells, bootstrap_reps=4000):
    if len(cells) != 48 or any(set(row) != set(VARIANTS) for row in cells.values()):
        raise ValueError('summary requires the validated complete six-variant preview')
    for variants in cells.values():
        for row in variants.values():
            if any(type(row.get(field)) is not bool for field in ('task_correct', 'strict_correct')):
                raise ValueError('summary requires unchanged scorer correctness booleans')
    seeds = sorted({row['dense']['engine_seed'] for row in cells.values()})
    datasets = sorted({key[0] for key in cells})
    variants = []
    for name in VARIANTS:
        variants.append(dict(variant=name, **_absolute([v[name] for v in cells.values()]),
                             by_engine_seed=[dict(engine_seed=seed, **_absolute(
                                 [v[name] for v in cells.values() if v[name]['engine_seed'] == seed])) for seed in seeds],
                             by_dataset=[dict(dataset=ds, **_absolute(
                                 [v[name] for key, v in cells.items() if key[0] == ds])) for ds in datasets]))
    comparisons = []
    for candidate, reference in COMPARISONS:
        result = _comparison(cells, candidate, reference, bootstrap_reps)
        result['comparison_role'] = ('descriptive_extra' if (candidate, reference) == ('main_legacy', 'dense')
                                     else 'frozen_family_pair')
        result['by_engine_seed'] = [dict(engine_seed=seed, **_comparison(
            {key: v for key, v in cells.items() if v[candidate]['engine_seed'] == seed},
            candidate, reference, bootstrap_reps)) for seed in seeds]
        result['by_dataset'] = [dict(dataset=ds, **_comparison(
            {key: v for key, v in cells.items() if key[0] == ds}, candidate, reference, bootstrap_reps)) for ds in datasets]
        comparisons.append(result)
    return dict(schema='v28_seed4_preview_aggregate_v1', preview_only=True, question_clusters=6,
                engine_seed_blocks=4, sequential_repeats_per_block=2, requests_per_variant=48,
                bootstrap=dict(seed=7, reps=bootstrap_reps, unit='question', stratified_by='dataset_length_bin'),
                ratio_definition='candidate/reference; W,S,SN,P below one mean less time; N below one means fewer forwards',
                interpretation='Six-question preview only. Engine seed matching is not identical stochastic trajectory pairing. '
                               'Repeat labels are not independent seeds. Correctness CIs cluster all eight executions per question. '
                               'Request-cell McNemar would be exploratory only. SN is amortized decode time per actual forward. '
                               'N includes actual speculative unused denoising; S includes canvas commits. '
                               'Zero observed monitor events do not prove absence of all JIT/autotuning.',
                descriptive_extra='main_legacy/dense supplements the seven frozen family pairs; descriptive only.',
                variants=variants, comparisons=comparisons)


def write_summary(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    rows = [dict(variant=row['variant'], requests=row['requests'], task_correct=row['task_correct'],
                 strict_correct=row['strict_correct'], **{metric + '_mean': row[metric]['mean'] for metric in METRICS},
                 N_sum=row['N']['sum']) for row in result['variants']]
    with (out_dir / 'variants.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [result['interpretation'], '', result['descriptive_extra'], '', 'Ratios are candidate/reference.', '',
            '| Candidate | Reference | W | S | S/N | N | P |', '|---|---|---:|---:|---:|---:|---:|']
    for row in result['comparisons']:
        text.append('| ' + ' | '.join([row['candidate'], row['reference']] +
                                      [f"{row[metric]['ratio']:.4f}" for metric in METRICS]) + ' |')
    (out_dir / 'summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--gold', action='append', required=True, metavar='DATASET=PATH')
    parser.add_argument('--nemo-root')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--bootstrap-reps', type=int, default=4000)
    args = parser.parse_args(argv)
    if Path(args.out_dir).exists():
        raise ValueError('aggregate destination must be a new directory')
    golds = {}
    for value in args.gold:
        dataset, separator, path = value.partition('=')
        if not separator or not dataset or not path or dataset in golds:
            raise ValueError('gold arguments must be unique DATASET=PATH entries')
        golds[dataset] = path
    cells, completions = load_family(args.family)
    old.score_panel(cells, completions, golds, args.nemo_root)
    result = summarize(cells, args.bootstrap_reps)
    write_summary(result, args.out_dir)
    return result


if __name__ == '__main__':
    main()
