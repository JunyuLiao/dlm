"""Strict CPU-only v29 scoring; export aggregate measurements only.

Private family input: {workers:[{binding:PATH,run_dir:PATH}, ...]}.
Gold paths are scorer-only and must match each frozen byte pin. Qualification
and formal modes are disjoint; neither filters failed or incomplete workers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics

from scripts import v29_expanded_panel as runner

old, qualify = runner.old, runner.qualify


def load_family(path, qualification=False):
    family = runner.panel.read(path)
    entries = family.get('workers')
    if not isinstance(entries, list) or not entries:
        raise ValueError('private worker inventory required')
    reference, identity, manifest_pins, config_fields = None, None, None, None
    cells, completions, worker_keys = defaultdict(dict), {}, set()
    by_host = defaultdict(lambda: [[], []])
    all_bindings = {}
    all_artifacts = {}
    for entry in entries:
        artifacts = runner.FrozenArtifacts(entry['binding'], entry.get('scorer_artifacts'), entry.get('binding_sha256'))
        binding, spec, config = runner.read_frozen(entry['binding'], artifacts)
        common = (binding['deploy_commit'], binding['source_files'], binding['gold_sha256'],
                  tuple(binding['files'][binding['model_config_files'][name]] for name in ('config.json', 'generation_config.json')))
        manifests = {ds: binding['files'][value] for ds, value in binding['manifests'].items()}
        fields = runner._method_fields(config)
        if reference is None:
            reference, identity, manifest_pins, config_fields = spec, common, manifests, fields
        elif spec != reference or common != identity or manifests != manifest_pins or fields != config_fields:
            raise ValueError('protocol/deployment/source/model/gold/manifest/method drift across workers')
        host = binding['host']
        if host in all_bindings and all_bindings[host] != binding:
            raise ValueError('a host changed its frozen binding')
        all_bindings[host] = binding
        all_artifacts[host] = artifacts
        root = Path(entry['run_dir'])
        if entry.get('scorer_artifacts') is not None:
            worker_pins = entry.get('worker_sha256')
            required_worker_files = {'terminal.json', 'records.jsonl', 'completions.private.jsonl'}
            if (not isinstance(worker_pins, dict) or set(worker_pins) != required_worker_files or
                    any(runner.panel.digest(root / name) != digest for name, digest in worker_pins.items())):
                raise ValueError('scorer mirror worker artifact byte pins differ')
        status = runner.panel.read(root / 'terminal.json')
        block, arm = status.get('block'), status.get('arm')
        if type(block) is not int or not 0 <= block < 8 or arm not in runner.ARMS:
            raise ValueError('unknown block/arm')
        dataset = status.get('qualification_dataset')
        worker_key = (host, dataset, arm) if qualification else (block, arm)
        if worker_key in worker_keys:
            raise ValueError('duplicate worker')
        worker_keys.add(worker_key)
        if (status.get('complete') is not True or status.get('mode') != ('qualification' if qualification else 'benchmark') or
                status.get('protocol_id') != spec['protocol_id'] or host != spec['blocks'][block]['host']):
            raise ValueError('worker is open/failed or has a wrong mode/protocol/host')
        settings = runner.execution_settings(spec)
        if status.get('v29_config') != settings:
            raise ValueError('terminal execution standard drift')
        records_path, completions_path = root / 'records.jsonl', root / 'completions.private.jsonl'
        records = list(old._jsonl([records_path]))
        qualify.summarize_closed_run(records, status, settings, qualification)
        qualify.validate_jit_deltas(records, True)
        for row in records:
            if any(not isinstance(row.get(key), str) or not row[key] for key in ('host', 'gpu_uuid', 'deploy_commit', 'torch', 'vllm')):
                raise ValueError('known host/hardware/software identities required')
            if (row.get('host') != host or row.get('gpu_uuid') != binding['gpu_uuid'] or
                    row.get('deploy_commit') != binding['deploy_commit'] or type(row.get('cpu_threads')) is not int or row['cpu_threads'] != 1 or
                    row.get('engine_seed') != spec['blocks'][block]['engine_seed']):
                raise ValueError('bound environment/OMP/seed drift')
            if row.get('adapter_sha256') != (None if arm == 'dense' else binding['source_files'][
                    'experiments/numerical_qk_reuse/vllm_adapter.py']):
                raise ValueError('adapter source drift')
            if row.get('method_fingerprint') != (config['fingerprint'] if arm == 'method' else None):
                raise ValueError('method fingerprint drift')
            if any(row.get(key) != value for key, value in spec['settings'].items()):
                raise ValueError('frozen execution settings drift')
            runner.validate_receipt(arm, row['receipts'], row['denoise_forward_count'], runner.METHOD, settings)
        if qualification:
            first_host_block = min(index for index, value in enumerate(spec['blocks']) if value['host'] == host)
            if block != first_host_block or dataset not in spec['datasets'] or len(records) != 1:
                raise ValueError('qualification needs exactly one request per dataset/arm in first block')
            manifests_rows = artifacts.read(binding['manifests'][dataset])
            sizes = {row['id']: len(row['prompt_tokens']) for row in manifests_rows}
            eligible = [cell for cell in artifacts.read(binding['cells']) if cell['dataset'] == dataset]
            target = max(eligible, key=lambda cell: sizes[cell['id']])
            row = records[0]
            if (row.get('schema') != old.SCHEMA or row.get('dataset') != dataset or row.get('index') != target['index'] or
                    row.get('repeat') != 0 or row.get('qualification_only') is not True or
                    row.get('seed_applied') is not False or row.get('measurement_mode') != 'request_boundary_sync'):
                raise ValueError('qualification differs from frozen longest-input/measurement contract')
            if any(row.get(field) != spec['arm_settings'][arm][field] for field in old.GRAPH):
                raise ValueError('qualification graph-mode drift')
            if row.get('finish_reason') not in ('eos', 'stop', 'length'):
                raise ValueError('unknown qualification finish reason')
            if row['prefill_s'] + row['decode_span_s'] > row['wall_s'] + 1e-6:
                raise ValueError('qualification phase spans exceed request wall')
            key = old._key(row)
            private = list(old._jsonl([completions_path]))
            if len(private) != 1 or old._key(private[0]) != key or any(
                    private[0].get(field) != row[field] for field in ('run_id', 'finish_reason')):
                raise ValueError('qualification private/public join differs')
            if private[0].get('id') != target['id'] or not isinstance(private[0].get('completion'), str):
                raise ValueError('qualification private identity/text invalid')
            # Qualification still records repeat=0. The dictionary-only block
            # key separates different hosts; no public record is rewritten.
            internal_key = (dataset, row['index'], block, arm)
            cells[internal_key[:3]][arm] = row
            completions[internal_key] = private[0]
        else:
            if dataset is not None:
                raise ValueError('formal worker cannot restrict datasets')
            if any(row['repeat'] not in spec['blocks'][block]['repeats'] for row in records):
                raise ValueError('worker block/repeat drift')
            by_host[host][0].append(records_path)
            by_host[host][1].append(completions_path)
    expected_workers = ({(host, dataset, arm) for host in {value['host'] for value in reference['blocks']}
                         for dataset in reference['datasets'] for arm in runner.ARMS} if qualification else
                        {(block, arm) for block in range(8) for arm in runner.ARMS})
    if worker_keys != expected_workers:
        raise ValueError('worker inventory is incomplete or foreign')
    if not qualification:
        for host, (public, private) in by_host.items():
            part = deepcopy(reference)
            part['blocks'] = [block for block in reference['blocks'] if block['host'] == host]
            repeats = [repeat for block in part['blocks'] for repeat in block['repeats']]
            for value in part['datasets'].values():
                value['repeats'] = repeats
            old.prevalidate_closed_runs(public, part)
            local_cells, local_completions = old.load_panel(public, private, part)
            frozen_ids = {(cell['dataset'], cell['index']): cell['id'] for cell in all_artifacts[host].read(all_bindings[host]['cells'])}
            for key, completion in local_completions.items():
                if completion['id'] != frozen_ids[key[:2]] or key in completions:
                    raise ValueError('completion frozen identity/uniqueness differs')
                completions[key] = completion
            for key, rows in local_cells.items():
                if key in cells:
                    raise ValueError('seed block assigned twice')
                cells[key] = rows
        actual = {(key[0], key[1], key[2], arm) for key, rows in cells.items() for arm in rows}
        if actual != old.expected_inventory(reference):
            raise ValueError('global formal inventory differs')
    else:
        for rows in cells.values():
            if set(rows) != set(runner.ARMS):
                raise ValueError('qualification arms did not use the same item')
            first = next(iter(rows.values()))
            if any(any(row[field] != first[field] for field in old.COMMON) for row in rows.values()):
                raise ValueError('same qualification cell environment differs')
    if len({(row.get('torch'), row.get('vllm')) for rows in cells.values() for row in rows.values()}) != 1:
        raise ValueError('software version drift across engine blocks/hosts')
    # This scoring-only return adds local contract locations; original generation
    # binding, method paths, host identities and public records remain untouched.
    first_host = next(iter(all_bindings))
    scoring_binding = deepcopy(all_bindings[first_host])
    scoring_binding['task_contracts'] = {ds: str(all_artifacts[first_host].local(value))
                                       for ds, value in scoring_binding['task_contracts'].items()}
    return dict(cells), completions, reference, scoring_binding


def _gold(path, expected):
    if runner.panel.digest(path) != expected:
        raise ValueError('scorer-only gold differs from frozen bytes')
    return runner.panel.read(path)


def scorer_selftest(spec, binding, nemo_root=None):
    """Real pinned scorers on public correct/wrong toys, never private outputs."""
    passed = {}
    if any(ds.startswith('longbench_v2_') for ds in spec['datasets']):
        if nemo_root is None:
            raise ValueError('explicit pinned NeMo root required')
        old.task.NEMO = Path(nemo_root)
        old.task.nemo_prompt_config.cache_clear()
        rows = old.task.score(['<channel|>Answer: A', '<channel|>Answer: B'], ['A','A'], ['eos','eos'])
        if [row['strict_correct'] for row in rows] != [True, False] or not all(row['parsed'] for row in rows):
            raise ValueError('pinned LongBench public toy self-test failed')
        passed.update({ds: True for ds in spec['datasets'] if ds.startswith('longbench_v2_')})
    if 'aime26' in spec['datasets']:
        from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
        rows = [numeric_score(final_response('<channel|>Answer: '+text, True), '42') for text in ('42','41')]
        if [bool(row['correct']) for row in rows] != [True, False]:
            raise ValueError('existing AIME public toy self-test failed')
        passed['aime26'] = True
    if 'humaneval' in spec['datasets']:
        from scripts import v27_humaneval as he
        he.sandbox_preflight()
        toy = dict(prompt='def toy(x):\n    """Public toy."""\n', entry_point='toy',
                   test='def check(f):\n    assert f(2) == 2\n')
        answers = []
        for expression in ('x','0'):
            code, reason = he.extract_code('<channel|>```python\ndef toy(x):\n    return '+expression+'\n```', 'toy')
            answers.append(he.run_test(he.program_for(toy, code))[0])
        if answers != [True, False]:
            raise ValueError('HumanEval official-test public toy self-test failed')
        passed['humaneval'] = True
    return passed


def score_tasks(cells, completions, spec, binding, gold_paths, nemo_root=None):
    """Delegate task extraction/evaluation unchanged; retain booleans only."""
    if set(gold_paths) != set(spec['datasets']):
        raise ValueError('exact scorer-only gold inventory required')
    golds = {ds: _gold(path, binding['gold_sha256'][ds]) for ds, path in gold_paths.items()}
    selftests = scorer_selftest(spec, binding, nemo_root)
    longbench = {ds for ds in spec['datasets'] if ds.startswith('longbench_v2_')}
    if longbench:
        if nemo_root is None:
            raise ValueError('explicit pinned NeMo root required for LongBench')
        old.score_panel({key: rows for key, rows in cells.items() if key[0] in longbench},
                        {key: row for key, row in completions.items() if key[0] in longbench},
                        {ds: gold_paths[ds] for ds in longbench}, nemo_root)
    if 'aime26' in spec['datasets']:
        from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
        source = golds['aime26']
        if not isinstance(source, list) or any(not isinstance(row, dict) or 'id' not in row for row in source):
            raise ValueError('AIME uses existing id/expected-or-answer row-list gold')
        gold = {row['id']: row for row in source}
        if len(gold) != len(source):
            raise ValueError('duplicate AIME gold identity')
        for key, completion in sorted(completions.items()):
            if key[0] != 'aime26':
                continue
            answer = gold.get(completion['id'], {})
            expected = answer.get('expected', answer.get('answer'))
            if expected is None:
                raise ValueError('AIME gold identity/answer missing')
            text = final_response(completion['completion'], True)
            result = numeric_score(text, str(expected))
            correct = bool(result['correct'])
            cells[key[:3]][key[3]].update(task_correct=correct,
                strict_correct=correct and completion['finish_reason'] in ('eos', 'stop'),
                parsed=result['extracted'] is not None, final_channel_present=bool(text))
    if 'humaneval' in spec['datasets']:
        from scripts import v27_humaneval as humaneval
        contract_path = binding['task_contracts'].get('humaneval')
        contract = runner.panel.read(contract_path)
        humaneval.validate_contract(contract)
        if contract['scorer_sha256'] != runner.panel.digest(Path(humaneval.__file__)):
            raise ValueError('HumanEval extractor/sandbox scorer byte drift')
        gold = golds['humaneval']
        if not isinstance(gold, dict):
            raise ValueError('HumanEval uses existing private task dictionary')
        humaneval.sandbox_preflight()
        for key, completion in sorted(completions.items()):
            if key[0] != 'humaneval':
                continue
            problem = gold.get(completion['id'])
            if not isinstance(problem, dict) or any(field not in problem for field in ('task_id', 'prompt', 'test', 'entry_point')):
                raise ValueError('HumanEval gold task/test missing')
            code, extraction = humaneval.extract_code(completion['completion'], problem['entry_point'])
            passed, reason = humaneval.run_test(humaneval.program_for(problem, code)) if code else (False, extraction)
            # Historical HumanEval contract uses test pass, even at a length cap.
            cells[key[:3]][key[3]].update(task_correct=bool(passed), strict_correct=bool(passed),
                parsed=code is not None, final_channel_present=bool(humaneval.final_text(completion['completion'])))
    return selftests


def summarize(cells, spec, bootstrap_reps=4000, qualification=False, scorer_selftests=None):
    result = old.summarize(cells, bootstrap_reps)
    result.update(schema='v29_expanded_aggregate_v1', protocol_id=spec['protocol_id'],
                  qualification_only=qualification, task_contracts=spec['task_contracts'],
                  scorer_version={ds: value['scorer'] for ds, value in spec['task_contracts'].items()},
                  inference='Observed differences and question-cluster CIs only; noninferiority margin unset. '
                            'No noninferiority decision. McNemar is exploratory; repeats share questions.',
                  quality_fields='LongBench/AIME strict_correct requires EOS; HumanEval strict_correct retains '
                                 'the existing EOS-independent official-test pass@1 contract.',
                  engine_seed_policy='Eight predeclared distinct load-time seeds, one timed request per item/seed; '
                                     'no per-request seed reset or identical-trajectory claim.',
                  qualification_scope='Generation/scorer qualification only; not accuracy or speed evidence' if qualification else None)
    result['deployment_commit'] = next(iter(next(iter(cells.values())).values()))['deploy_commit']
    if qualification:
        result['engine_seed_policy'] = 'One predeclared qualification engine seed per assigned host; no scientific inference.'
        result['qualification_ready'] = (set(scorer_selftests or {}) == set(spec['datasets']) and
                                        all(value is True for value in scorer_selftests.values()))
        result['qualification_policy'] = ('Complete source/inventory/receipts/join and real public-toy scorer self-tests. '
                                          'Wrong, capped, unparsed and missing-final responses remain scored failures; '
                                          'they are not infrastructure failures or grounds to replace an item/seed.')
        fields = ('host', 'gpu_uuid', 'torch', 'vllm', 'cpu_threads', 'deploy_commit')
        environments = {tuple(row[field] for field in fields) for rows in cells.values() for row in rows.values()}
        result['qualified_environments'] = [dict(zip(fields, values)) for values in sorted(environments)]
        for environment in result['qualified_environments']:
            method_rows = [rows['method'] for rows in cells.values() if rows['method']['host'] == environment['host']]
            for field in ('method_fingerprint', 'adapter_sha256'):
                values = {row[field] for row in method_rows}
                if len(values) != 1:
                    raise ValueError('qualified host method/adapter identity differs')
                environment[field] = next(iter(values))
        result['scorer_public_toy_selftests'] = scorer_selftests
    by_seed, comparisons_by_seed = [], []
    for seed in sorted({row['engine_seed'] for rows in cells.values() for row in rows.values()}):
        for ds in spec['datasets']:
            part = {key: rows for key, rows in cells.items() if key[0] == ds and next(iter(rows.values()))['engine_seed'] == seed}
            if not part:
                continue
            # Only points/counts are exported here; no seed ranking or selection.
            summary = old.summarize(part, 40)
            for comparison in summary['comparisons']:
                comparisons_by_seed.append(dict(engine_seed=seed, **{key: value for key, value in comparison.items()
                    if not key.endswith('_ci95') and key != 'exploratory_mcnemar_exact_p'}))
            for row in summary['arms']:
                values = [rows[row['arm']]['denoise_forward_count'] for rows in part.values()]
                by_seed.append(dict(engine_seed=seed, dataset=ds, arm=row['arm'], requests=len(values),
                                    N_mean=statistics.mean(values), N_median=statistics.median(values),
                                    N_min=min(values), N_max=max(values), N_sum=sum(values),
                                    N_distribution=dict(sorted(Counter(values).items())),
                                    W_mean=row['wall_s_mean'], S_mean=row['decode_span_s_mean'],
                                    correct=row['strict_correct']))
    result['by_engine_seed'] = by_seed
    result['by_engine_seed_comparisons'] = comparisons_by_seed
    return result


def parse_gold(arguments):
    result = {}
    for argument in arguments:
        dataset, separator, path = argument.partition('=')
        if not separator or not path or dataset in result:
            raise ValueError('gold arguments must be unique DATASET=PATH')
        result[dataset] = Path(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', type=Path, required=True)
    parser.add_argument('--gold', action='append', required=True)
    parser.add_argument('--nemo-root', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--bootstrap-reps', type=int, default=4000)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise ValueError('a new output directory is required')
    cells, completions, spec, binding = load_family(args.family, args.qualification)
    selftests = score_tasks(cells, completions, spec, binding, parse_gold(args.gold), args.nemo_root)
    result = summarize(cells, spec, args.bootstrap_reps, args.qualification, selftests)
    old.write_summary(result, args.out_dir)
    # Adapt the legacy renderer without changing the original shared scorer.
    markdown = args.out_dir / 'summary.md'
    text = markdown.read_text().replace('# v27 vLLM complete panel', '# v29 expanded vLLM panel', 1)
    markdown.write_text(text + '\n\n' + result['quality_fields'] + '\n\n' + result['inference'] + '\n')
    print(json.dumps(dict(protocol_id=spec['protocol_id'], requests=sum(len(rows) for rows in cells.values()),
                          qualification_only=args.qualification, scoring_complete=True)))


if __name__ == '__main__':
    main()
