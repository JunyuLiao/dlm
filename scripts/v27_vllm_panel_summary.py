"""CPU-only complete-panel summary for the v27 native vLLM experiment.

Private completion ids join the v15 LongBench gold JSON object (id -> A-D).
Scoring calls v15_longbench_task.score unchanged; no prompt, id, prediction,
completion or private path is written to the aggregate artifacts.
Ratios are candidate/reference (<1 faster). Repeat numbers identify repeated
requests, not applied sampling seeds. Item-cluster bootstrap retains all repeats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scripts import v15_longbench_task as task
from scripts.v27_vllm_panel_run import validate_receipts

SCHEMA = 'v27_vllm_panel_v1'
ARMS = ('dense', 'method', 'native', 'allkept')
COMMON = ('deploy_commit', 'host', 'gpu_uuid', 'engine_seed', 'max_model_len',
          'chunk', 'gpu_memory_utilization', 'block_size', 'torch', 'vllm')
GRAPH = ('compilation_config', 'cudagraph_mode')
FROZEN_SETTINGS = ('max_model_len', 'chunk', 'gpu_memory_utilization', 'block_size')


def _integer(value, label, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f'{label} must be an integer >= {1 if positive else 0}')
    return value


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return value


def _integers(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} must be a nonempty integer list')
    for value in values:
        _integer(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f'duplicate {label}')
    return values


def expected_inventory(protocol):
    if not isinstance(protocol, dict) or not all(protocol.get(k) for k in ('schema', 'name', 'protocol_id')):
        raise ValueError('protocol identity is missing')
    if protocol.get('arms') != ['dense', 'method'] or protocol.get('controls') != ['native', 'allkept']:
        raise ValueError('protocol must declare dense/method and native/allkept controls')
    if not isinstance(protocol.get('primary_receipt_method'), dict) or not protocol['primary_receipt_method']:
        raise ValueError('frozen primary_receipt_method must be a nonempty object')
    settings = protocol.get('settings')
    if not isinstance(settings, dict) or any(settings.get(field) is None for field in FROZEN_SETTINGS):
        raise ValueError('protocol must freeze execution settings')
    datasets = protocol.get('datasets')
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError('protocol datasets are missing')
    controls = protocol.get('control_indices')
    if not isinstance(controls, dict) or set(controls) != set(datasets):
        raise ValueError('control_indices dataset inventory differs')
    graph = protocol.get('arm_settings')
    if not isinstance(graph, dict) or set(graph) != set(ARMS):
        raise ValueError('arm_settings must predeclare every arm graph configuration')
    for arm in ARMS:
        if not isinstance(graph[arm], dict) or any(graph[arm].get(k) not in ('default', 'PIECEWISE') for k in GRAPH):
            raise ValueError('arm_settings graph configuration is invalid')
        if graph[arm]['compilation_config'] != graph[arm]['cudagraph_mode']:
            raise ValueError('arm_settings graph fields disagree')
        required_mode = 'default' if arm == 'dense' else 'PIECEWISE'
        if graph[arm]['cudagraph_mode'] != required_mode:
            raise ValueError('only dense default / adapter PIECEWISE difference is allowed')
    inventory = set()
    for dataset, config in datasets.items():
        if not isinstance(dataset, str) or not dataset or not isinstance(config, dict):
            raise ValueError('invalid protocol dataset')
        indices = _integers(config.get('indices'), 'indices')
        repeats = _integers(config.get('repeats'), 'repeats')
        control_indices = _integers(controls[dataset], 'control_indices')
        if not set(control_indices) <= set(indices):
            raise ValueError('controls lie outside the main panel')
        for index in indices:
            for repeat in repeats:
                for arm in protocol['arms'] + (protocol['controls'] if index in control_indices else []):
                    inventory.add((dataset, index, repeat, arm))
    blocks = protocol.get('blocks')
    if blocks is not None:
        if not isinstance(blocks, list) or not blocks:
            raise ValueError('protocol blocks must be a nonempty list')
        declared = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError('protocol block must be an object')
            _integer(block.get('engine_seed'), 'block engine_seed')
            declared.extend(_integers(block.get('repeats'), 'block repeats'))
        required = {key[2] for key in inventory}
        if len(declared) != len(set(declared)) or set(declared) != required:
            raise ValueError('block repeat inventory differs from panel')
    return inventory


def _jsonl(paths):
    for path in paths:
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('JSONL record must be an object')
                    yield row


def _key(row):
    if not isinstance(row.get('dataset'), str) or not isinstance(row.get('arm'), str):
        raise ValueError('record dataset/arm must be strings')
    return (row['dataset'], _integer(row.get('index'), 'index'),
            _integer(row.get('repeat'), 'repeat'), row['arm'])


def prevalidate_closed_runs(record_paths, protocol):
    """CLI gate: a worker ledger counts only after its matching terminal closes."""
    for path in record_paths:
        path = Path(path)
        terminal_path = path.parent / 'terminal.json'
        if not terminal_path.is_file():
            raise ValueError('worker terminal is missing; open runs cannot be summarized')
        terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
        if not isinstance(terminal, dict) or terminal.get('complete') is not True:
            raise ValueError('worker terminal is incomplete or failed')
        rows = list(_jsonl([path]))
        if not rows:
            raise ValueError('closed worker record file is empty')
        if terminal.get('protocol_id') != protocol['protocol_id']:
            raise ValueError('terminal protocol identity differs')
        if any(not terminal.get(field) or any(row.get(field) != terminal[field] for row in rows)
               for field in ('arm', 'run_id')):
            raise ValueError('terminal arm/run_id identity differs from record file')
        for field in ('completed_timed', 'expected_timed'):
            if type(terminal.get(field)) is not int or terminal[field] != len(rows):
                raise ValueError('terminal timed count differs from actual record inventory')


def load_panel(record_paths, completion_paths, protocol):
    """Reject incomplete or drifted panels before reading any gold or scoring."""
    expected = expected_inventory(protocol)
    engine_seeds = {repeat: block['engine_seed'] for block in protocol.get('blocks', []) for repeat in block['repeats']}
    records, completions = {}, {}
    for row in _jsonl(record_paths):
        key = _key(row)
        if key in records:
            raise ValueError('duplicate public execution record')
        if key not in expected:
            raise ValueError('public execution outside frozen inventory')
        if row.get('schema') != SCHEMA or row.get('protocol_id') != protocol['protocol_id']:
            raise ValueError('record schema/protocol identity differs')
        if row.get('seed_applied') is not False or row.get('measurement_mode') != 'request_boundary_sync':
            raise ValueError('seed or measurement contract differs')
        if type(row.get('graph_captures_timed')) is not int or row['graph_captures_timed'] != 0:
            raise ValueError('timed graph captures must be known and zero')
        if not isinstance(row.get('run_id'), str) or not row['run_id']:
            raise ValueError('run_id is missing')
        for field in COMMON:
            if row.get(field) is None:
                raise ValueError(f'execution setting {field} is missing')
        if any(row.get(field) != protocol['settings'][field] for field in FROZEN_SETTINGS):
            raise ValueError('execution settings drift from frozen protocol')
        deltas = row.get('compilation_deltas')
        if deltas is not None:
            if not isinstance(deltas, dict):
                raise ValueError('timed compilation_deltas must be an object')
            for field in ('num_backend_compilations', 'num_inductor_compiles'):
                if field in deltas and (type(deltas[field]) is not int or deltas[field] != 0):
                    raise ValueError('timed compilation delta must be known and zero')
        for field in ('deploy_commit', 'host', 'gpu_uuid', 'torch', 'vllm'):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f'execution setting {field} is invalid')
        _integer(row['engine_seed'], 'engine_seed')
        if engine_seeds and row['engine_seed'] != engine_seeds[row['repeat']]:
            raise ValueError('engine seed differs from frozen block')
        for field in ('max_model_len', 'chunk', 'block_size', 'denoise_forward_count'):
            _integer(row.get(field), field, positive=True)
        if 'commit_forward_count' in row:
            _integer(row['commit_forward_count'], 'commit_forward_count')
        _integer(row.get('output_tokens'), 'output_tokens')
        _positive(row['gpu_memory_utilization'], 'gpu_memory_utilization')
        if row['gpu_memory_utilization'] > 1:
            raise ValueError('gpu_memory_utilization exceeds one')
        for field in ('wall_s', 'decode_span_s', 'prefill_s'):
            _positive(row.get(field), field)
        if row['decode_span_s'] > row['wall_s'] or row['prefill_s'] > row['wall_s']:
            raise ValueError('phase span exceeds request wall time')
        if row.get('finish_reason') not in ('stop', 'eos', 'length'):
            raise ValueError('unsupported finish_reason')
        for field in GRAPH:
            if row.get(field) != protocol['arm_settings'][row['arm']][field]:
                raise ValueError('graph setting differs from predeclared arm_settings')
        if row['arm'] != 'dense' and (not isinstance(row.get('adapter_sha256'), str) or not row['adapter_sha256']):
            raise ValueError('adapter source identity is missing')
        if row['arm'] == 'method':
            if not isinstance(row.get('method_fingerprint'), str) or not row['method_fingerprint']:
                raise ValueError('method fingerprint is missing')
        try:
            validate_receipts(row['arm'], row.get('receipts'), row['denoise_forward_count'],
                              protocol['primary_receipt_method'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('execution receipt invalid or differs from frozen method/coverage') from exc
        records[key] = row
    if set(records) != expected:
        raise ValueError('public record inventory is incomplete')
    adapters = {r['adapter_sha256'] for r in records.values() if r['arm'] != 'dense'}
    fingerprints = {r['method_fingerprint'] for r in records.values() if r['arm'] == 'method'}
    if len(adapters) != 1 or len(fingerprints) != 1:
        raise ValueError('adapter source or method fingerprint drift')
    if len({r['deploy_commit'] for r in records.values()}) != 1:
        raise ValueError('deploy commit drift')
    cells = defaultdict(dict)
    for key, row in records.items():
        cells[key[:3]][key[3]] = row
    for arms in cells.values():
        reference = next(iter(arms.values()))
        if any(any(row[field] != reference[field] for field in COMMON) for row in arms.values()):
            raise ValueError('same-cell execution environment/settings drift')
    private_ids, reverse_ids = {}, {}
    for row in _jsonl(completion_paths):
        key = _key(row)
        if key in completions:
            raise ValueError('duplicate private completion')
        if key not in records:
            raise ValueError('private completion outside frozen inventory')
        public = records[key]
        if row.get('run_id') != public['run_id'] or row.get('finish_reason') != public['finish_reason']:
            raise ValueError('private/public execution identity differs')
        if not isinstance(row.get('completion'), str) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('private completion text/id is missing')
        item = key[:2]
        if item in private_ids and private_ids[item] != row['id']:
            raise ValueError('private item identity differs across arms/repeats')
        reverse = (key[0], row['id'])
        if reverse in reverse_ids and reverse_ids[reverse] != key[1]:
            raise ValueError('duplicate private item assigned to different public indices')
        reverse_ids[reverse] = key[1]
        private_ids[item] = row['id']
        completions[key] = row
    if set(completions) != expected:
        raise ValueError('private completion inventory is incomplete')
    return cells, completions


def score_panel(cells, completions, gold_paths, nemo_root=None):
    """Use the unchanged final-channel NeMo MCQ scorer, including strict EOS."""
    if nemo_root is not None:
        task.NEMO = Path(nemo_root)
        task.nemo_prompt_config.cache_clear()
    datasets = {key[0] for key in cells}
    if set(gold_paths) != datasets:
        raise ValueError('gold dataset inventory differs')
    golds = {}
    for dataset, path in gold_paths.items():
        labels = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(labels, dict) or any(not isinstance(k, str) or v not in ('A', 'B', 'C', 'D')
                                              for k, v in labels.items()):
            raise ValueError('gold must use the v15 private id -> A-D JSON contract')
        golds[dataset] = labels
    ordered = sorted(completions)
    labels = []
    for key in ordered:
        label = golds[key[0]].get(completions[key]['id'])
        if label is None:
            raise ValueError('private completion id missing from gold')
        labels.append(label)
    scores = task.score([completions[k]['completion'] for k in ordered], labels,
                        ['eos' if completions[k]['finish_reason'] in ('stop', 'eos') else 'length' for k in ordered])
    if len(scores) != len(ordered):
        raise ValueError('scorer output inventory differs')
    for key, score in zip(ordered, scores):
        # Keep only public-safe boolean quality fields, never the prediction.
        row = cells[key[:3]][key[3]]
        for field in ('task_correct', 'strict_correct', 'parsed', 'final_channel_present'):
            if type(score.get(field)) is not bool:
                raise ValueError('scorer quality field is invalid')
            row[field] = score[field]


def _geo(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def cluster_ci(groups, geometric=True, reps=4000, seed=7):
    """Resample entire items; repeats never become independent clusters."""
    if len(groups) < 2:
        return [None, None]
    if type(reps) is not int or reps < 40:
        raise ValueError('bootstrap repetitions must be >= 40')
    items = sorted(groups)
    rng = random.Random(seed)
    estimate = _geo if geometric else statistics.mean
    draws = sorted(estimate([value for item in rng.choices(items, k=len(items)) for value in groups[item]])
                   for _ in range(reps))
    return [draws[int(.025 * reps)], draws[int(.975 * reps) - 1]]


def _mcnemar_exploratory(arm_only, base_only):
    n = arm_only + base_only
    if not n:
        return 1.0
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2))
               for k in range(min(arm_only, base_only) + 1))
    return min(1.0, 2 * tail)


def summarize(cells, bootstrap_reps=4000):
    if type(bootstrap_reps) is not int or bootstrap_reps < 40:
        raise ValueError('bootstrap repetitions must be >= 40')
    arm_rows, comparisons = [], []
    for dataset in sorted({key[0] for key in cells}):
        selected = {key: arms for key, arms in cells.items() if key[0] == dataset}
        for arm in ARMS:
            present = [(key, arms[arm]) for key, arms in selected.items() if arm in arms]
            wall = [row['wall_s'] for _, row in present]
            decode = [row['decode_span_s'] for _, row in present]
            prefill = [row['prefill_s'] for _, row in present]
            forwards = [row['denoise_forward_count'] for _, row in present]
            commit = [row['commit_forward_count'] for _, row in present if 'commit_forward_count' in row]
            arm_rows.append(dict(dataset=dataset, arm=arm, cells=len(present),
                                 items=len({key[1] for key, _ in present}),
                                 strict_correct=sum(row['strict_correct'] for _, row in present),
                                 task_correct=sum(row['task_correct'] for _, row in present),
                                 capped=sum(row['finish_reason'] == 'length' for _, row in present),
                                 unparsed=sum(not row['parsed'] for _, row in present),
                                 wall_s_mean=statistics.mean(wall), wall_s_median=statistics.median(wall),
                                 decode_span_s_mean=statistics.mean(decode),
                                 decode_span_s_median=statistics.median(decode),
                                 prefill_s_mean=statistics.mean(prefill),
                                 denoise_forward_count_mean=statistics.mean(forwards),
                                 denoise_forward_count_median=statistics.median(forwards),
                                 denoise_forward_count_sum=sum(forwards),
                                 decode_s_per_denoise_forward_geomean=_geo(
                                     [row['decode_span_s'] / row['denoise_forward_count'] for _, row in present]),
                                 commit_forward_count_mean=statistics.mean(commit) if commit else None,
                                 commit_forward_count_observed_cells=len(commit)))
        for arm, base in (('method', 'dense'), ('native', 'dense'), ('allkept', 'dense'),
                          ('method', 'native'), ('method', 'allkept')):
            pairs = [(key, arms[arm], arms[base]) for key, arms in selected.items() if arm in arms and base in arms]
            row = dict(dataset=dataset, arm=arm, base=base, cells=len(pairs),
                       items=len({key[1] for key, _, _ in pairs}),
                       strict_correct=sum(a['strict_correct'] for _, a, _ in pairs),
                       base_strict_correct=sum(b['strict_correct'] for _, _, b in pairs),
                       arm_only=sum(a['strict_correct'] and not b['strict_correct'] for _, a, b in pairs),
                       base_only=sum(b['strict_correct'] and not a['strict_correct'] for _, a, b in pairs))
            for name, ratio in (
                    ('W', lambda a, b: a['wall_s'] / b['wall_s']),
                    ('S', lambda a, b: a['decode_span_s'] / b['decode_span_s']),
                    ('P', lambda a, b: a['prefill_s'] / b['prefill_s']),
                    ('SN', lambda a, b: (a['decode_span_s'] / a['denoise_forward_count']) /
                     (b['decode_span_s'] / b['denoise_forward_count'])),
                    ('N', lambda a, b: a['denoise_forward_count'] / b['denoise_forward_count'])):
                groups = defaultdict(list)
                for key, a, b in pairs:
                    groups[key[1]].append(ratio(a, b))
                row[name] = _geo([value for values in groups.values() for value in values])
                row[name + '_ci95'] = cluster_ci(groups, reps=bootstrap_reps)
            differences = defaultdict(list)
            for key, a, b in pairs:
                differences[key[1]].append(int(a['strict_correct']) - int(b['strict_correct']))
            row['accuracy_difference'] = statistics.mean([value for values in differences.values() for value in values])
            row['accuracy_difference_ci95'] = cluster_ci(differences, geometric=False, reps=bootstrap_reps)
            row['exploratory_mcnemar_exact_p'] = _mcnemar_exploratory(row['arm_only'], row['base_only'])
            comparisons.append(row)
    return dict(schema='v27_vllm_panel_summary_v1', scorer_version=task.SCORER_VERSION,
                pairing='dataset/index/repeat; repeat is not a sampling seed; seed_applied=false',
                confidence='95% item-cluster percentile bootstrap retaining all repeats',
                timing='W=request boundary wall; S=decode span excluding initial prefill; P=initial prefill; '
                       'N=actual denoising forwards excluding encoder commits; SN=S/N amortized cost',
                absolute_statistics='Per-arm means/medians use all timed requests; '
                                    'decode_s_per_denoise_forward_geomean is the geometric mean of request S/N; '
                                    'commit mean uses only its explicitly reported observed cells',
                inference='McNemar is exploratory only: repeats are not independent seeded samples',
                arms=arm_rows, comparisons=comparisons)


def write_summary(summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ('summary.json', 'summary.csv', 'arms.csv', 'summary.md')
    if any((out_dir / name).exists() for name in names):
        raise ValueError('summary outputs already exist; use a new output directory')
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, rows in (('summary.csv', summary['comparisons']), ('arms.csv', summary['arms'])):
        with (out_dir / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def metric(row, name):
        lo, hi = row[name + '_ci95']
        interval = 'CI unavailable (one item)' if lo is None else f'[{lo:.4f}, {hi:.4f}]'
        return f'{row[name]:.4f} {interval}'
    lines = ['# v27 vLLM complete panel', '', summary['pairing'], '', summary['confidence'], '',
             summary['timing'], '', summary['inference'], '',
             '| dataset | arm | cells | items | strict correct | task correct | cap | unparsed |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['arms']:
        lines.append('| ' + ' | '.join(str(row[k]) for k in
                     ('dataset', 'arm', 'cells', 'items', 'strict_correct', 'task_correct', 'capped', 'unparsed')) + ' |')
    lines += ['', summary['absolute_statistics'], '',
              '| dataset | arm | W mean / median (s) | S mean / median (s) | prefill mean (s) | denoise N mean / median / sum | S/N geometric mean (s) | commit forwards mean (observed cells) |',
              '|---|---|---|---|---:|---|---:|---|']
    for row in summary['arms']:
        commit = ('unreported' if row['commit_forward_count_mean'] is None else
                  f"{row['commit_forward_count_mean']:.4f} ({row['commit_forward_count_observed_cells']})")
        lines.append(f"| {row['dataset']} | {row['arm']} | "
                     f"{row['wall_s_mean']:.4f} / {row['wall_s_median']:.4f} | "
                     f"{row['decode_span_s_mean']:.4f} / {row['decode_span_s_median']:.4f} | "
                     f"{row['prefill_s_mean']:.4f} | "
                     f"{row['denoise_forward_count_mean']:.4f} / {row['denoise_forward_count_median']:.4f} / "
                     f"{row['denoise_forward_count_sum']} | {row['decode_s_per_denoise_forward_geomean']:.6f} | {commit} |")
    lines += ['', 'Ratios are candidate/reference; values below one mean faster or fewer forwards.', '',
              '| dataset | comparison | cells | items | correct (base) | W [95% CI] | S [95% CI] | prefill P [95% CI] | S/N [95% CI] | N [95% CI] | accuracy difference [95% CI] | exploratory McNemar p |',
              '|---|---|---:|---:|---|---|---|---|---|---|---|---:|']
    for row in summary['comparisons']:
        lines.append(f"| {row['dataset']} | {row['arm']}/{row['base']} | {row['cells']} | {row['items']} | "
                     f"{row['strict_correct']} ({row['base_strict_correct']}) | " +
                     ' | '.join(metric(row, name) for name in ('W', 'S', 'P', 'SN', 'N', 'accuracy_difference')) +
                     f" | {row['exploratory_mcnemar_exact_p']:.4g} |")
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', action='extend', type=Path, required=True)
    parser.add_argument('--completions', nargs='+', action='extend', type=Path, required=True)
    parser.add_argument('--gold', action='append', required=True, metavar='DATASET=PATH')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--nemo-root', type=Path)
    parser.add_argument('--bootstrap-reps', type=int, default=4000)
    args = parser.parse_args(argv)
    gold_paths = {}
    for entry in args.gold:
        dataset, separator, path = entry.partition('=')
        if not separator or not dataset or not path or dataset in gold_paths:
            raise ValueError('--gold needs unique DATASET=PATH entries')
        gold_paths[dataset] = Path(path)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    prevalidate_closed_runs(args.records, protocol)
    cells, completions = load_panel(args.records, args.completions, protocol)
    score_panel(cells, completions, gold_paths, args.nemo_root)
    summary = summarize(cells, args.bootstrap_reps)
    summary['protocol_id'] = protocol['protocol_id']
    summary['name'] = protocol['name']
    write_summary(summary, args.out_dir)
    print('Complete panel validated and aggregate summaries written.')
    return summary


if __name__ == '__main__':
    main()
