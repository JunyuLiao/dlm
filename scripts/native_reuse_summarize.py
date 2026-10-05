"""Offline, answer-free summary of native numerical-QK attempt-0 receipts.

Pass explicit run roots, and optionally offline quality JSONs made by
native_reuse_score.py. Every attempt-0 receipt stays in the denominator;
there is no timing or quality filter. No prompt, answer, or token sequence is
written. Per-token latency is unavailable without output-ready chunk events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


SCHEMA = 'numerical_qk_summary_v1'
ARMS = ('native_dense', 'fresh_junyu_T', 'M1', 'M3')
PAIR_CONTROL_FIELDS = (
    'manifest_sha256', 'model', 'model_metadata_hashes', 'revision', 'dtype',
    'max_new_tokens', 'thinking', 'temperature', 'eos_enabled',
    'native_adaptive', 'policy_sha256', 'policy_name', 'diagnostic',
    'timing_events',
)
CSV_FIELDS = (
    'phase', 'question', 'seed', 'arm', 'source_fingerprint',
    'whole_wall_seconds', 'device_generation_timeline_seconds',
    'device_generation_timeline_note', 'cpu_generation_seconds',
    'output_tokens', 'canvas_count', 'per_canvas_calls', 'total_calls',
    'capped', 'unparsed', 'correct', 'score_refresh_calls',
    'decision_refresh_calls', 'current_qk_elements', 'reused_score_elements',
    'pv_physical_eligible', 'pv_physical_skipped',
)
TIMING_CSV_FIELDS = (
    'phase', 'question', 'seed', 'arm', 'source_fingerprint',
    'original_quality_source_fingerprint', 'original_quality_receipt_sha256',
    'whole_wall_seconds', 'device_generation_timeline_seconds',
    'device_generation_timeline_note', 'cpu_generation_seconds', 'output_tokens',
    'canvas_count', 'per_canvas_calls', 'total_calls', 'score_refresh_calls',
    'decision_refresh_calls', 'current_qk_elements', 'reused_score_elements',
    'pv_physical_eligible', 'pv_physical_skipped', 'cross_load_token_match',
    'cross_load_call_match', 'quality_eligible', 'timing_retry_only',
    'independent_question_count_increment', 'performance_qualified',
    'retry_event_config_persisted', 'performance_note',
)
PAIRED_TIMING_CSV_FIELDS = (
    'phase', 'question', 'seed', 'arm', 'dense_timing_source_fingerprint',
    'method_source_fingerprint', 'whole_wall_speedup',
    'device_timeline_speedup', 'cross_load_token_match',
    'cross_load_call_match', 'configuration_basis',
    'performance_qualified', 'performance_note',
)


def _read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return result


def _identity(row: dict) -> tuple:
    return (row['phase'], row['condition'], str(row['id']), int(row['seed']))


def _config_key(config: dict) -> str | None:
    if any(field not in config for field in PAIR_CONTROL_FIELDS):
        return None
    shared = {field: config[field] for field in PAIR_CONTROL_FIELDS}
    return hashlib.sha256(json.dumps(shared, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _retry_controls_match(original: dict, method: dict,
                          retry_config: dict | None = None) -> bool:
    # Dense quality had events OFF; the separate retry and method had them ON.
    # The current batch script does not persist event_cfg. When one is present,
    # compare every common field; otherwise use the original dense config and
    # independently require an observed retry timeline and method events flag.
    if retry_config is not None:
        return all(x in retry_config and x in method and retry_config[x] == method[x]
                   for x in PAIR_CONTROL_FIELDS)
    fields = tuple(x for x in PAIR_CONTROL_FIELDS if x != 'timing_events')
    return (all(x in original and x in method and original[x] == method[x] for x in fields)
            and method.get('timing_events') is True)


def _routing_counts(routing: object) -> tuple[int | None, int | None]:
    if routing is None:
        return None, None
    if not isinstance(routing, list):
        raise ValueError('Routing must be a list or null')
    skipped = eligible = 0
    for item in routing:
        if 'physical_skipped' in item and 'physical_eligible' in item:
            s, e = int(item['physical_skipped']), int(item['physical_eligible'])
        elif 'skipped' in item and 'eligible' in item:
            s, e = int(item['skipped']), int(item['eligible'])
        else:
            raise ValueError('Unrecognized per-head routing counters')
        if not 0 <= s <= e:
            raise ValueError('Invalid physical skip/eligible counts')
        skipped += s
        eligible += e
    return skipped, eligible


def _quality_map(paths: list[Path]) -> tuple[dict, set]:
    records = {}
    arms = set()
    for path in paths:
        payload = _read(path)
        if payload.get('schema') != 'numerical_qk_quality_v1':
            raise ValueError(f'Invalid quality schema: {path}')
        for item in payload['records']:
            key = (item['phase'], item['condition'], str(item['id']), int(item['seed']))
            if key in records:
                raise ValueError(f'Duplicate offline quality record: {key}')
            if not isinstance(item.get('score'), dict) or 'correct' not in item['score']:
                raise ValueError(f'Missing frozen scorer result: {key}')
            records[key] = item
            arms.add((item['phase'], item['condition']))
    return records, arms


def collect(run_roots: list[Path], quality_files: list[Path] | None = None,
            timing_control_roots: list[Path] | None = None) -> tuple[list[dict], dict]:
    """Read explicit roots only; reject duplicate or non-attempt-0 records."""
    if not run_roots or len({p.resolve() for p in run_roots}) != len(run_roots):
        raise ValueError('Unique explicit run roots required')
    quality, scored_arms = _quality_map(quality_files or [])
    seen = {}
    configs = {}
    internal = []
    for root in run_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths = sorted(root.rglob('*.attempt0.json'))
        if not paths:
            raise ValueError(f'No attempt-0 receipts: {root}')
        for path in paths:
            receipt = _read(path)
            if receipt.get('schema') != 'numerical_qk_attempt0_v1' or receipt.get('attempt') != 0:
                raise ValueError(f'Invalid attempt-0 receipt: {path}')
            key = _identity(receipt)
            if key in seen:
                raise ValueError(f'Duplicate attempt-0 {key}: {seen[key]} and {path}')
            seen[key] = path
            phase, arm, question, seed = key
            if arm not in ARMS:
                raise ValueError(f'Unknown arm in receipt: {arm}')
            config_path = root / 'configs' / f'{phase}.{arm}.json'
            config = _read(config_path)
            if (not isinstance(receipt.get('fingerprint'), str) or not receipt['fingerprint'] or
                    config.get('fingerprint') != receipt['fingerprint'] or
                    config.get('condition') != arm or config.get('phase') != phase):
                raise ValueError(f'Config/receipt identity mismatch: {path}')
            configs[key] = config
            if not isinstance(receipt.get('prompt_hash'), str) or len(receipt['prompt_hash']) != 64:
                raise ValueError(f'Missing prompt hash: {path}')
            canvases = receipt.get('per_canvas')
            if not isinstance(canvases, list) or not canvases:
                raise ValueError(f'Missing actual per-canvas calls: {path}')
            calls = []
            for i, canvas in enumerate(canvases):
                n = canvas.get('decoder_calls')
                if canvas.get('canvas_index') != i or not isinstance(n, int) or n < 1:
                    raise ValueError(f'Invalid canvas calls: {path}')
                calls.append(n)
            if sum(calls) != receipt.get('total_decoder_calls'):
                raise ValueError(f'Total decoder calls disagree with canvases: {path}')
            output_tokens = receipt.get('output_tokens')
            if not isinstance(output_tokens, int) or output_tokens < 0:
                raise ValueError(f'Invalid output-token count: {path}')
            wall = receipt.get('request_wall_seconds')
            if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                raise ValueError(f'Invalid request wall time: {path}')
            timeline = receipt.get('generation_gpu_timeline_seconds')
            if timeline is not None and (not isinstance(timeline, (int, float)) or
                                         not math.isfinite(timeline) or timeline <= 0 or
                                         'host gaps' not in str(receipt.get('generation_gpu_timeline_note'))):
                raise ValueError(f'Invalid or unqualified device timeline: {path}')
            pv_skipped, pv_eligible = _routing_counts(receipt.get('routing'))
            counters = receipt.get('counters') or {}
            if not isinstance(counters, dict):
                raise ValueError(f'Invalid work counters: {path}')
            item = quality.get(key)
            if (phase, arm) in scored_arms and item is None:
                raise ValueError(f'Offline quality omits attempt-0: {key}')
            if item is not None and item.get('fingerprint') != receipt['fingerprint']:
                raise ValueError(f'Offline quality/receipt fingerprint mismatch: {key}')
            capped = receipt.get('termination_reason') == 'length' and output_tokens >= 8192
            if item is not None and bool(item.get('capped')) != capped:
                raise ValueError(f'Offline quality cap flag mismatch: {key}')
            row = dict(phase=phase, question=question, seed=seed, arm=arm,
                       source_fingerprint=receipt['fingerprint'],
                       whole_wall_seconds=float(wall),
                       device_generation_timeline_seconds=timeline,
                       device_generation_timeline_note=receipt.get('generation_gpu_timeline_note'),
                       cpu_generation_seconds=None, output_tokens=output_tokens,
                       canvas_count=len(calls), per_canvas_calls=calls,
                       total_calls=sum(calls), capped=capped,
                       unparsed=None if item is None else bool(item['unparsed']),
                       correct=None if item is None else bool(item['score']['correct']),
                       score_refresh_calls=counters.get('score_refresh_calls'),
                       decision_refresh_calls=counters.get('decision_refresh_calls'),
                       current_qk_elements=counters.get('current_qk_elements'),
                       reused_score_elements=counters.get('reused_qk_elements'),
                       pv_physical_eligible=pv_eligible,
                       pv_physical_skipped=pv_skipped)
            internal.append((row, receipt['prompt_hash'], _config_key(config)))
    if set(quality) - set(seen):
        raise ValueError(f'Offline quality references absent attempts: {len(set(quality)-set(seen))}')
    rows = [item[0] for item in internal]
    rows.sort(key=lambda x: (x['phase'], x['question'], x['seed'], ARMS.index(x['arm'])))
    summary = _summaries(internal)
    controls, pairs = _timing_controls(timing_control_roots or [], seen, configs, internal)
    summary['timing_controls'] = controls
    summary['paired_timing_controls'] = pairs
    summary['timing_control_scope'] = ('Retry-only, quality-ineligible, no new independent questions; '
                                       'cold/JIT state unknown, so observed ratios are unqualified performance')
    return rows, summary


def _timing_controls(roots: list[Path], seen: dict, configs: dict,
                     internal: list[tuple]) -> tuple[list[dict], list[dict]]:
    controls = []
    used = set()
    retry_configs = {}
    method_rows = {(r['phase'], r['question'], r['seed'], r['arm']): (r, prompt)
                   for r, prompt, _ in internal}
    for root in roots:
        directory = root / 'timing_controls' / 'native_dense'
        if not directory.is_dir() and root.name == 'native_dense':
            directory = root
        if not directory.is_dir():
            raise FileNotFoundError(f'Dense timing controls absent: {root}')
        paths = sorted(directory.glob('*.json'))
        if not paths:
            raise ValueError(f'No dense timing controls: {directory}')
        for path in paths:
            if path.name == 'config.json':
                continue
            control = _read(path)
            if (control.get('schema') != 'numerical_qk_dense_timing_control_v1' or
                    control.get('condition') != 'native_dense' or
                    control.get('quality_eligible') is not False or
                    control.get('timing_retry_only') is not True or
                    control.get('independent_question_count_increment') != 0):
                raise ValueError(f'Invalid retry-only dense timing control: {path}')
            question, seed = str(control['id']), int(control['seed'])
            key = ('smoke', 'native_dense', question, seed)
            if key in used:
                raise ValueError(f'Duplicate dense timing control: {key}')
            used.add(key)
            original_path = seen.get(key)
            if original_path is None:
                raise ValueError(f'Timing control lacks original quality attempt-0: {key}')
            if Path(control['original_quality_receipt']).resolve() != original_path.resolve():
                raise ValueError(f'Timing control original path mismatch: {path}')
            original_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
            if original_sha != control.get('original_quality_receipt_sha256'):
                raise ValueError(f'Timing control original SHA256 mismatch: {path}')
            original = _read(original_path)
            retry = control.get('retry')
            if not isinstance(retry, dict) or retry.get('schema') != 'numerical_qk_attempt0_v1':
                raise ValueError(f'Timing control missing full retry receipt: {path}')
            if (control.get('original_quality_fingerprint') != original['fingerprint'] or
                    control.get('original_quality_decoder_calls') != original['total_decoder_calls'] or
                    retry.get('phase') != 'observer_qualification' or
                    retry.get('condition') != 'native_dense' or retry.get('attempt') != 0 or
                    retry.get('id') != question or retry.get('seed') != seed or
                    retry.get('fingerprint') != control.get('retry_source_fingerprint') or
                    retry.get('prompt_hash') != original['prompt_hash'] or
                    control.get('retry_decoder_calls') != retry.get('total_decoder_calls')):
                raise ValueError(f'Timing control identity mismatch: {path}')
            config_path = directory / 'config.json'
            retry_config = _read(config_path) if config_path.is_file() else None
            if retry_config is not None and (retry_config.get('fingerprint') != retry['fingerprint'] or
                                             retry_config.get('condition') != 'native_dense' or
                                             retry_config.get('phase') != 'observer_qualification' or
                                             retry_config.get('timing_events') is not True):
                raise ValueError(f'Timing control event config mismatch: {config_path}')
            retry_configs[(question, seed)] = retry_config
            original_token_hash = _fingerprint(original['completion_tokens'])
            retry_token_hash = _fingerprint(retry['completion_tokens'])
            token_match = original_token_hash == retry_token_hash
            call_match = original['total_decoder_calls'] == retry['total_decoder_calls']
            if (control.get('original_quality_token_hash') != original_token_hash or
                    control.get('retry_token_hash') != retry_token_hash or
                    control.get('cross_load_token_match') is not token_match or
                    control.get('cross_load_call_match') is not call_match):
                raise ValueError(f'Timing control cross-load check mismatch: {path}')
            canvases = retry.get('per_canvas')
            if (not isinstance(canvases, list) or not canvases or
                    any(c.get('canvas_index') != i or not isinstance(c.get('decoder_calls'), int) or
                        c['decoder_calls'] < 1 for i, c in enumerate(canvases))):
                raise ValueError(f'Timing retry lacks valid per-canvas calls: {path}')
            calls = [c['decoder_calls'] for c in canvases]
            if sum(calls) != retry['total_decoder_calls']:
                raise ValueError(f'Timing retry total calls mismatch: {path}')
            wall = retry.get('request_wall_seconds')
            timeline = retry.get('generation_gpu_timeline_seconds')
            if (not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0 or
                    not isinstance(timeline, (int, float)) or not math.isfinite(timeline) or
                    timeline <= 0 or 'host gaps' not in str(retry.get('generation_gpu_timeline_note'))):
                raise ValueError(f'Timing retry lacks qualified measurement boundary: {path}')
            pv_skipped, pv_eligible = _routing_counts(retry.get('routing'))
            counters = retry.get('counters') or {}
            if not isinstance(counters, dict):
                raise ValueError(f'Timing retry work counters invalid: {path}')
            item = dict(phase='smoke', question=question, seed=seed,
                        arm='native_dense_timing_control', source_fingerprint=retry['fingerprint'],
                        original_quality_source_fingerprint=original['fingerprint'],
                        original_quality_receipt_sha256=original_sha,
                        whole_wall_seconds=float(wall),
                        device_generation_timeline_seconds=float(timeline),
                        device_generation_timeline_note=retry['generation_gpu_timeline_note'],
                        cpu_generation_seconds=None,
                        output_tokens=int(retry['output_tokens']),
                        canvas_count=len(calls), per_canvas_calls=calls,
                        total_calls=sum(calls),
                        score_refresh_calls=counters.get('score_refresh_calls'),
                        decision_refresh_calls=counters.get('decision_refresh_calls'),
                        current_qk_elements=counters.get('current_qk_elements'),
                        reused_score_elements=counters.get('reused_qk_elements'),
                        pv_physical_eligible=pv_eligible, pv_physical_skipped=pv_skipped,
                        cross_load_token_match=token_match, cross_load_call_match=call_match,
                        quality_eligible=False, timing_retry_only=True,
                        independent_question_count_increment=0,
                        retry_event_config_persisted=retry_config is not None,
                        performance_qualified=False,
                        performance_note='Cold/JIT state unknown; measured timing is descriptive')
            controls.append(item)
    controls.sort(key=lambda x: (x['question'], x['seed']))
    paired = []
    for control in controls:
        question, seed = control['question'], control['seed']
        original_config = configs[('smoke', 'native_dense', question, seed)]
        retry_config = retry_configs[(question, seed)]
        original_prompt = _read(seen[('smoke', 'native_dense', question, seed)])['prompt_hash']
        for arm in ARMS[1:]:
            method = method_rows.get(('smoke', question, seed, arm))
            if method is None:
                continue
            row, prompt = method
            method_config = configs[('smoke', arm, question, seed)]
            if (prompt != original_prompt or not _retry_controls_match(original_config, method_config, retry_config)
                    or not control['cross_load_token_match'] or not control['cross_load_call_match']):
                continue
            device = row['device_generation_timeline_seconds']
            if not isinstance(device, (int, float)) or not math.isfinite(device) or device <= 0:
                continue
            paired.append(dict(phase='smoke', question=question, seed=seed, arm=arm,
                               dense_timing_source_fingerprint=control['source_fingerprint'],
                               method_source_fingerprint=row['source_fingerprint'],
                               whole_wall_speedup=control['whole_wall_seconds']/row['whole_wall_seconds'],
                               device_timeline_speedup=control['device_generation_timeline_seconds']/device,
                               cross_load_token_match=True, cross_load_call_match=True,
                               configuration_basis=('persisted_retry_event_config' if retry_config is not None else
                                                    'original_dense_common_controls_plus_observed_event_timeline'),
                               performance_qualified=False,
                               performance_note='Cold/JIT state unknown; exploratory same-question timing ratio'))
    paired.sort(key=lambda x: (x['arm'], x['question'], x['seed']))
    return controls, paired


def _summaries(internal: list[tuple]) -> dict:
    by_arm = {}
    for row, _, _ in internal:
        by_arm.setdefault((row['phase'], row['arm']), []).append(row)
    arms = []
    for (phase, arm), group in sorted(by_arm.items()):
        scored = [x for x in group if x['correct'] is not None]
        if scored and len(scored) != len(group):
            raise ValueError(f'Incomplete quality denominator for {phase}/{arm}')
        arms.append(dict(phase=phase, arm=arm, attempts=len(group),
                         distinct_questions=len({x['question'] for x in group}),
                         seeds=sorted({x['seed'] for x in group}),
                         median_whole_wall_seconds=median(x['whole_wall_seconds'] for x in group),
                         mean_output_tokens=mean(x['output_tokens'] for x in group),
                         total_decoder_calls=sum(x['total_calls'] for x in group),
                         per_canvas_calls=[n for x in group for n in x['per_canvas_calls']],
                         capped=sum(x['capped'] for x in group),
                         unparsed=None if not scored else sum(x['unparsed'] for x in group),
                         correct=None if not scored else sum(x['correct'] for x in group),
                         accuracy=None if not scored else sum(x['correct'] for x in group)/len(group),
                         pv_physical_eligible=sum(x['pv_physical_eligible'] or 0 for x in group)
                             if all(x['pv_physical_eligible'] is not None for x in group) else None,
                         pv_physical_skipped=sum(x['pv_physical_skipped'] or 0 for x in group)
                             if all(x['pv_physical_skipped'] is not None for x in group) else None))
    dense = {}
    for row, prompt_hash, config_key in internal:
        if row['arm'] == 'native_dense':
            dense[(row['phase'], row['question'], row['seed'])] = (row, prompt_hash, config_key)
    pairs = []
    for row, prompt_hash, config_key in internal:
        if row['arm'] == 'native_dense':
            continue
        control = dense.get((row['phase'], row['question'], row['seed']))
        if control is None:
            continue
        base, base_prompt, base_config = control
        if prompt_hash != base_prompt or config_key is None or config_key != base_config:
            continue
        device_a = base['device_generation_timeline_seconds']
        device_b = row['device_generation_timeline_seconds']
        pairs.append(dict(phase=row['phase'], question=row['question'], seed=row['seed'],
                          arm=row['arm'], dense_source_fingerprint=base['source_fingerprint'],
                          method_source_fingerprint=row['source_fingerprint'],
                          whole_wall_speedup=base['whole_wall_seconds']/row['whole_wall_seconds'],
                          device_timeline_speedup=(device_a/device_b if isinstance(device_a, (int, float))
                                                   and isinstance(device_b, (int, float)) and device_b > 0 else None)))
    pairs.sort(key=lambda x: (x['phase'], x['arm'], x['question'], x['seed']))
    paired_arms = []
    for phase, arm in sorted({(x['phase'], x['arm']) for x in pairs}):
        group = [x for x in pairs if (x['phase'], x['arm']) == (phase, arm)]
        paired_arms.append(dict(phase=phase, arm=arm, pairs=len(group),
                                distinct_questions=len({x['question'] for x in group}),
                                geometric_mean_whole_wall_speedup=math.exp(mean(math.log(x['whole_wall_speedup']) for x in group)),
                                interpretation='Exploratory paired timing only; no noninferiority claim'))
    return dict(arms=arms, pairs=pairs, paired_arms=paired_arms,
                pairing_rule='Exact phase/question/seed/prompt hash and common config controls; arm-specific source fingerprints retained')


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if key == 'per_canvas_calls' else
                             'N/A' if row[key] is None else row[key] for key in fields})
    temporary.replace(path)


def write(rows: list[dict], summary: dict, json_path: Path, csv_path: Path,
          timing_csv_path: Path | None = None,
          paired_timing_csv_path: Path | None = None) -> None:
    if json_path.resolve() == csv_path.resolve():
        raise ValueError('JSON and CSV destinations must differ')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, attempts=len(rows), records=rows, summary=summary,
                   generation_timing_scope='Device timeline excludes initial encoder but includes host gaps and later encoder/commit work; CPU generation and time-between-tokens are N/A',
                   quality_denominator='All unique attempt-0 receipts; no timing filtering')
    json_tmp = json_path.with_suffix(json_path.suffix+'.tmp')
    json_tmp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    _write_csv(csv_path, CSV_FIELDS, rows)
    if summary.get('timing_controls'):
        timing_csv_path = timing_csv_path or csv_path.with_name(csv_path.stem+'.timing_controls.csv')
        paired_timing_csv_path = (paired_timing_csv_path or
                                  csv_path.with_name(csv_path.stem+'.paired_timing_controls.csv'))
        for extra in (timing_csv_path, paired_timing_csv_path):
            if extra.resolve() in (json_path.resolve(), csv_path.resolve()):
                raise ValueError('Timing CSV destinations must differ from primary outputs')
        if timing_csv_path.resolve() == paired_timing_csv_path.resolve():
            raise ValueError('Timing CSV destinations must differ')
        _write_csv(timing_csv_path, TIMING_CSV_FIELDS, summary['timing_controls'])
        _write_csv(paired_timing_csv_path, PAIRED_TIMING_CSV_FIELDS,
                   summary['paired_timing_controls'])
    json_tmp.replace(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, action='append', required=True)
    parser.add_argument('--quality', type=Path, action='append', default=[])
    parser.add_argument('--timing-control-root', type=Path, action='append', default=[])
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--timing-control-csv', type=Path)
    parser.add_argument('--paired-timing-control-csv', type=Path)
    args = parser.parse_args()
    rows, summary = collect(args.run_root, args.quality, args.timing_control_root)
    write(rows, summary, args.json, args.csv,
          args.timing_control_csv, args.paired_timing_control_csv)
    print(json.dumps(dict(attempts=len(rows), arms=len(summary['arms']),
                          pairs=len(summary['pairs']), timing_controls=len(summary['timing_controls']),
                          paired_timing_controls=len(summary['paired_timing_controls']),
                          json=str(args.json), csv=str(args.csv))))


if __name__ == '__main__':
    main()
