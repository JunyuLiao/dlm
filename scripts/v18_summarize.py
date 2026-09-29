"""Offline v18 scoring and question-cluster summaries; never used by GPU workers.

Gold is read only in this separate CPU process. Output contains aggregate counts
and intervals, with no raw completions, predictions, prompt text, or gold.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v18_evaluate import logical_protocol_digest, strict_warm, write_immutable_json
from scripts.v18_protocol import read_rows, sha

PAIRS = (('T60', 'U50'), ('T60', 'U60'), ('T50', 'U50'), ('U50', 'D_native'),
         ('T60', 'D_native'), ('D_matched', 'D_native'))


def scorer_source_identity(protocol, ruler_root=None):
    root = Path(__file__).resolve().parents[1]
    sources = {'v18_summarize': Path(__file__)}
    if protocol['stage'] == 'ruler4k_primary':
        if ruler_root is None:
            raise ValueError('pinned RULER checkout required for scoring')
        from dllm.evaluation.ruler import official
        provenance = official.verify_checkout(ruler_root)
        sources.update(ruler_wrapper=root / 'experiments/diffusion_gemma_ruler8k_jl.py',
                       ruler_official=root / 'src/dllm/evaluation/ruler/official.py',
                       nemo_ruler_metric=Path(ruler_root) / 'scripts/eval/synthetic/constants.py')
        source_revision = provenance['commit']
    elif protocol['stage'] == 'aime26_primary':
        sources.update(aime_final_response=root / 'experiments/diffusion_gemma_aime26_modes/protocol.py',
                       aime_numeric_extractor=root / 'experiments/diffusion_gemma_aime30/protocol.py')
        source_revision = None
    else:
        raise ValueError('unsupported frozen scorer stage')
    return dict(protocol_logical_sha256=protocol['logical_protocol_sha256'],
                source_manifest_sha256=protocol['source_manifest_sha256'],
                ruler_commit=source_revision,
                source_sha256={name: sha(path.read_bytes()) for name, path in sources.items()})


def check_scorer_lock(protocol, scorer_lock, ruler_root):
    expected = json.loads(Path(scorer_lock).read_text())
    actual = scorer_source_identity(protocol, ruler_root)
    if expected != actual:
        raise ValueError('scorer/extractor source differs from frozen scorer lock')
    return actual


def load_records(protocol, ledger_paths):
    schedule = {execution_key(e): e for e in protocol['schedule']}
    if len(schedule) != len(protocol['schedule']):
        raise ValueError('duplicate frozen schedule key')
    records = {}
    for path in ledger_paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get('event') != 'run':
                continue
            key = e.get('execution_key')
            if key not in schedule or key in records:
                raise ValueError('foreign or duplicate execution record')
            spec = schedule[key]
            if any(e.get(k) != spec[k] for k in ('arm', 'id', 'seed', 'role', 'repeat', 'cell_id', 'block')):
                raise ValueError('ledger record differs from schedule')
            assigned = protocol['block_assignments'][str(spec['block'])]
            if e.get('host') != assigned['host'] or e.get('gpu_uuid') != assigned['gpu_uuid']:
                raise ValueError('execution host/GPU differs from frozen block assignment')
            if e.get('generation_seed') != spec['seed']:
                raise ValueError('actual generation seed differs from schedule')
            records[key] = e
    return records


def routing_counts(receipt):
    buckets = {kind: Counter() for kind in ('whole', 'local', 'global')}
    rows = receipt.get('routing')
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError('routing records malformed')
    for row in rows:
        kind = row.get('attention_type')
        if kind not in ('local', 'global'):
            raise ValueError('unknown attention type')
        eligible, skipped = row.get('eligible'), row.get('skipped')
        if type(eligible) is not int or type(skipped) is not int or not 0 <= skipped <= eligible:
            raise ValueError('invalid physical tile counts')
        for bucket in ('whole', kind):
            buckets[bucket]['eligible'] += eligible
            buckets[bucket]['skipped'] += skipped
            for region in ('prefix', 'canvas', 'boundary'):
                r_eligible, r_skipped = row.get(region + '_eligible'), row.get(region + '_skipped')
                if type(r_eligible) is not int or type(r_skipped) is not int or not 0 <= r_skipped <= r_eligible:
                    raise ValueError('invalid regional physical tile counts')
                buckets[bucket][region + '_eligible'] += r_eligible
                buckets[bucket][region + '_skipped'] += r_skipped
    return {k: dict(v) for k, v in buckets.items()}


def score_one(dataset, gold_row, receipt, record, *, ruler_scorers=None):
    if dataset == 'aime26':
        from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
        expected = gold_row.get('expected', gold_row.get('answer'))
        if expected is None:
            raise ValueError('AIME scorer gold field absent')
        result = numeric_score(final_response(receipt['raw_completion'], True), str(expected))
        parsed = result['extracted'] is not None
        correct = bool(result['correct']) and record['termination'] == 'eos'
        score = float(correct)
    elif __import__('scripts.v27_datasets', fromlist=['x']).base_task(dataset) == 'ruler4k':
        from dllm.evaluation.ruler import official
        if ruler_scorers is None:
            raise ValueError('pinned official RULER scorer was not loaded')
        score = float(ruler_scorers[gold_row['task_base']](
            [official.postprocess_prediction(receipt['prediction'])], [gold_row['outputs']])) / 100.0
        correct, parsed = bool(score >= 1.0 - 1e-12), True
    else:
        raise ValueError('unknown task scorer')
    return dict(score=score, correct=correct, parsed=parsed, capped=record['termination'] == 'length',
                eos=record['termination'] == 'eos')


def bootstrap_question_clusters(cells, candidate, reference, *, task_by_id=None,
                                resamples=10000, seed=20260926):
    grouped = defaultdict(list)
    for (arm, rid, generation_seed), cell in cells.items():
        if arm != candidate:
            continue
        other = cells.get((reference, rid, generation_seed))
        if not other or cell.get('quality') is None or other.get('quality') is None:
            continue
        if (cell['attempt0']['host'], cell['attempt0']['gpu_uuid']) != (other['attempt0']['host'], other['attempt0']['gpu_uuid']):
            raise ValueError('paired arms were not run on the same host/GPU')
        qdiff = cell['quality']['score'] - other['quality']['score']
        timing = None if cell.get('warm_s') is None or other.get('warm_s') is None else math.log(cell['warm_s'] / other['warm_s'])
        calls = None if not cell.get('work') or not other.get('work') else math.log(
            cell['work']['decoder_calls'] / other['work']['decoder_calls'])
        canvases = None if not cell.get('work') or not other.get('work') else math.log(
            cell['work']['canvases'] / other['work']['canvases'])
        grouped[rid].append(dict(seed=generation_seed, quality=qdiff, timing=timing,
                                 calls=calls, canvases=canvases,
                                 host=cell['attempt0']['host'],
                                 candidate_warm_s=cell.get('warm_s'), reference_warm_s=other.get('warm_s'),
                                 candidate_calls=cell['work']['decoder_calls'] if cell.get('work') else None,
                                 reference_calls=other['work']['decoder_calls'] if other.get('work') else None,
                                 candidate_native_stop=cell['work']['native_stop_canvases'] if cell.get('work') else None,
                                 reference_native_stop=other['work']['native_stop_canvases'] if other.get('work') else None,
                                 candidate_capped=bool(cell['quality'].get('capped')),
                                 reference_capped=bool(other['quality'].get('capped'))))
    if not grouped:
        return dict(questions=0, paired_cells=0, quality_difference=None, time_geomean_ratio=None,
                    quality_difference_95=None, time_geomean_ratio_95=None)
    questions = sorted(grouped)
    by_task = defaultdict(list)
    for rid in questions:
        by_task[(task_by_id or {}).get(rid, 'all')].append(rid)
    qval = {rid: statistics.mean(row['quality'] for row in grouped[rid]) for rid in questions}
    def complete_metric(name):
        return {rid: statistics.mean(row[name] for row in grouped[rid]) for rid in questions
                if len(grouped[rid]) == 3 and all(row[name] is not None for row in grouped[rid])}
    tval, callval, canvasval = (complete_metric(name) for name in ('timing', 'calls', 'canvases'))
    def macro(sample, values):
        groups = defaultdict(list)
        for rid in sample:
            if rid in values:
                groups[(task_by_id or {}).get(rid, 'all')].append(values[rid])
        return statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None
    timed_questions = sorted(tval)
    timed_by_task = defaultdict(list)
    for rid in timed_questions:
        timed_by_task[(task_by_id or {}).get(rid, 'all')].append(rid)
    q_point, t_point = macro(questions, qval), macro(timed_questions, tval)
    rng = random.Random(seed)
    qboot, tboot = [], []
    for _ in range(resamples):
        sample = [rng.choice(ids) for ids in by_task.values() for _ in ids]
        qboot.append(macro(sample, qval))
        if timed_questions:
            timed_sample = [rng.choice(ids) for ids in timed_by_task.values() for _ in ids]
            tboot.append(math.exp(macro(timed_sample, tval)))
    def interval(values):
        if not values:
            return None
        values = sorted(values)
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
    per_seed = {str(s): macro([rid for rid in questions if any(row['seed'] == s for row in grouped[rid])],
                              {rid: next(row['quality'] for row in grouped[rid] if row['seed'] == s)
                               for rid in questions if any(row['seed'] == s for row in grouped[rid])})
                for s in sorted({row['seed'] for values in grouped.values() for row in values})}
    host_timing = defaultdict(list)
    for values in grouped.values():
        for row in values:
            if row['timing'] is not None:
                host_timing[row['host']].append(row['timing'])
    host_paired = {}
    for host in sorted(host_timing):
        matched = [row for values in grouped.values() for row in values
                   if row['host'] == host and row['timing'] is not None and
                   row['candidate_calls'] and row['reference_calls']]
        tc = sum(row['candidate_warm_s'] for row in matched)
        tr = sum(row['reference_warm_s'] for row in matched)
        cc = sum(row['candidate_calls'] for row in matched)
        cr = sum(row['reference_calls'] for row in matched)
        time_ratio = tc / tr if tr else None
        calls_ratio = cc / cr if cr else None
        per_call_ratio = (tc / cc) / (tr / cr) if cc and cr and tr else None
        host_paired[host] = dict(cells=len(matched), candidate_request_wall_sum_s=tc,
                                 reference_request_wall_sum_s=tr, total_time_ratio=time_ratio,
                                 decoder_call_ratio=calls_ratio, wall_per_call_ratio=per_call_ratio,
                                 decomposition_product=(calls_ratio * per_call_ratio)
                                 if calls_ratio is not None and per_call_ratio is not None else None)
    leave_one_out = [macro([other for other in questions if other != rid], qval) for rid in questions]
    leave_one_out = [v for v in leave_one_out if v is not None]
    return dict(questions=len(questions), paired_cells=sum(len(v) for v in grouped.values()),
                timed_questions=len(tval), task_strata={k: len(v) for k, v in sorted(by_task.items())},
                timed_task_strata={k: len(v) for k, v in sorted(timed_by_task.items())},
                quality_difference=q_point,
                time_geomean_ratio=math.exp(t_point) if t_point is not None else None,
                call_geomean_ratio=math.exp(macro(sorted(callval), callval)) if callval else None,
                canvas_geomean_ratio=math.exp(macro(sorted(canvasval), canvasval)) if canvasval else None,
                candidate_capped=sum(row['candidate_capped'] for values in grouped.values() for row in values),
                reference_capped=sum(row['reference_capped'] for values in grouped.values() for row in values),
                host_paired_time_geomean={host: math.exp(statistics.mean(logs))
                                           for host, logs in sorted(host_timing.items())},
                host_paired_decomposition=host_paired,
                candidate_native_stop_canvases=sum(row['candidate_native_stop'] or 0
                                                   for values in grouped.values() for row in values),
                reference_native_stop_canvases=sum(row['reference_native_stop'] or 0
                                                   for values in grouped.values() for row in values),
                quality_difference_95=interval(qboot), time_geomean_ratio_95=interval(tboot),
                per_seed_quality_difference=per_seed,
                leave_one_question_out_quality_range=[min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
                resamples=resamples, seed=seed)


def summarize(protocol_path, ledger_paths, gold_path, *, scorer_lock, ruler_root=None,
              private_roots=None, resamples=10000):
    protocol = json.loads(Path(protocol_path).read_text())
    if logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']:
        raise ValueError('frozen logical protocol changed')
    scorer_identity = check_scorer_lock(protocol, scorer_lock, ruler_root)
    if sha(Path(gold_path).read_bytes()) != protocol['source_manifest_sha256']:
        raise ValueError('scorer-only gold source byte identity differs from frozen source')
    gold_rows = read_rows(gold_path)
    by_id = {r['id']: r for r in gold_rows}
    if len(by_id) != len(gold_rows) or set(protocol['ids']) - set(by_id):
        raise ValueError('scorer-only ID coverage mismatch')
    records = load_records(protocol, ledger_paths)
    if protocol['stage'] == 'ruler4k_primary':
        from dllm.evaluation.ruler import official
        ruler_scorers = official.load_scorers(ruler_root)
    else:
        ruler_scorers = None
    private_roots = private_roots or {}
    warm_planned = {(e['arm'], e['id'], e['seed']) for e in protocol['schedule'] if e['role'] == 'warm'}
    unfinished_blocks = sorted({e['block'] for e in protocol['schedule']
                                if execution_key(e) not in records})
    cells = {}
    for entry in protocol['schedule']:
        key = (entry['arm'], entry['id'], entry['seed'])
        cell = cells.setdefault(key, dict(arm=entry['arm'], id=entry['id'], seed=entry['seed'], block=entry['block'],
                                          attempt0=None, warm=None, quality=None, warm_s=None, work=None))
        row = records.get(execution_key(entry))
        cell['attempt0' if entry['role'] == 'attempt0' else 'warm'] = row
    for cell in cells.values():
        first, warm = cell['attempt0'], cell['warm']
        if first is not None:
            if first.get('ok'):
                receipt_path = Path(first['private_receipt'])
                if first['host'] in private_roots:
                    if receipt_path.name != 'attempt00.json':
                        raise ValueError('private receipt name differs from first execution')
                    receipt_path = private_roots[first['host']] / 'cells' / first['cell_id'] / receipt_path.name
                receipt = json.loads(receipt_path.read_text())
                if (receipt['id'] != cell['id'] or receipt['seed'] != cell['seed'] or
                        receipt['prompt_token_hash'] != first['prompt_token_hash'] or
                        sha(json.dumps(receipt['completion_tokens'], separators=(',', ':'))) != first['completion_token_hash'] or
                        receipt['termination_reason'] != first['termination'] or
                        [c['decoder_calls'] for c in receipt['per_canvas']] != first['per_canvas_calls']):
                    raise ValueError('receipt/ledger identity mismatch')
                cell['quality'] = score_one(protocol['stage'].split('_primary')[0], by_id[cell['id']], receipt, first,
                                            ruler_scorers=ruler_scorers)
                cell['work'] = dict(decoder_calls=receipt['total_decoder_calls'], canvases=len(receipt['per_canvas']),
                                    native_stop_canvases=sum(c['native_stop_final_call'] for c in receipt['per_canvas']),
                                    iteration_cap_canvases=sum(c['iteration_cap_final_call'] for c in receipt['per_canvas']),
                                    output_tokens=receipt['output_tokens'], routing=routing_counts(receipt))
            else:
                cell['quality'] = dict(score=0.0, correct=False, parsed=False, capped=False, eos=False, failed=True)
        if warm is not None:
            acceptance = strict_warm(first, warm)
            if warm.get('acceptance') != acceptance:
                raise ValueError('stored warm acceptance differs from strict recomputation')
            if acceptance['accepted']:
                if type(warm.get('api_wall_s')) not in (int, float) or warm['api_wall_s'] <= 0:
                    raise ValueError('accepted warm missing positive request wall')
                cell['warm_s'] = warm['api_wall_s']
    arms = {}
    for arm in protocol['arms']:
        group = [c for c in cells.values() if c['arm'] == arm]
        counts = {kind: Counter() for kind in ('whole', 'local', 'global')}
        by_host = defaultdict(list)
        for cell in group:
            work = cell['work']
            if work and work['routing']:
                for kind, bucket in work['routing'].items():
                    counts[kind].update(bucket)
            if cell['warm_s'] is not None:
                by_host[cell['attempt0']['host']].append(cell['warm_s'])
        arms[arm] = dict(scheduled_cells=len(group), attempted=sum(c['attempt0'] is not None for c in group),
                         failures=sum(c['attempt0'] is not None and not c['attempt0'].get('ok') for c in group),
                         correct=sum(bool(c['quality'] and c['quality']['correct']) for c in group),
                         capped=sum(bool(c['quality'] and c['quality']['capped']) for c in group),
                         unparsed=sum(bool(c['quality'] and not c['quality']['parsed']) for c in group),
                         warm_scheduled=sum((arm, c['id'], c['seed']) in warm_planned for c in group),
                         warm_attempted=sum(c['warm'] is not None for c in group),
                         strict_warm_accepted=sum(c['warm_s'] is not None for c in group),
                         decoder_calls=sum(c['work']['decoder_calls'] for c in group if c['work']),
                         canvases=sum(c['work']['canvases'] for c in group if c['work']),
                         native_stop_canvases=sum(c['work']['native_stop_canvases'] for c in group if c['work']),
                         iteration_cap_canvases=sum(c['work']['iteration_cap_canvases'] for c in group if c['work']),
                         output_tokens=sum(c['work']['output_tokens'] for c in group if c['work']),
                         host_warm_request_wall={host: dict(accepted=len(values), sum_s=sum(values),
                                                            median_s=statistics.median(values))
                                                 for host, values in sorted(by_host.items())},
                         pv_bitmap={k: dict(v) for k, v in counts.items()},
                         qk_saved_measured=None,
                         qk_note='No physical QK-saving counter; U/T use fresh QK, bitmap skipped/eligible is PV support work')
    task_by_id = {rid: by_id[rid]['task'] for rid in protocol['ids']} if protocol['stage'] == 'ruler4k_primary' else None
    if task_by_id:
        if len(set(task_by_id.values())) != 13:
            raise ValueError('RULER scorer requires 13 task strata')
        for arm in protocol['arms']:
            task_scores = {}
            for task in sorted(set(task_by_id.values())):
                group = [c for c in cells.values() if c['arm'] == arm and task_by_id[c['id']] == task]
                scored = [c['quality']['score'] for c in group if c['quality'] is not None]
                task_scores[task] = dict(scored=len(scored), scheduled=len(group),
                                         mean=statistics.mean(scored) if scored else None)
            available = [v['mean'] for v in task_scores.values() if v['mean'] is not None]
            arms[arm]['task_scores'] = task_scores
            arms[arm]['task_macro_score'] = statistics.mean(available) if len(available) == 13 else None
    primary_cells = {key: cell for key, cell in cells.items() if cell['block'] not in unfinished_blocks}
    pairs = {f'{a}/{b}': bootstrap_question_clusters(primary_cells, a, b, task_by_id=task_by_id,
                                                      resamples=resamples)
             for a, b in PAIRS if a in arms and b in arms}
    return dict(schema='v18_redacted_summary_v1', protocol_id=protocol['protocol_id'],
                stage=protocol['stage'], planned_executions=protocol['planned_executions'],
                recorded_executions=len(records), complete=len(records) == len(protocol['schedule']),
                unfinished_blocks=unfinished_blocks,
                complete_blocks=protocol['planned_blocks'] - len(unfinished_blocks),
                calibration_sha256=protocol['calibration_sha256'],
                scorer_source_identity=scorer_identity,
                thresholds={a: {'attained': p['attained'], 'actual': p['actual'], 'points': p['points']}
                            for a, p in protocol['calibrated_points'].items() if a in arms},
                arms=arms, pairs=pairs,
                timing_note='Accepted warm whole-request wall only; GPU timeline from initial prefill end is separate, no direct full-forward latency inferred',
                generation_latency_s_qualified=None, time_between_tokens_s_qualified=None,
                quality_scope='arm totals are descriptive for all available first outputs; primary pairs use only complete logical blocks',
                bootstrap_note='Descriptive question-cluster bootstrap; seeds and arms stay paired, incomplete questions disclosed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--ledger', type=Path, nargs='+')
    p.add_argument('--gold', type=Path)
    p.add_argument('--ruler-root', type=Path)
    p.add_argument('--scorer-lock', type=Path, required=True)
    p.add_argument('--freeze-lock', action='store_true', help='CPU-only immutable scorer/extractor source lock')
    p.add_argument('--private-root', action='append', default=[], metavar='HOST=PATH',
                   help='Local copy of a host private root, repeated for copied host receipts')
    p.add_argument('--out', type=Path)
    p.add_argument('--resamples', type=int, default=10000)
    a = p.parse_args()
    if a.freeze_lock:
        protocol = json.loads(a.protocol.read_text())
        if logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']:
            p.error('frozen logical protocol changed')
        write_immutable_json(a.scorer_lock, scorer_source_identity(protocol, a.ruler_root))
        print(json.dumps({'scorer_lock_sha256': sha(a.scorer_lock.read_bytes()),
                          'stage': protocol['stage']}))
        return
    if not a.ledger or not a.gold or not a.out:
        p.error('--ledger, --gold and --out required for offline scoring')
    roots = {}
    for item in a.private_root:
        host, sep, path = item.partition('=')
        if not sep or not host or not path or host in roots:
            p.error('--private-root must be unique HOST=PATH')
        roots[host] = Path(path)
    result = summarize(a.protocol, a.ledger, a.gold, scorer_lock=a.scorer_lock,
                       ruler_root=a.ruler_root, private_roots=roots, resamples=a.resamples)
    write_immutable_json(a.out, result)
    print(json.dumps({'stage': result['stage'], 'recorded_executions': result['recorded_executions'],
                      'complete': result['complete']}))


if __name__ == '__main__':
    main()
