"""v15 offline scoring + paired, question-clustered summaries of the scored LongBench-v2 panel (CPU only).

Run only after timing has stopped. Gold is read only here (private scorer-only file).
Strict ledger rules: a duplicate execution key, a record outside the frozen schedule (e.g. an
old AIME record), or a record whose id/seed/arm differs from its scheduled entry raises; nothing
is last-record-wins. Every scheduled cell appears in the outputs (missing = explicitly missing).
Quality: attempt-0 only (warm repeats are timing repetitions). task_correct (NeMo MCQ on the final
channel; valid final choice counts at a cap) and strict_correct (task_correct and EOS).
Time: accepted warm request wall (v15 warm acceptance). Ratios candidate/reference (<1 faster).
Exact pooled decomposition over cells where both arms have accepted timing:
  sum(t_c)/sum(t_r) = [sum(calls_c)/sum(calls_r)] x [(sum t_c/sum calls_c)/(sum t_r/sum calls_r)]
The paired geometric mean is reported separately and never multiplied by pooled factors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v15_seed_runs import warm_acceptance

PAIRS = [('CVM_T', 'B8_P'), ('CVM_T', 'T_P'), ('CVM_T', 'D_native'), ('CVM_T', 'T_G_original'), ('B8_P', 'D_native'),
         ('T_P', 'T_G_original'), ('B8_P', 'T_P'), ('T_G_original', 'D_native'), ('T_P', 'D_native')]


def load_records(events: list[dict], schedule: list[dict]) -> dict:
    by_key = {execution_key(e): e for e in schedule}
    records = {}
    for e in events:
        if e.get('event') != 'run':
            continue
        key = e.get('execution_key')
        if key not in by_key:
            raise ValueError(f'record outside the frozen schedule: {key}')
        if key in records:
            raise ValueError(f'duplicate execution record: {key}')
        s = by_key[key]
        for f in ('arm', 'id', 'seed', 'role', 'repeat', 'cell_id'):
            if e.get(f) != s[f]:
                raise ValueError(f'record {key} field {f}={e.get(f)!r} differs from schedule {s[f]!r}')
        if e.get('ok') and e.get('generation_seed') is not None and e['generation_seed'] != s['seed']:
            raise ValueError(f'record {key} generation seed differs from its schedule')
        records[key] = e
    return records


def score_quality(records: dict, gold: dict, read_receipt) -> dict:
    """cell_id -> quality dict for attempt-0 records (failures count as incorrect/unparsed)."""
    from scripts import v15_longbench_task as task
    ok = [r for r in records.values() if r['role'] == 'attempt0' and r.get('ok')]
    receipts = [read_receipt(r) for r in ok]
    for r, rec in zip(ok, receipts):
        if rec['seed'] != r['seed'] or str(rec['id']) != r['id']:
            raise ValueError('receipt identity differs from the scheduled cell')
    scored = task.score([rec['raw_completion'] for rec in receipts], [gold[r['id']] for r in ok],
                        [r['termination'] for r in ok]) if ok else []
    quality = {r['cell_id']: dict(s, correct=s['task_correct'], capped=r['termination'] == 'length',
                                  eos=r['termination'] == 'eos') for r, s in zip(ok, scored)}
    for r in records.values():
        if r['role'] == 'attempt0' and not r.get('ok'):
            quality[r['cell_id']] = dict(correct=False, task_correct=False, strict_correct=False, parsed=False, malformed=False,
                                         final_channel_present=False, capped=False, eos=False, failure=r.get('error'))
    return quality


def build_cells(schedule, records, quality):
    cells = {}
    for entry in schedule:
        key = (entry['arm'], entry['id'], entry['seed'])
        cell = cells.setdefault(key, dict(arm=entry['arm'], id=entry['id'], seed=entry['seed'], cell_id=entry['cell_id'],
                                          attempt0=None, warm=[], quality=None))
        record = records.get(execution_key(entry))
        if entry['role'] == 'attempt0':
            cell['attempt0'], cell['quality'] = record, quality.get(entry['cell_id'])
        else:
            cell['warm'].append(record)
    for cell in cells.values():
        accepted, reasons = [], set()
        for w in cell['warm']:
            if w is None:
                continue
            acc = warm_acceptance(cell['attempt0'], w)
            (accepted.append(w['api_wall_s']) if acc['accepted'] else reasons.update(acc['reasons']))
        cell['time'] = accepted[0] if accepted else None
        cell['timing_status'] = ('accepted' if accepted else 'missing_warm' if not any(cell['warm'])
                                 else 'unaccepted:' + ','.join(sorted(reasons)))
    return cells


def q(cell, field='task_correct'):
    return int(bool(cell['quality'] and cell['quality'].get(field)))


def pair(cells, ids, seeds, cand, ref, field='task_correct'):
    dis = dict(both=0, candidate_only=0, reference_only=0, neither=0)
    logs, missing, T, C = [], [], [0., 0.], [0, 0]
    for i in ids:
        per = []
        for s in seeds:
            c, r = cells[(cand, i, s)], cells[(ref, i, s)]
            a, b = q(c, field), q(r, field)
            dis['both' if a and b else 'candidate_only' if a else 'reference_only' if b else 'neither'] += 1
            if c['time'] is None or r['time'] is None:
                missing.append(dict(id=i, seed=s, candidate=c['timing_status'], reference=r['timing_status']))
                continue
            per.append(math.log(c['time'] / r['time']))
            T[0] += c['time']; T[1] += r['time']
            C[0] += c['attempt0']['decoder_calls']; C[1] += r['attempt0']['decoder_calls']
        if len(per) == len(seeds):
            logs.append(statistics.mean(per))
    out = dict(candidate=cand, reference=ref, questions=len(ids), outcomes=len(ids) * len(seeds),
               candidate_correct=sum(q(cells[(cand, i, s)], field) for i in ids for s in seeds),
               reference_correct=sum(q(cells[(ref, i, s)], field) for i in ids for s in seeds),
               paired_outcomes=dis, timing_complete=not missing, missing_timings=missing,
               questions_with_complete_timing=len(logs))
    if logs:
        out['geometric_time_ratio'] = math.exp(statistics.mean(logs))
    if T[1] > 0:
        calls, per_call = C[0] / C[1], (T[0] / C[0]) / (T[1] / C[1])
        out.update(summed_time_ratio=T[0] / T[1], pooled_call_ratio=calls, pooled_time_per_call_ratio=per_call,
                   decomposition_check=abs(T[0] / T[1] - calls * per_call) < 1e-9, candidate_time_s=T[0], reference_time_s=T[1],
                   candidate_calls=C[0], reference_calls=C[1],
                   note='pooled time/call is amortized request wall per decoder call, not pure GPU latency')
    return out


def bootstrap(cells, ids, seeds, cand, ref, n=10000, seed=13):
    rng = random.Random(seed)
    per_q = {}
    for i in ids:
        cs = [cells[(cand, i, s)] for s in seeds]
        rs = [cells[(ref, i, s)] for s in seeds]
        ok = all(c['time'] and r['time'] for c, r in zip(cs, rs))
        per_q[i] = (statistics.mean(q(c) for c in cs) - statistics.mean(q(r) for r in rs),
                    statistics.mean(math.log(c['time'] / r['time']) for c, r in zip(cs, rs)) if ok else None,
                    sum(c['time'] for c in cs) if ok else None, sum(r['time'] for r in rs) if ok else None)
    timed = [i for i in ids if per_q[i][1] is not None]
    qd, geo, summ = [], [], []
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        qd.append(statistics.mean(per_q[i][0] for i in sample))
        if timed:
            ts = [timed[rng.randrange(len(timed))] for _ in timed]
            geo.append(math.exp(statistics.mean(per_q[i][1] for i in ts)))
            summ.append(sum(per_q[i][2] for i in ts) / sum(per_q[i][3] for i in ts))

    def iv(v):
        v = sorted(v)
        return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]] if v else None
    return dict(resamples=n, seed=seed, quality_diff_95=iv(qd), geometric_time_ratio_95=iv(geo), summed_time_ratio_95=iv(summ),
                timed_questions=len(timed), note='question clusters (seeds and arms travel together); descriptive, not a test')


def summarize(protocol, events, gold, read_receipt, boot=10000):
    schedule, ids, seeds = protocol['schedule'], protocol['ids'], protocol['seeds']
    records = load_records(events, schedule)
    quality = score_quality(records, gold, read_receipt)
    cells = build_cells(schedule, records, quality)
    arms = sorted(protocol['arms'])
    bins = {}
    for i, b in protocol.get('bins', {}).items():
        bins.setdefault(b, []).append(i)
    per_arm = {}
    for a in arms:
        cs = [cells[(a, i, s)] for i in ids for s in seeds]
        r0 = [c['attempt0'] or {} for c in cs]
        per_arm[a] = dict(
            task_correct=sum(q(c) for c in cs), strict_correct=sum(q(c, 'strict_correct') for c in cs), outcomes=len(cs),
            per_seed={str(s): sum(q(cells[(a, i, s)]) for i in ids) for s in seeds},
            questions_with_any_correct=sum(any(q(cells[(a, i, s)]) for s in seeds) for i in ids),
            unparsed=sum(1 for c in cs if c['quality'] and not c['quality'].get('parsed')),
            malformed=sum(1 for c in cs if c['quality'] and c['quality'].get('malformed')),
            no_final_channel=sum(1 for c in cs if c['quality'] and not c['quality'].get('final_channel_present')),
            caps=sum(r.get('termination') == 'length' for r in r0), eos=sum(r.get('termination') == 'eos' for r in r0),
            failures=sum(1 for r in r0 if r and not r.get('ok')), not_executed=sum(1 for r in r0 if not r),
            decoder_calls=sum(r.get('decoder_calls') or 0 for r in r0), canvases=sum(r.get('canvases') or 0 for r in r0),
            output_tokens=sum(r.get('output_tokens') or 0 for r in r0),
            warm_accepted=sum(c['time'] is not None for c in cs),
            warm_time_s=sum(c['time'] for c in cs if c['time'] is not None),
            attempt0_time_s=sum(r.get('api_wall_s') or 0 for r in r0),
            peak_allocated_gib_max=max((r.get('peak_allocated_bytes') or 0) for r in r0) / 2 ** 30)
        pa = per_arm[a]
        pa['calls_per_canvas'] = pa['decoder_calls'] / pa['canvases'] if pa['canvases'] else None
    pairs = {}
    for cand, ref in PAIRS:
        p = pair(cells, ids, seeds, cand, ref)
        p['strict'] = {k: v for k, v in pair(cells, ids, seeds, cand, ref, 'strict_correct').items()
                       if k in ('candidate_correct', 'reference_correct', 'paired_outcomes')}
        p['bootstrap'] = bootstrap(cells, ids, seeds, cand, ref, n=boot)
        p['per_seed'] = {str(s): {k: v for k, v in pair(cells, ids, [s], cand, ref).items()
                                  if k in ('candidate_correct', 'reference_correct', 'geometric_time_ratio', 'summed_time_ratio')}
                         for s in seeds}
        p['per_bin'] = {b: {k: v for k, v in pair(cells, members, seeds, cand, ref).items()
                            if k in ('candidate_correct', 'reference_correct', 'geometric_time_ratio', 'summed_time_ratio',
                                     'pooled_call_ratio', 'pooled_time_per_call_ratio')} for b, members in sorted(bins.items())}
        p['leave_one_question_out'] = {i: {k: v for k, v in pair(cells, [x for x in ids if x != i], seeds, cand, ref).items()
                                           if k in ('geometric_time_ratio', 'summed_time_ratio', 'candidate_correct', 'reference_correct')}
                                       for i in ids}
        pairs[f'{cand}/{ref}'] = p
    executed = sum(1 for e in schedule if execution_key(e) in records)
    meta = dict(protocol_id=protocol['protocol_id'], questions=len(ids), seeds=seeds, arms=arms,
                planned_executions=len(schedule), executed=executed, complete=executed == len(schedule),
                quality_denominator_per_arm=len(ids) * len(seeds))
    return dict(meta=meta, per_arm=per_arm, pairs=pairs), cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    sel = json.loads((a.protocol.parent / 'panel_selection.json').read_text())
    protocol['bins'] = {r['id']: r['bin'] for r in sel['panel']}
    events = [json.loads(l) for l in a.ledger.read_text().splitlines() if l.strip()]
    gold = json.loads(a.gold.read_text())
    summary, cells = summarize(protocol, events, gold, lambda r: json.loads(Path(r['private_receipt']).read_text()),
                               boot=a.bootstrap)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'complete_request_results.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n')
    dom = {r['id']: (r['domain'], r['bin'], r['prompt_tokens_n']) for r in sel['panel']}
    with (a.out / 'complete_request_results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'id', 'seed', 'domain', 'bin', 'prompt_tokens', 'cell_id', 'ok', 'task_correct', 'strict_correct',
                    'parsed', 'malformed', 'final_channel', 'termination', 'output_tokens', 'decoder_calls', 'canvases',
                    'calls_per_canvas', 'token_hash16', 'attempt0_api_wall_s', 'warm_api_wall_s', 'timing_status',
                    'peak_allocated_bytes', 'error'])
        for (arm, i, s), c in sorted(cells.items()):
            r, qq = c['attempt0'] or {}, c['quality'] or {}
            calls, canv = r.get('decoder_calls'), r.get('canvases')
            w.writerow([arm, i, s, *dom[i], c['cell_id'], r.get('ok'), qq.get('task_correct'), qq.get('strict_correct'),
                        qq.get('parsed'), qq.get('malformed'), qq.get('final_channel_present'), r.get('termination'),
                        r.get('output_tokens'), calls, canv, round(calls / canv, 3) if calls and canv else None,
                        (r.get('completion_token_hash') or '')[:16], r.get('api_wall_s'), c['time'], c['timing_status'],
                        r.get('peak_allocated_bytes'), r.get('error') or ''])
    print(json.dumps(dict(meta=summary['meta'], task={k: v['task_correct'] for k, v in summary['per_arm'].items()},
                          strict={k: v['strict_correct'] for k, v in summary['per_arm'].items()}), indent=1))
    for name, p in summary['pairs'].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()
                     if k in ('candidate_correct', 'reference_correct', 'geometric_time_ratio', 'summed_time_ratio',
                              'pooled_call_ratio', 'pooled_time_per_call_ratio', 'timing_complete')},
              p['paired_outcomes'], p['bootstrap']['geometric_time_ratio_95'])


if __name__ == '__main__':
    main()
