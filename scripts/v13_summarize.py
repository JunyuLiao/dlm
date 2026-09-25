"""v13 offline scoring + multi-seed, question-clustered summaries (CPU only).

Gold is read only here. Every scheduled cell (arm, question, seed) appears in the
outputs, including failures and caps. Time = the accepted single warm API wall
(same cell, no new compilation, reproduces attempt 0); a missing/unaccepted
timing is listed explicitly and makes any aggregate that needs it 'incomplete'
(never silently dropped). Ratios are candidate/reference (<1 = less time).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from scripts.v13_seed_runs import execution_key, warm_acceptance

PAIRS = [('G3', 'T_G'), ('G3', 'D_native'), ('G3', 'B8_G'), ('G1', 'G3'), ('G1', 'B8_G'), ('G1', 'T_G'),
         ('T_G', 'D_native'), ('B8_G', 'D_native'), ('G1', 'D_native')]


def build_cells(schedule: list[dict], records: dict, quality: dict) -> dict:
    """cells[(arm, id, seed)] from the frozen schedule + ledger records (by execution key)."""
    cells = {}
    for entry in schedule:
        key = (entry['arm'], entry['id'], entry['seed'])
        cell = cells.setdefault(key, dict(arm=entry['arm'], id=entry['id'], seed=entry['seed'], cell_id=entry['cell_id'],
                                          attempt0=None, warm=[], quality=None))
        record = records.get(execution_key(entry))
        if entry['role'] == 'attempt0':
            cell['attempt0'] = record
            cell['quality'] = quality.get(entry['cell_id'])
        else:
            cell['warm'].append(record)
    for cell in cells.values():
        accepted = []
        for w in cell['warm']:
            if w is None:
                continue
            acc = warm_acceptance(cell['attempt0'], w)
            w['_acceptance'] = acc
            if acc['accepted']:
                accepted.append(w['api_wall_s'])
        cell['time'] = accepted[0] if accepted else None
        cell['timing_status'] = ('accepted' if accepted else 'missing_warm' if not any(cell['warm']) else
                                 'unaccepted:' + ','.join(sorted({r for w in cell['warm'] if w for r in w['_acceptance']['reasons']})))
    return cells


def correct(cell) -> int:
    return int(bool(cell['quality'] and cell['quality'].get('correct')))


def pair_stats(cells, ids, seeds, cand, ref):
    """Quality + time for one ordered pair over the given questions."""
    q_diff, logs, missing, sums, calls = [], [], [], [0., 0.], [0, 0]
    disagreements = dict(both=0, candidate_only=0, reference_only=0, neither=0)
    for q in ids:
        per_seed_log, qc, qr = [], [], []
        for s in seeds:
            c, r = cells[(cand, q, s)], cells[(ref, q, s)]
            a, b = correct(c), correct(r)
            qc.append(a), qr.append(b)
            disagreements[('both' if a and b else 'candidate_only' if a else 'reference_only' if b else 'neither')] += 1
            if c['time'] is None or r['time'] is None:
                missing.append(dict(id=q, seed=s, candidate=c['timing_status'], reference=r['timing_status']))
                continue
            per_seed_log.append(math.log(c['time'] / r['time']))
            sums[0] += c['time']; sums[1] += r['time']
            calls[0] += c['attempt0']['decoder_calls']; calls[1] += r['attempt0']['decoder_calls']
        q_diff.append(statistics.mean(qc) - statistics.mean(qr))
        if len(per_seed_log) == len(seeds):
            logs.append(per_seed_log)
    complete = not missing
    out = dict(n_questions=len(ids), seeds=list(seeds), quality_diff_mean_per_question=statistics.mean(q_diff),
               candidate_correct=sum(correct(cells[(cand, q, s)]) for q in ids for s in seeds),
               reference_correct=sum(correct(cells[(ref, q, s)]) for q in ids for s in seeds),
               outcomes=len(ids) * len(seeds), paired_disagreements=disagreements,
               timing_complete=complete, missing_timings=missing)
    if complete:
        geo = math.exp(statistics.mean(statistics.mean(x) for x in logs))
        summed = sums[0] / sums[1]
        call_factor = calls[0] / calls[1]
        out.update(geometric_time_ratio=geo, summed_time_ratio=summed, speedup_geometric=1 / geo,
                   speedup_summed=1 / summed, candidate_time_s=sums[0], reference_time_s=sums[1],
                   candidate_calls=calls[0], reference_calls=calls[1], call_factor=call_factor,
                   amortized_time_per_call_factor=summed / call_factor)
    return out


def bootstrap(cells, ids, seeds, cand, ref, n=10000, seed=13):
    """Question-cluster bootstrap (both seeds and both arms travel together)."""
    rng = random.Random(seed)
    q_vals, geo_vals, sum_vals = [], [], []
    per_q = {}
    for q in ids:
        qc = statistics.mean(correct(cells[(cand, q, s)]) for s in seeds)
        qr = statistics.mean(correct(cells[(ref, q, s)]) for s in seeds)
        times = [(cells[(cand, q, s)]['time'], cells[(ref, q, s)]['time']) for s in seeds]
        ok = all(a is not None and b is not None for a, b in times)
        per_q[q] = (qc - qr, statistics.mean(math.log(a / b) for a, b in times) if ok else None,
                    sum(a for a, _ in times) if ok else None, sum(b for _, b in times) if ok else None)
    time_ok = all(v[1] is not None for v in per_q.values())
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        q_vals.append(statistics.mean(per_q[q][0] for q in sample))
        if time_ok:
            geo_vals.append(math.exp(statistics.mean(per_q[q][1] for q in sample)))
            sum_vals.append(sum(per_q[q][2] for q in sample) / sum(per_q[q][3] for q in sample))

    def interval(values):
        if not values:
            return None
        values = sorted(values)
        return [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]
    return dict(resamples=n, seed=seed, quality_diff_95=interval(q_vals), geometric_time_ratio_95=interval(geo_vals),
                summed_time_ratio_95=interval(sum_vals), note='question clusters; 95% percentile intervals; not a noninferiority test')


def aggregate(cells, ids, seeds, dev_ids, pairs=PAIRS, boot=10000):
    arms = sorted({a for a, _, _ in cells})
    quality = {a: dict(per_seed={str(s): sum(correct(cells[(a, q, s)]) for q in ids) for s in seeds},
                       combined=sum(correct(cells[(a, q, s)]) for q in ids for s in seeds),
                       outcomes=len(ids) * len(seeds),
                       mean_per_question=statistics.mean(statistics.mean(correct(cells[(a, q, s)]) for s in seeds) for q in ids),
                       caps=sum(1 for q in ids for s in seeds if (cells[(a, q, s)]['attempt0'] or {}).get('termination') == 'length'),
                       failures=sum(1 for q in ids for s in seeds if not (cells[(a, q, s)]['attempt0'] or {}).get('ok')))
               for a in arms}
    out = dict(quality=quality, pairs={})
    reserved = [q for q in ids if q not in dev_ids]
    for cand, ref in pairs:
        name = f'{cand}/{ref}'
        p = pair_stats(cells, ids, seeds, cand, ref)
        p['bootstrap'] = bootstrap(cells, ids, seeds, cand, ref, n=boot)
        p['per_seed'] = {str(s): pair_stats(cells, ids, [s], cand, ref) for s in seeds}
        p['strata'] = {'development': pair_stats(cells, [q for q in ids if q in dev_ids], seeds, cand, ref)}
        if reserved:
            p['strata']['reserved'] = pair_stats(cells, reserved, seeds, cand, ref)
        p['leave_one_question_out'] = {q: {k: v for k, v in pair_stats(cells, [x for x in ids if x != q], seeds, cand, ref).items()
                                           if k in ('geometric_time_ratio', 'summed_time_ratio', 'quality_diff_mean_per_question')}
                                       for q in ids} if len(ids) > 1 else {}
        out['pairs'][name] = p
    return out


def score_attempt0(records, gold):
    from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
    quality = {}
    for r in records.values():
        if r.get('role') != 'attempt0':
            continue
        if not r.get('ok'):
            quality[r['cell_id']] = dict(correct=False, parsed=False, capped=False, failure=r.get('error'))
            continue
        receipt = json.loads(Path(r['private_receipt']).read_text())
        if receipt['seed'] != r['seed']:
            raise AssertionError('receipt seed differs from the scheduled seed')
        score = numeric_score(final_response(receipt['raw_completion'], True), gold[r['id']])
        capped = r['termination'] == 'length' and r['output_tokens'] >= 8192
        quality[r['cell_id']] = dict(correct=bool(score['correct']) and not capped, scorer_correct=bool(score['correct']),
                                     parsed=score['extracted'] is not None, capped=capped, eos=r['termination'] == 'eos')
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    events = [json.loads(l) for l in args.ledger.read_text().splitlines() if l.strip()]
    records = {e['execution_key']: e for e in events if e.get('event') == 'run'}
    gold = {str(r['id']): str(r['expected']) for r in json.loads(args.manifest.read_text())}
    quality = score_attempt0(records, gold)
    cells = build_cells(protocol['schedule'], records, quality)
    ids, seeds = protocol['ids'], protocol['seeds']
    summary = aggregate(cells, ids, seeds, protocol['development_ids'], boot=args.bootstrap)
    planned = len(protocol['schedule'])
    executed = sum(1 for e in protocol['schedule'] if execution_key(e) in records)
    meta = dict(protocol_id=protocol['protocol_id'], mode=protocol['mode'], questions=len(ids), seeds=seeds,
                arms=sorted(protocol['arms']), scope='GLOBAL-only temporal methods (layers 5,11,17,23,29); LOCAL native in all arms',
                planned_executions=planned, executed=executed, complete=executed == planned,
                effective_arm_fields=protocol['effective_arm_fields'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'paired_summary.json').write_text(json.dumps(dict(meta=meta, **summary), indent=2, sort_keys=True, default=str) + '\n')
    with (args.out / 'first_output_quality.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'id', 'seed', 'stratum', 'cell_id', 'ok', 'correct', 'parsed', 'capped', 'termination', 'output_tokens',
                    'decoder_calls', 'canvases', 'token_hash16', 'anchor_calls', 'reselect_calls', 'held_calls', 'error'])
        for (a, q, s), c in sorted(cells.items()):
            r, qq, ph = c['attempt0'] or {}, c['quality'] or {}, (c['attempt0'] or {}).get('phases') or {}
            w.writerow([a, q, s, 'development' if q in protocol['development_ids'] else 'reserved', c['cell_id'], r.get('ok'),
                        qq.get('correct'), qq.get('parsed'), qq.get('capped'), r.get('termination'), r.get('output_tokens'),
                        r.get('decoder_calls'), r.get('canvases'), (r.get('completion_token_hash') or '')[:16],
                        ph.get('anchor'), ph.get('reselect'), ph.get('held'), r.get('error') or ''])
    with (args.out / 'warm_request_times.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'id', 'seed', 'cell_id', 'attempt0_api_wall_s', 'warm_api_wall_s', 'warm_outer_wall_s', 'timing_status',
                    'warm_triton_misses', 'decoder_calls'])
        for (a, q, s), c in sorted(cells.items()):
            r = c['attempt0'] or {}
            for wr in c['warm']:
                wr = wr or {}
                w.writerow([a, q, s, c['cell_id'], r.get('api_wall_s'), wr.get('api_wall_s'), wr.get('outer_wall_s'),
                            c['timing_status'], wr.get('triton_misses'), r.get('decoder_calls')])
    with (args.out / 'trajectory_robustness.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'id', 'seed', 'decoder_calls', 'canvases', 'calls_per_canvas', 'output_tokens', 'termination',
                    'accepted_warm_s', 'amortized_ms_per_call'])
        for (a, q, s), c in sorted(cells.items()):
            r = c['attempt0'] or {}
            calls, canv = r.get('decoder_calls'), r.get('canvases')
            w.writerow([a, q, s, calls, canv, round(calls / canv, 3) if calls and canv else None, r.get('output_tokens'),
                        r.get('termination'), c['time'], round(1e3 * c['time'] / calls, 2) if c['time'] and calls else None])
    print(json.dumps(dict(meta={k: meta[k] for k in ('executed', 'planned_executions', 'complete')},
                          quality={a: v['combined'] for a, v in summary['quality'].items()}), indent=1))


if __name__ == '__main__':
    main()
