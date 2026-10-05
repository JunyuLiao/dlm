"""Offline scoring + redacted summary of the v9 clean request-timing ledger.

Runs in a separate process after generation. Reads gold only here. Emits
clean_request_timing.{json,csv}: absolute walls per execution, attempt-0
quality booleans (frozen final-channel scorer; cap/unparsed = failure), L/S
token/call equality, and paired L->S and D->L/S ratios on WARM repeats only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
    gold = {str(r['id']): str(r['expected']) for r in json.loads(args.manifest.read_text())}
    events = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
    start = next(e for e in events if e['event'] == 'start')
    runs = [e for e in events if e['event'] == 'run']
    for run in runs:
        if run.get('attempt0'):
            receipt = json.loads(Path(run['private_receipt']).read_text())
            score = numeric_score(final_response(receipt['raw_completion'], True), gold[run['id']])
            run['quality'] = dict(correct=bool(score['correct']) and run['termination'] == 'eos',
                                  scorer_correct=bool(score['correct']), unparsed=score['extracted'] is None,
                                  capped=run['termination'] == 'length' and run['output_tokens'] >= 8192)
    ids = sorted({r['id'] for r in runs})
    table = {}
    for id_ in ids:
        for label in 'DLS':
            mine = [r for r in runs if r['id'] == id_ and r['label'] == label]
            first = next(r for r in mine if r['attempt0'])
            warm = [r for r in mine if not r['attempt0']]
            table[f'{id_}|{label}'] = dict(
                id=id_, label=label, attempt0_api_wall_s=first['api_wall_s'],
                attempt0_outer_wall_s=first['outer_wall_s'],
                attempt0_new_triton_specializations=first['new_triton_specialization_total'],
                warm_api_walls_s=[r['api_wall_s'] for r in warm],
                warm_api_median_s=statistics.median([r['api_wall_s'] for r in warm]) if warm else None,
                warm_new_specializations=[r['new_triton_specialization_total'] for r in warm],
                warm_all_match_attempt0=all(all(r['matches_own_attempt0'].values()) for r in warm),
                decoder_calls=first['decoder_calls'], canvases=first['canvases'],
                per_canvas_calls=first['per_canvas_calls'], termination=first['termination'],
                output_tokens=first['output_tokens'], completion_token_hash=first['completion_token_hash'],
                quality=first['quality'],
                ms_per_decoder_call_warm=(1e3 * statistics.median([r['api_wall_s'] for r in warm])
                                          / first['decoder_calls']) if warm else None)
    paired = {}
    for id_ in ids:
        d, l, s = (table[f'{id_}|{x}'] for x in 'DLS')
        same = (l['completion_token_hash'] == s['completion_token_hash']
                and l['per_canvas_calls'] == s['per_canvas_calls'] and l['termination'] == s['termination'])
        entry = dict(L_S_tokens_calls_termination_identical=same)
        # Pair repeat r of L with repeat r of S (same round), per the alternation.
        rounds = sorted({r['round'] for r in runs if not r['attempt0']})
        pairs = []
        for rnd in rounds:
            row = {x: next(r for r in runs if r['id'] == id_ and r['label'] == x and r['round'] == rnd)
                   for x in 'DLS'}
            pairs.append(dict(round=rnd, order=row['L']['order'], D=row['D']['api_wall_s'],
                              L=row['L']['api_wall_s'], S=row['S']['api_wall_s'],
                              L_over_S=row['L']['api_wall_s'] / row['S']['api_wall_s'],
                              L_over_D=row['L']['api_wall_s'] / row['D']['api_wall_s'],
                              S_over_D=row['S']['api_wall_s'] / row['D']['api_wall_s']))
        entry['warm_pairs'] = pairs
        if pairs:
            entry['warm_L_over_S_median'] = statistics.median(p['L_over_S'] for p in pairs)
            entry['warm_S_over_D_median'] = statistics.median(p['S_over_D'] for p in pairs)
            entry['warm_L_over_D_median'] = statistics.median(p['L_over_D'] for p in pairs)
            entry['warm_S_minus_L_seconds_median'] = statistics.median(p['S'] - p['L'] for p in pairs)
        entry['attempt0_L_over_S_cold_inclusive'] = l['attempt0_api_wall_s'] / s['attempt0_api_wall_s']
        entry['note_D'] = (f"D trajectory differs from L/S (calls {d['decoder_calls']} vs {l['decoder_calls']}); "
                           'D ratios are request-level, not per-call, and include the native/legacy mask difference')
        paired[id_] = entry
    report = dict(schema='v9_clean_request_timing_v1', ledger=str(args.ledger),
                  ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
                  environment={k: start[k] for k in ('gpu', 'torch', 'cuda', 'cudnn', 'triton')},
                  config_fingerprints=start['fingerprints'], source_hashes=start['source_hashes'],
                  boundaries=dict(api_wall_s='synchronize -> adapter.generate -> synchronize (runner._one)',
                                  outer_wall_s='whole runner._one incl. binding install/teardown and receipt assembly'),
                  executions=len(runs), per_condition=table, paired=paired,
                  runs=[{k: v for k, v in r.items() if k not in ('counters', 'private_receipt')}
                        | dict(counters={k: r['counters'][k] for k in (
                            'summary_hits', 'summary_builds', 'summary_misses', 'summary_peak_bytes',
                            'score_peak_bytes', 'history_summary_peak_bytes', 'score_refresh_calls',
                            'preqk_consumer_calls', 'attention_calls') if r['counters'] and k in r['counters']}
                               if r.get('counters') else None)
                        for r in runs])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'clean_request_timing.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    with (args.out / 'clean_request_timing.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label', 'round', 'attempt0', 'order', 'api_wall_s', 'outer_wall_s', 'decoder_calls',
                         'canvases', 'termination', 'output_tokens', 'completion_token_hash', 'new_triton_specs',
                         'warm_status', 'matches_attempt0', 'correct'])
        for r in runs:
            writer.writerow([r['id'], r['label'], r['round'], r['attempt0'], '-'.join(r['order']),
                             f"{r['api_wall_s']:.3f}", f"{r['outer_wall_s']:.3f}", r['decoder_calls'],
                             r['canvases'], r['termination'], r['output_tokens'], r['completion_token_hash'][:16],
                             r['new_triton_specialization_total'], r['warm_status'],
                             None if r['attempt0'] else all(r['matches_own_attempt0'].values()),
                             r.get('quality', {}).get('correct') if r['attempt0'] else ''])
    print(json.dumps(paired, indent=1))


if __name__ == '__main__':
    main()
