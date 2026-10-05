"""v10 CP3: offline scoring + redacted complete-request panel summary.

Separate process after generation; the only place gold is read. Quality uses
attempt 0 only (cap/unparsed/timeout = failure). Timing repeats are checked
against their own attempt 0 and never replace it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', nargs='+', default=['O/P', 'O/D', 'O/T', 'P/D', 'T/D'],
                        help='numerator/denominator arm labels (v11: e.g. H1/T H3/T H1/D H3/D T/D)')
    parser.add_argument('--prefix', default='complete_request_results')
    args = parser.parse_args()
    from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
    gold = {str(r['id']): str(r['expected']) for r in json.loads(args.manifest.read_text())}
    events = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
    start = next(e for e in events if e['event'] == 'start')
    warmup = next((e for e in events if e['event'] == 'kernel_warmup'), None)
    runs = [e for e in events if e['event'] == 'run']
    for run in runs:
        if run.get('attempt0') and run.get('ok'):
            receipt = json.loads(Path(run['private_receipt']).read_text())
            answer = final_response(receipt['raw_completion'], True)
            score = numeric_score(answer, gold[run['id']])
            capped = run['termination'] == 'length' and run['output_tokens'] >= 8192
            run['quality'] = dict(correct=bool(score['correct']) and not capped, scorer_correct=bool(score['correct']),
                                  parsed=score['extracted'] is not None, capped=capped, eos=run['termination'] == 'eos',
                                  channel_policy='final channel, thinking ON (frozen scorer)')
        elif run.get('attempt0'):
            run['quality'] = dict(correct=False, failure=run.get('error'))
    ids = sorted({r['id'] for r in runs}, key=lambda x: int(x.split('/')[1]))
    arms = list(dict.fromkeys(r['label'] for r in runs))
    cells = {}
    for id_ in ids:
        for arm in arms:
            mine = [r for r in runs if r['id'] == id_ and r['label'] == arm]
            if not mine:
                continue
            first = next((r for r in mine if r.get('attempt0')), mine[0])
            warm = [r for r in mine if not r.get('attempt0') and r.get('ok')]
            wm = statistics.median([r['api_wall_s'] for r in warm]) if warm else None
            counters = first.get('counters') or {}
            cells[f'{id_}|{arm}'] = dict(
                id=id_, arm=arm, ok=first.get('ok'), error=first.get('error'), quality=first.get('quality'),
                termination=first.get('termination'), output_tokens=first.get('output_tokens'),
                decoder_calls=first.get('decoder_calls'), canvases=first.get('canvases'),
                per_canvas_calls=first.get('per_canvas_calls'), completion_token_hash=first.get('completion_token_hash'),
                attempt0_api_wall_s=first.get('api_wall_s'), attempt0_outer_wall_s=first.get('outer_wall_s'),
                attempt0_triton_misses=first.get('triton_misses'), attempt0_triton_miss_s=first.get('triton_miss_s'),
                warm_api_walls_s=[r['api_wall_s'] for r in warm], warm_api_median_s=wm,
                warm_repeats_match_attempt0=[all(r['matches_own_attempt0'].values()) for r in warm],
                warm_triton_misses=[r['triton_misses'] for r in warm],
                amortized_ms_per_decoder_call_warm=(1e3 * wm / first['decoder_calls']) if wm and first.get('decoder_calls') else None,
                initial_prefill_generation_split='N/A (no nonintrusive boundary in the API path)',
                tbt='N/A (no finalized output-ready events without extra synchronization)',
                peak_allocated_bytes=first.get('peak_allocated_bytes'), kernel_variant=first.get('kernel_variant'),
                telemetry=first.get('telemetry'),
                summary=dict((k, counters.get(k)) for k in ('summary_builds', 'summary_hits', 'summary_misses', 'summary_peak_bytes',
                                                            'summary_budget_bytes', 'score_peak_bytes', 'score_budget_bytes',
                                                            'history_summary_peak_bytes', 'per_tile_statistics')) if counters else None)
    paired = {}
    for a, b in (pair.split('/') for pair in args.pairs):
        ratios, sums = [], [0., 0.]
        for id_ in ids:
            x, y = cells.get(f'{id_}|{a}'), cells.get(f'{id_}|{b}')
            if x and y and x['warm_api_median_s'] and y['warm_api_median_s']:
                ratios.append(x['warm_api_median_s'] / y['warm_api_median_s'])
                sums[0] += x['warm_api_median_s']; sums[1] += y['warm_api_median_s']
        if ratios:
            paired[f'{a}/{b}'] = dict(n_questions=len(ratios), per_question=ratios,
                                      geometric_mean_of_warm_time_ratios=math.exp(sum(map(math.log, ratios)) / len(ratios)),
                                      ratio_of_summed_warm_times=sums[0] / sums[1])
    same = {}
    for id_ in ids:
        p, o = cells.get(f'{id_}|P'), cells.get(f'{id_}|O')
        if p and o:
            same[id_] = dict(tokens=p['completion_token_hash'] == o['completion_token_hash'],
                             calls=p['per_canvas_calls'] == o['per_canvas_calls'], termination=p['termination'] == o['termination'])
    quality = {arm: dict(correct=sum(bool((cells[f'{i}|{arm}']['quality'] or {}).get('correct')) for i in ids if f'{i}|{arm}' in cells),
                         n=sum(f'{i}|{arm}' in cells for i in ids)) for arm in arms}
    report = dict(schema='v10_complete_request_results_v1', ledger=str(args.ledger),
                  ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
                  identity=start['identity'], fingerprints=start['fingerprints'], source_hashes=start['source_hashes'],
                  kernel_warmup=warmup, executions=len(runs), failures=[r for r in runs if not r.get('ok')],
                  quality_attempt0=quality, P_O_same_trajectory=same, paired_warm=paired, cells=cells,
                  notes=['4 independent questions, seed 42: descriptive, not population noninferiority',
                         'D uses the native mask; P/O/T use the legacy Junyu local mask: D ratios are runtime references, '
                         'not a model-faithful dense speedup claim',
                         'amortized ms/call = warm API wall / decoder calls; includes prefill, commits, sampler; not model latency',
                         'T = fresh Junyu T via verified extension (collect=True keeps bitmap references; no per-step device work)'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.prefix}.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    with (args.out / f'{args.prefix}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'arm', 'row_type', 'index', 'ok', 'api_wall_s', 'outer_wall_s', 'decoder_calls', 'canvases',
                         'termination', 'output_tokens', 'token_hash16', 'matches_attempt0', 'correct_attempt0',
                         'triton_misses', 'triton_miss_s', 'kernel_variant', 'telemetry', 'error'])
        for r in runs:
            writer.writerow([r['id'], r['label'], 'attempt0' if r.get('attempt0') else 'timing_repeat', r['index'], r.get('ok'),
                             f"{r['api_wall_s']:.3f}" if r.get('api_wall_s') else '', f"{r['outer_wall_s']:.3f}",
                             r.get('decoder_calls'), r.get('canvases'), r.get('termination'), r.get('output_tokens'),
                             (r.get('completion_token_hash') or '')[:16],
                             '' if r.get('attempt0') or not r.get('ok') else all(r['matches_own_attempt0'].values()),
                             (r.get('quality') or {}).get('correct') if r.get('attempt0') else '',
                             r.get('triton_misses'), f"{r.get('triton_miss_s', 0):.2f}", r.get('kernel_variant'),
                             r.get('telemetry'), r.get('error') or ''])
    print(json.dumps(dict(quality=quality, paired={k: (round(v['geometric_mean_of_warm_time_ratios'], 4),
                                                       round(v['ratio_of_summed_warm_times'], 4)) for k, v in paired.items()},
                          same=same), indent=1))


if __name__ == '__main__':
    main()
