"""v27 Tier-3 summary from scored.csv files (no gold, no text; CPU only).

Per task and arm:
- first-output quality: strict correct, caps, EOS, unparsed; paired discordance vs native
  (arm-only-correct / native-only-correct cells);
- calls, canvases;
- paired ratios vs native (and vs D_matched) per question x seed cell: accepted warm
  request wall W, decoder calls N, and amortized per-call cost (W/N). Geometric means with
  a question-clustered bootstrap 95% interval (resample questions, keep both seeds and
  host of each question together), per-seed geometric means, and leave-one-question-out
  range.
W/N and S/N are amortized costs, not direct forward prices. S is the decode span that excludes
the initial prefill (first encoder forward end to the final CUDA event).
usage: python -m scripts.v27_panel_summary OUT.md OUT.csv SCORED.csv [...]
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def cluster_ci(cells, reps=4000, seed=7):
    """cells: {question: [ratio,...]} -> 95% interval of the geometric mean."""
    questions = list(cells)
    if len(questions) < 2:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [r for q in (rng.choice(questions) for _ in questions) for r in cells[q]]
        draws.append(geo(sample))
    draws.sort()
    return draws[int(.025 * reps)], draws[int(.975 * reps) - 1]


def summarize(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r['dataset'], r['id'], r['seed'])][r['arm']] = r
    # quality reference: D_native when the panel has it, else the fastest official dense (FA4 panels)
    ref = 'D_native' if any(r['arm'] == 'D_native' for r in rows) else 'D_fa4_allkept'
    arms = sorted({r['arm'] for r in rows}, key=lambda a: (a != ref, a != 'D_matched', a))
    out = []
    for dataset in sorted({k[0] for k in by}):
        cells = {k: v for k, v in by.items() if k[0] == dataset and ref in v}
        for arm in arms:
            present = {k: v for k, v in cells.items() if arm in v}
            if not present:
                continue
            first = [v[arm] for v in present.values()]
            ok = lambda r: r.get('first_status') == 'success'
            row = dict(dataset=dataset, arm=arm, cells=len(present), quality_reference=ref,
                       correct=sum(r['strict_correct'] == 'True' for r in first),
                       native_correct=sum(v[ref]['strict_correct'] == 'True' for v in present.values()),
                       arm_only=sum(v[arm]['strict_correct'] == 'True' and v[ref]['strict_correct'] != 'True'
                                    for v in present.values()),
                       native_only=sum(v[arm]['strict_correct'] != 'True' and v[ref]['strict_correct'] == 'True'
                                       for v in present.values()),
                       capped=sum(r['capped'] == 'True' for r in first),
                       unparsed=sum(r['parsed'] != 'True' for r in first),
                       failed=sum(not ok(r) for r in first),
                       calls=sum(int(r['decoder_calls'] or 0) for r in first),
                       score_mean=(round(statistics.mean(float(r['first_score']) for r in first
                                                         if r.get('first_score') not in ('', 'None', None)), 4)
                                   if any(r.get('first_score') not in ('', 'None', None) for r in first) else None))
            fak = {k: v for k, v in present.items() if 'D_fa4_allkept' in v}
            if fak:
                row.update(fak_cells=len(fak),
                           fak_correct=sum(v['D_fa4_allkept']['strict_correct'] == 'True' for v in fak.values()),
                           arm_only_vs_fak=sum(v[arm]['strict_correct'] == 'True' and
                                               v['D_fa4_allkept']['strict_correct'] != 'True' for v in fak.values()),
                           fak_only=sum(v[arm]['strict_correct'] != 'True' and
                                        v['D_fa4_allkept']['strict_correct'] == 'True' for v in fak.values()))
            for base in ('D_native', 'D_matched', 'D_c64', 'D_fa4', 'D_fa4_allkept'):
                ratios = defaultdict(lambda: defaultdict(list))
                per_seed = defaultdict(lambda: defaultdict(list))
                for (d, q, s), v in present.items():
                    if base not in v:
                        continue
                    a, b = v[arm], v[base]
                    try:
                        wa, wb = float(a['accepted_warm_request_wall_s']), float(b['accepted_warm_request_wall_s'])
                    except (TypeError, ValueError):
                        continue
                    na, nb = int(a['decoder_calls']), int(b['decoder_calls'])
                    pairs = [('W', wa / wb), ('N', na / nb), ('WN', (wa / na) / (wb / nb))]
                    try:   # decode span: first encoder forward end -> final event (excludes the initial prefill)
                        sa = float(a['accepted_warm_prefill_end_to_finish_cuda_span_s'])
                        sb = float(b['accepted_warm_prefill_end_to_finish_cuda_span_s'])
                        pairs += [('S', sa / sb), ('SN', (sa / na) / (sb / nb))]
                    except (TypeError, ValueError, KeyError, ZeroDivisionError):
                        pass
                    for key, value in pairs:
                        ratios[key][q].append(value)
                        per_seed[key][s].append(value)
                tag = {'D_native': 'nat', 'D_matched': 'dm', 'D_c64': 'c64', 'D_fa4': 'fa4',
                       'D_fa4_allkept': 'fak'}[base]
                for key in ('W', 'N', 'WN', 'S', 'SN'):
                    flat = [x for v in ratios[key].values() for x in v]
                    if not flat:
                        continue
                    lo, hi = cluster_ci(ratios[key])
                    loo = [geo([x for q2, v in ratios[key].items() if q2 != q for x in v]) for q in ratios[key]]
                    row[f'{key}_{tag}'] = round(geo(flat), 4)
                    row[f'{key}_{tag}_ci'] = f'[{lo:.3f},{hi:.3f}]' if lo else ''
                    if tag in ('nat', 'c64', 'fa4', 'fak'):
                        row[f'{key}_{tag}_by_seed'] = ' '.join(f'{s}:{geo(v):.3f}' for s, v in sorted(per_seed[key].items()))
                        row[f'{key}_{tag}_loo'] = f'{min(loo):.3f}-{max(loo):.3f}' if len(loo) > 1 else ''
            out.append(row)
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    md, csv_out, paths = argv[0], argv[1], argv[2:]
    rows = [r for p in paths for r in csv.DictReader(open(p, encoding='utf-8'))]
    summary = summarize(rows)
    fields = []
    for r in summary:
        fields += [k for k in r if k not in fields]
    with open(csv_out, 'w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    lines = []
    for dataset in sorted({r['dataset'] for r in summary}):
        lines += [f'### {dataset}', '',
                  '| arm | correct (native) | +/- vs native | cap | calls | W/native [CI] | N/native [CI] | (W/N)/native [CI] | (W/N)/D_matched | W by seed | W leave-one-question-out |',
                  '|---|---|---|---:|---:|---|---|---|---:|---|---|']
        for r in [x for x in summary if x['dataset'] == dataset]:
            lines.append(f"| {r['arm']} | {r['correct']}/{r['cells']} ({r['native_correct']}) | +{r['arm_only']}/-{r['native_only']} | "
                         f"{r['capped']} | {r['calls']} | {r.get('W_nat')} {r.get('W_nat_ci','')} | {r.get('N_nat')} {r.get('N_nat_ci','')} | "
                         f"{r.get('WN_nat')} {r.get('WN_nat_ci','')} | {r.get('WN_dm')} | {r.get('W_nat_by_seed','')} | {r.get('W_nat_loo','')} |")
        lines.append('')
        for tag, base in (('fak', 'D_fa4_allkept (fastest official dense)'), ('fa4', 'D_fa4 (FA4 plain dense path)')):
            if not any(f'W_{tag}' in x for x in summary if x['dataset'] == dataset):
                continue
            lines += [f'#### {dataset}: paired vs {base}', '',
                      f'| arm | correct (D_fa4_allkept) | +/- vs D_fa4_allkept | score mean | W [CI] | decode span S [CI] | calls N [CI] | S per call [CI] | W by seed | W leave-one-question-out |',
                      '|---|---|---|---:|---|---|---|---|---|---|']
            for r in [x for x in summary if x['dataset'] == dataset]:
                lines.append(f"| {r['arm']} | {r['correct']}/{r['cells']} ({r.get('fak_correct')}) | "
                             f"+{r.get('arm_only_vs_fak')}/-{r.get('fak_only')} | {r.get('score_mean')} | {r.get(f'W_{tag}')} {r.get(f'W_{tag}_ci', '')} | "
                             f"{r.get(f'S_{tag}')} {r.get(f'S_{tag}_ci', '')} | {r.get(f'N_{tag}')} {r.get(f'N_{tag}_ci', '')} | "
                             f"{r.get(f'SN_{tag}')} {r.get(f'SN_{tag}_ci', '')} | {r.get(f'W_{tag}_by_seed', '')} | "
                             f"{r.get(f'W_{tag}_loo', '')} |")
            lines.append('')
        if any('W_c64' in x for x in summary if x['dataset'] == dataset):
            lines += [f'#### {dataset}: paired vs D_c64 (strongest dense)', '',
                      '| arm | score mean | W/D_c64 [CI] | decode span S/D_c64 [CI] | N/D_c64 [CI] | S per call [CI] | W by seed | W leave-one-question-out |',
                      '|---|---:|---|---|---|---|---|---|']
            for r in [x for x in summary if x['dataset'] == dataset]:
                lines.append(f"| {r['arm']} | {r.get('score_mean')} | {r.get('W_c64')} {r.get('W_c64_ci', '')} | "
                             f"{r.get('S_c64')} {r.get('S_c64_ci', '')} | {r.get('N_c64')} {r.get('N_c64_ci', '')} | "
                             f"{r.get('SN_c64')} {r.get('SN_c64_ci', '')} | {r.get('W_c64_by_seed', '')} | "
                             f"{r.get('W_c64_loo', '')} |")
            lines.append('')
    open(md, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
