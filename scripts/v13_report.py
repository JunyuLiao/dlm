"""Render paired_summary.md tables from paired_summary.json (no hand arithmetic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, digits=3):
    return '—' if x is None else (f'{x:.{digits}f}' if isinstance(x, float) else str(x))


def interval(pair):
    return '—' if not pair else f'[{pair[0]:.3f}, {pair[1]:.3f}]'


def main() -> None:
    root = Path(sys.argv[1])
    s = json.loads((root / 'paired_summary.json').read_text())
    m = s['meta']
    lines = [f"# v13 paired summary ({m['protocol_id']})", '',
             f"Scope: {m['scope']}. Questions: {m['questions']}; generation seeds: {m['seeds']}; arms: {', '.join(m['arms'])}.",
             f"Executions: {m['executed']}/{m['planned_executions']} planned ({'complete' if m['complete'] else 'INCOMPLETE'}).",
             'Quality = attempt 0 (cap/unparsed/failure = incorrect). Time = accepted single warm API wall.',
             'Ratios are candidate/reference: < 1 means less time. Speedup = 1/ratio.', '',
             '## Quality (attempt 0)', '',
             '| arm | ' + ' | '.join(f'seed {x}' for x in m['seeds']) + ' | combined | mean per question | caps | failures | not executed |',
             '|---|' + '---:|' * (len(m['seeds']) + 5)]
    for arm, q in sorted(s['quality'].items()):
        lines.append(f"| {arm} | " + ' | '.join(f"{q['per_seed'][str(x)]}/{m['questions']}" for x in m['seeds']) +
                     f" | {q['combined']}/{q['outcomes']} | {q['mean_per_question']:.3f} | {q['caps']} | {q['failures']} | {q['not_executed']} |")
    lines += ['', '## Paired comparisons (all questions)', '',
              '| pair | quality diff / question [95% CI] | cand correct | ref correct | disagreements (cand-only/ref-only) '
              '| geometric time ratio [95% CI] | summed time ratio [95% CI] | call factor | time/call factor |',
              '|---|---|---:|---:|---|---|---|---:|---:|']
    for name, p in s['pairs'].items():
        b = p['bootstrap']
        d = p['paired_disagreements']
        lines.append(f"| {name} | {p['quality_diff_mean_per_question']:+.3f} {interval(b['quality_diff_95'])} | "
                     f"{p['candidate_correct']} | {p['reference_correct']} | {d['candidate_only']}/{d['reference_only']} | "
                     f"{fmt(p.get('geometric_time_ratio'))} {interval(b['geometric_time_ratio_95'])} | "
                     f"{fmt(p.get('summed_time_ratio'))} {interval(b['summed_time_ratio_95'])} | "
                     f"{fmt(p.get('call_factor'))} | {fmt(p.get('amortized_time_per_call_factor'))} |"
                     + ('' if p['timing_complete'] else f" (timing INCOMPLETE: {len(p['missing_timings'])} missing)"))
    lines += ['', '## Robustness', '', '| pair | seed | geometric | summed | quality diff |', '|---|---|---:|---:|---:|']
    for name, p in s['pairs'].items():
        for seed, ps in p['per_seed'].items():
            lines.append(f"| {name} | {seed} | {fmt(ps.get('geometric_time_ratio'))} | {fmt(ps.get('summed_time_ratio'))} | "
                         f"{ps['quality_diff_mean_per_question']:+.3f} |")
    lines += ['', '| pair | stratum | questions | geometric | summed | cand correct | ref correct |', '|---|---|---:|---:|---:|---:|---:|']
    for name, p in s['pairs'].items():
        for stratum, ps in p['strata'].items():
            lines.append(f"| {name} | {stratum} | {ps['n_questions']} | {fmt(ps.get('geometric_time_ratio'))} | "
                         f"{fmt(ps.get('summed_time_ratio'))} | {ps['candidate_correct']} | {ps['reference_correct']} |")
    lines += ['', '### Leave-one-question-out (range over dropped question)', '',
              '| pair | geometric min..max | summed min..max | question whose removal changes summed most |', '|---|---|---|---|']
    for name, p in s['pairs'].items():
        loo = p['leave_one_question_out']
        geo = [v['geometric_time_ratio'] for v in loo.values() if v.get('geometric_time_ratio') is not None]
        summ = {q: v['summed_time_ratio'] for q, v in loo.items() if v.get('summed_time_ratio') is not None}
        full = p.get('summed_time_ratio')
        worst = max(summ, key=lambda q: abs(summ[q] - full)) if summ and full else None
        lines.append(f"| {name} | {fmt(min(geo)) if geo else '—'}..{fmt(max(geo)) if geo else '—'} | "
                     f"{fmt(min(summ.values())) if summ else '—'}..{fmt(max(summ.values())) if summ else '—'} | "
                     f"{worst} ({fmt(summ.get(worst)) if worst else '—'}) |")
    (root / 'paired_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
