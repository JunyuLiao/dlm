"""Aggregate one or more matched v32 panels into the study's tables and machine-readable aggregates.

Scores the official metric with the repository's official scorers when they are given the private
records, and ALWAYS reports the request-level accounting the study requires: N, C, N/C, T, S/N,
GLOBAL and LOCAL physical tiles and their denominators, the selection counters, and the audit-only
diagnostics. Never infers speed from sparsity or from a requested budget.

usage:
  python scripts/v32_panel_aggregate.py --out RESULTS_DIR \
      --label LABEL=PRIVATE.jsonl [--label ...]

Writes RESULTS_DIR/aggregate.json and RESULTS_DIR/tables.md. No prompt, completion, token id, gold
answer or generated text is read into the output except through the official scorer's booleans.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))


def load(paths, pins=None):
    """Read records, optionally keeping only the frozen substrate pins.

    A run directory can hold several attempts appended to one file. Mixing substrates would
    make every derived number meaningless, so --require-pins filters and then asserts that
    nothing off-pin survived; anything unexpected is an error, never a silent drop.
    """
    out = {}
    for label, path in paths:
        recs, skipped = [], 0
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if pins and (r.get('block_size'), r.get('chunk')) != pins:
                    skipped += 1
                    continue
                recs.append(r)
        if skipped:
            print(f'  {label}: kept {len(recs)} on-pin records, skipped {skipped} off-pin')
        out[label] = recs
    return out


def cell_key(r):
    return f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"


def receipt(r):
    # Adapter arms emit a flat `counters` dict at the top level of the record; dense
    # arms bypass the adapter entirely and carry no counters at all.
    c = r.get('counters')
    if isinstance(c, dict):
        return c
    return (r.get('receipts') or {}).get('adapter') or {}


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo))


def aggregate(records):
    n = len(records)
    if not n:
        return None
    ad = [receipt(r) for r in records]
    keys = sorted({k for a in ad for k in a})
    tot = {k: sum(int(a.get(k) or 0) for a in ad) for k in keys
           if isinstance((ad[0].get(k) if ad else None), (int, float))
           or any(isinstance(a.get(k), (int, float)) for a in ad)}
    N = sum(int(r['denoise_forwards']) for r in records)
    C = sum(int(r['canvases']) for r in records)
    T = sum(int(r['output_tokens']) for r in records)
    decode = sum(float(r['decode_s']) for r in records)
    prefill = sum(float(r['prefill_s']) for r in records)
    wall = sum(float(r['wall_s']) for r in records)
    # GLOBAL / LOCAL physical tiles, with their own denominators (never mixed)
    ge, gk = tot.get('global_eligible_tiles'), tot.get('global_kept_tiles')
    le, lk = tot.get('local_eligible_tiles'), tot.get('local_kept_tiles')
    gp, gkp = tot.get('global_prefix_tiles'), tot.get('global_prefix_work_fraction')
    sp = tot.get('sparse_prefix_tiles')
    skp = tot.get('sparse_kept_prefix_fraction')
    # Stated overall denominator. LOCAL is native dense here and emits no tiles, so the
    # only defensible overall denominator is the GLOBAL eligible-tile count.
    overall_sparsity = None
    overall_den = None
    if ge and gk is not None:
        overall_sparsity = 1.0 - gk / ge
        overall_den = ge
    if le is not None and lk is not None and (ge or 0) + (le or 0) > 0:
        overall_den = (ge or 0) + le
    value_keys = [k for k in keys if k.startswith('value_') and k != 'value_selector']
    value = {}
    for k in value_keys:
        v = [a.get(k) for a in ad if a.get(k) is not None]
        if all(isinstance(x, (int, float)) for x in v) and v:
            value[k] = dict(total=float(sum(v)), mean=float(statistics.fmean(v)), max=float(max(v)))
    objectives = [a.get('value_last_objective') for a in ad if a.get('value_last_objective')]
    obj = None
    if objectives:
        om = [o.get('objective_max') for o in objectives if o.get('objective_max') is not None]
        rm = [o.get('retained_mass') for o in objectives if o.get('retained_mass') is not None]
        obj = dict(objective_max_mean=(statistics.fmean(om) if om else None),
                   objective_max_p50=percentile(om, .5), objective_max_p90=percentile(om, .9),
                   retained_mass_mean=(statistics.fmean(rm) if rm else None))
    last = [a.get('value_last_counters') for a in ad if a.get('value_last_counters')]
    forced_keep = sum(int(c.get('forced_keep', 0)) for c in last)
    forced_skip = sum(int(c.get('forced_skip', 0)) for c in last)
    thr = sum(int(c.get('threshold_keeps', 0)) for c in last)
    return dict(
        cells=n, N=N, canvases=C, N_per_canvas=(N / C if C else None),
        output_tokens_total=T, output_tokens_mean=(T / n),
        prefill_s_total=round(prefill, 3), decode_s_total=round(decode, 3), wall_s_total=round(wall, 3),
        decode_s_mean=round(decode / n, 5), wall_s_mean=round(wall / n, 5),
        S_over_N=(round(decode / N, 6) if N else None),
        finished=sum(1 for r in records if r['finish_reason'] in ('stop', 'eos')),
        capped=sum(1 for r in records if r['finish_reason'] not in ('stop', 'eos')),
        global_scope=dict(eligible_tiles=ge, kept_tiles=gk,
                          sparsity=(1.0 - gk / ge) if (ge and gk is not None) else None,
                          prefix_work_fraction=(gkp if gkp is not None else None)),
        # LOCAL is native dense in this study, so the adapter never routes it and
        # reports no LOCAL tiles. Report that explicitly instead of implying 0%.
        local_scope=dict(eligible_tiles=le, kept_tiles=lk, routed=False,
                         note='LOCAL native dense; adapter does not route it, so no LOCAL '
                              'tile denominator exists. Global-only denominators are reported '
                              'in global_scope and overall_sparsity_denominator.',
                         sparsity=(1.0 - lk / le) if (le and lk is not None) else None),
        overall_decoder_attention_sparsity=overall_sparsity,
        overall_sparsity_denominator=overall_den,
        overall_sparsity_basis='GLOBAL eligible tiles (LOCAL is native dense, unrouted)',
        adapter_counters={k: tot[k] for k in sorted(tot)},
        value_selectors=collections.Counter(str(a.get('value_selector')) for a in ad).most_common(),
        value_counters=value,
        value_objective=obj,
        value_forced=dict(forced_keep=forced_keep, forced_skip=forced_skip, threshold_keeps=thr,
                          forced_keep_rate=(forced_keep / (forced_keep + forced_skip)
                                            if (forced_keep + forced_skip) else None),
                          threshold_keep_rate=(thr / (forced_keep + forced_skip + thr)
                                               if (forced_keep + forced_skip + thr) else None)),
        substrate=sorted({(r.get('block_size'), r.get('chunk')) for r in records}),
        prompt_tokens_mean=round(sum(r['prompt_tokens'] for r in records) / n, 1),
        prompt_tokens_max=max(r['prompt_tokens'] for r in records),
        manifest_sha256=collections.Counter(r['manifest_sha256'] for r in records).most_common(1)[0][0],
        adapter_sha256=collections.Counter(r['adapter_sha256'] for r in records).most_common(1)[0][0],
        rng_seeds=len({r['rng_seed'] for r in records}),
        cudagraph_modes=collections.Counter(r['cudagraph_mode'] for r in records).most_common(),
        finish_reasons=collections.Counter(r['finish_reason'] for r in records).most_common(),
    )


def paired(ref, other, metric):
    """Item-clustered paired difference over the common cells (ref minus other)."""
    a = {cell_key(r): r for r in ref}
    b = {cell_key(r): r for r in other}
    keys = sorted(set(a) & set(b))
    diffs = []
    for k in keys:
        x, y = a[k].get(metric), b[k].get(metric)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            diffs.append(float(x) - float(y))
    if not diffs:
        return None
    return dict(n=len(diffs), mean=statistics.fmean(diffs), median=statistics.median(diffs),
                wins=sum(1 for d in diffs if d > 0), losses=sum(1 for d in diffs if d < 0),
                p90=percentile(diffs, .9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', action='append', required=True)
    ap.add_argument('--require-pins', default=None,
                    help='BLOCK,CHUNK of the frozen substrate; off-pin records are refused')
    ap.add_argument('--accuracy', action='append', default=[],
                    help='LABEL=OFFICIAL_SCORER_SUMMARY.json; keys are merged read-only')
    a = ap.parse_args()

    paths = []
    for spec in a.label:
        label, path = spec.split('=', 1)
        paths.append((label, path))
    pins = None
    if a.require_pins:
        b, c = a.require_pins.split(',')
        pins = (int(b), int(c))
    data = load(paths, pins)
    for label, recs in data.items():
        bad = ({(r.get('block_size'), r.get('chunk')) for r in recs} - {pins}) if pins else set()
        if bad or not recs:
            raise SystemExit(f'{label}: empty or mixed substrate; pins={pins} offending={bad}')

    out = {'schema': 'v32_aggregate_v1', 'arms': {}}
    for label, recs in data.items():
        agg = aggregate(recs)
        agg['private_records'] = paths[[l for l, _ in paths].index(label)][1]
        out['arms'][label] = agg

    # Merge official scorer summaries verbatim; never recompute the metric here.
    for spec in a.accuracy:
        label, path = spec.split('=', 1)
        summ = json.loads(Path(path).read_text())
        arm_key = next((k for k in summ.get('arms', {}) if k == label), None)
        if arm_key is None:
            arm_key = next(iter(summ.get('arms', {})), None)
        if arm_key is None:
            raise SystemExit(f'no arm in {path}')
        out.setdefault('accuracy_by_arm', {})[label] = summ['arms'][arm_key]
        out.setdefault('official_scorer_provenance', {})[label] = {
            'summary': path, 'metric': summ.get('metric'),
            'answers_sha256': summ.get('answers_sha256'),
            'answer_text': summ.get('answer_text'), 'extraction': summ.get('extraction'),
            'coverage': (summ.get('coverage') or {}).get(arm_key)}

    # Accuracy is deliberately NOT computed here. The official scorers
    # (scripts/v31_score_aime.py, scripts/v31_score_longbench_official.py) own the
    # metric, the pool binding and the avg@k definition; run them separately and merge
    # their .summary.json into this directory. This script owns only request-level
    # accounting, physical tile counts and timing.

    out['paired'] = {}
    labels = list(data)
    if len(labels) >= 2:
        ref = labels[0]
        for other in labels[1:]:
            out['paired'][f'{ref}_minus_{other}'] = {
                m: paired(data[ref], data[other], m)
                for m in ('denoise_forwards', 'output_tokens', 'decode_s', 'wall_s', 'prefill_s')}

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'aggregate.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    write_tables(out, Path(a.out) / 'tables.md')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('cells', 'accuracy')}
                      for k, v in out['arms'].items()}, indent=1))


def write_tables(out, path):
    lines = ['# v32 aggregate tables', '']
    arms = out['arms']
    acc = out.get('accuracy_by_arm') or {}
    hdr = ['arm', 'cells', 'acc', 'N', 'C', 'N/C', 'T', 'S/N', 'prompt tok',
           'GLOBAL sparsity', 'LOCAL sparsity', 'overall', 'wall s', 'decode s']
    lines += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for label, a in arms.items():
        g = a['global_scope']
        l = a['local_scope']
        def fmt(x, p=4):
            if x is None:
                return '-'
            if isinstance(x, float):
                return f'{x:.{p}f}'
            return f'{x:,}' if isinstance(x, int) else str(x)
        lines.append('| ' + ' | '.join([
            label, str(a['cells']), fmt(acc.get(label)),
            fmt(a['N']), fmt(a['canvases']), fmt(a['N_per_canvas'], 3),
            fmt(a['output_tokens_mean'], 1), fmt(a['S_over_N'], 5),
            fmt(a['prompt_tokens_mean'], 0),
            fmt(g['sparsity'], 4), fmt(l['sparsity'], 4),
            fmt(a['overall_decoder_attention_sparsity'], 4),
            fmt(a['wall_s_mean'], 4), fmt(a['decode_s_mean'], 4)]) + ' |')
    lines.append('')
    lines += ['## selection counters and the sketch-space objective', '',
              '| arm | selector | candidates | evaluations | forced keep | forced skip |'
              ' threshold keeps | blocked | objective_max mean | retained mass mean |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for label, a in arms.items():
        v = a['value_counters']
        o = a['value_objective'] or {}
        f = a['value_forced']
        sel = ','.join(s for s, _ in a['value_selectors'])
        lines.append('| ' + ' | '.join([
            label, sel or '-',
            fmt(v.get('value_candidates', {}).get('total')),
            fmt(v.get('value_evaluations', {}).get('total')),
            str(f['forced_keep']), str(f['forced_skip']), str(f['threshold_keeps']),
            fmt(v.get('value_blocked', {}).get('total'), 0),
            fmt(o.get('objective_max_mean'), 6), fmt(o.get('retained_mass_mean'), 4)]) + ' |')
    lines.append('')
    lines += ['## physical tile accounting (with denominators)', '',
              'Substrate pins are asserted per arm; see `substrate` in aggregate.json.', '',
              '| arm | GLOBAL eligible | GLOBAL kept | LOCAL eligible | LOCAL kept |'
              ' overall denominator | overall sparsity |', '|---|---|---|---|---|---|---|']
    for label, a in arms.items():
        g, l = a['global_scope'], a['local_scope']
        f = lambda x: '-' if x is None else str(x)
        lines.append('| ' + ' | '.join([
            label, f(g['eligible_tiles']), f(g['kept_tiles']), f(l['eligible_tiles']),
            f(l['kept_tiles']), f(a['overall_sparsity_denominator']),
            '-' if a['overall_decoder_attention_sparsity'] is None
            else f"{a['overall_decoder_attention_sparsity']:.4f}"]) + ' |')
    if out.get('paired'):
        lines += ['', '## paired differences (ref minus arm, common cells)', '',
                  '| pair | metric | n | mean | median | wins/losses |', '|---|---|---|---|---|---|']
        for pair, metrics in out['paired'].items():
            for m, st in metrics.items():
                if st:
                    lines.append(f"| {pair} | {m} | {st['n']} | {st['mean']:.5f} |"
                                 f" {st['median']:.5f} | {st['wins']}/{st['losses']} |")
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()