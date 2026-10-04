"""Score v31 private LongBench-v2 completions of the OFFICIAL-protocol pools with the official extraction and
aggregation (THUDM/LongBench @ 2e00731f pred.py extract_answer, result.py).

Rule: scripts/v31_official.py (planned cells, binding). Per completion: answer text = final_response(raw, row thinking)
-- the final response after the thinking channel (thinking ON and no '<channel|>' = the thought never closed -> '') --,
then pred.py's extract_answer (strip '*', 'The correct answer is (X)', then 'The correct answer is X'); judge = pred ==
gold; an unparsed answer (None) is wrong. Every pool item counts (pred.py only skips an item whose API call returned '',
an error path). Headline columns: longbench_v2_0shot (w/o CoT) and longbench_v2_0shot_think (w/ CoT);
longbench_v2_cot_think is not a headline column.
Outputs (OUT_PREFIX):
  .official.json             {arm: {cell: bool}}  PRIMARY: pred.py's judge, no finish requirement
  .judge_and_finished.json   {arm: {cell: bool}}  SECONDARY: judge AND a stop / eos finish
  .binding.json              per-cell run settings for the comparison tools (v31_paired_official.py longbench)
  .summary.json              per arm and dataset, per run (panel seed, repeat): result.py's Overall / Easy / Hard /
                             Short / Medium / Long (round(100 * acc / n, 1)), cells, finished / capped / other,
                             unparsed, no_final_response, truncated items; with several runs the mean over runs of the
                             UNROUNDED run values, rounded once; coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids or answers written. Run with cwd / PYTHONPATH holding experiments/.
usage: python v31_score_longbench_official.py OUT_PREFIX MANIFEST_DIR GOLD_DIR --cells CELLS.json [--repeats N]
           [--allow-missing] [--legacy-unbound] PRIVATE.jsonl [...]
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

SECONDARY = 'judge_and_finished'
FIELDS = ('difficulty', 'length', 'thinking', 'truncated')


def score(records, pools, gold_of, final_response, legacy=False):
    official, strict, binding, diag, meta = {}, {}, {}, {}, {}
    for label, key, r in records:
        row, b, verified = op.bind(r, pools, legacy)
        gold = gold_of(r['dataset'])[r['id']]
        if gold not in ('A', 'B', 'C', 'D'):
            raise ValueError('gold must be one of A-D')
        text = final_response(r['completion'], bool(row['thinking']))
        pred = op.lb_extract_answer(text)
        judge = pred == gold
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, judge)
        op.put(strict, label, key, judge and fc == 'finished')
        op.put(binding, label, key, b)
        meta[key] = dict(difficulty=row['difficulty'], length=row['length'], truncated=bool(row.get('truncated')))
        d = diag.setdefault((label, r['dataset']), collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['unparsed'] += pred is None
        d['no_final_response'] += text == ''
        d['truncated_items'] += bool(row.get('truncated'))
        d['verified' if verified else 'legacy_unbound'] += 1
    return official, {SECONDARY: strict}, binding, diag, meta


def summarize(official, diag, meta):
    out = {}
    for label, cells in official.items():
        runs = collections.defaultdict(list)
        for k, v in cells.items():
            ds, _, seed, repeat = k.split('|')
            runs[(ds, f'seed{seed}_repeat{repeat}')].append(dict(judge=v, **meta[k]))
        for ds in sorted({d for d, _ in runs}):
            per_run = {run: rows for (d, run), rows in sorted(runs.items()) if d == ds}
            entry = dict(runs={run: dict(op.lb_result(rows), items=len(rows)) for run, rows in per_run.items()},
                         **dict(diag[(label, ds)]))
            if len(per_run) > 1:
                raw = [op.lb_result(rows, digits=None) for rows in per_run.values()]
                entry['mean_over_runs'] = {c: round(statistics.mean(r[c] for r in raw), 1)
                                           for c in op.LB_COLUMNS if all(r[c] is not None for r in raw)}
            out.setdefault(label, {})[ds] = entry
    return out


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'manifest_dir', 'gold_dir'))
    final_response, source = op.load_final_response()
    golds = {}

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((Path(a.gold_dir) / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    records, coverage = op.planned_records(a)
    official, secondary, binding, diag, meta = score(records, op.Pools(a.manifest_dir, FIELDS), gold_of, final_response,
                                                     a.legacy_unbound)
    summary = dict(metric="LongBench-v2 pred.py extract_answer judge; result.py aggregation (one decimal)",
                   answer_text=source, coverage=coverage, arms=summarize(official, diag, meta))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    for label, per_ds in sorted(summary['arms'].items()):
        for ds, s in per_ds.items():
            for run, res in s['runs'].items():
                print(label, ds, run, ' '.join(f'{c}={res[c]}' for c in op.LB_COLUMNS), f"n={res['items']} "
                      f"capped={s.get('capped', 0)} unparsed={s['unparsed']}")


if __name__ == '__main__':
    main()
