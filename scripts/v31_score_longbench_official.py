"""Score v31 private LongBench-v2 completions of the OFFICIAL-protocol pools with the official extraction and
aggregation (THUDM/LongBench @ 2e00731f pred.py extract_answer, result.py).

Rule: scripts/v31_official.py. Per completion: answer text = final_response(raw, row thinking) -- the final response
after the thinking channel (thinking ON and no '<channel|>' = the thought never closed -> '') --, then pred.py's
extract_answer (strip '*', 'The correct answer is (X)', then 'The correct answer is X'); judge = pred == gold; an
unparsed answer (None) is wrong. Every pool item counts (pred.py only skips an item whose API call returned '', an
error path).
Outputs (OUT_PREFIX):
  .official.json             {arm: {cell: bool}}  PRIMARY: pred.py's judge, no finish requirement
  .judge_and_finished.json   {arm: {cell: bool}}  SECONDARY: judge AND a stop / eos finish
  .summary.json              per arm and dataset, per run (panel seed, repeat): result.py's Overall / Easy / Hard /
                             Short / Medium / Long (round(100 * acc / n, 1)), cells, finished / capped / other,
                             unparsed, no_final_response, truncated items; plus the mean over runs when there are several
A duplicate cell key raises; every cell's id must be the manifest row at its index. No text, ids or answers written.
Run with cwd / PYTHONPATH holding experiments/ (CPU only, no torch).
usage: python v31_score_longbench_official.py OUT_PREFIX MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR holds {dataset}_rowinfo.json (or the manifest: id, difficulty, length, thinking, truncated); GOLD_DIR
  {dataset}_gold.json ({id: letter}); arm labels from file names <tag>_<label>.private.jsonl
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

SECONDARY = 'judge_and_finished'
COLUMNS = ('Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long')


def score(records, rows_of, gold_of, final_response):
    official, strict, diag, meta = {}, {}, {}, {}
    for label, key, r in records:
        row = op.check_cell(rows_of(r['dataset']), r['dataset'], r)
        gold = gold_of(r['dataset'])[r['id']]
        if gold not in ('A', 'B', 'C', 'D'):
            raise ValueError('gold must be one of A-D')
        text = final_response(r['completion'], bool(row['thinking']))
        pred = op.lb_extract_answer(text)
        judge = pred == gold
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, judge)
        op.put(strict, label, key, judge and fc == 'finished')
        meta[key] = dict(difficulty=row['difficulty'], length=row['length'], truncated=bool(row.get('truncated')))
        d = diag.setdefault((label, r['dataset']), collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['unparsed'] += pred is None
        d['no_final_response'] += text == ''
        d['truncated_items'] += bool(row.get('truncated'))
    return official, {SECONDARY: strict}, diag, meta


def summarize(official, diag, meta):
    out = {}
    for label, cells in official.items():
        runs = collections.defaultdict(list)
        for k, v in cells.items():
            ds, _, seed, repeat = k.split('|')
            runs[(ds, f'seed{seed}_repeat{repeat}')].append(dict(judge=v, **meta[k]))
        for ds in sorted({d for d, _ in runs}):
            per_run = {run: dict(op.lb_result(rows), items=len(rows)) for (d, run), rows in sorted(runs.items()) if d == ds}
            entry = dict(runs=per_run, **dict(diag[(label, ds)]))
            if len(per_run) > 1:
                entry['mean_over_runs'] = {c: round(statistics.mean(r[c] for r in per_run.values()), 1)
                                           for c in COLUMNS if all(r[c] is not None for r in per_run.values())}
            out.setdefault(label, {})[ds] = entry
    return out


def main():
    prefix, man_dir, gold_dir, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    final_response, source = op.load_final_response()
    rows, golds = {}, {}

    def rows_of(ds):
        if ds not in rows:
            rows[ds] = op.load_rows(man_dir, ds, ('id', 'difficulty', 'length', 'thinking', 'truncated'))
        return rows[ds]

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    official, secondary, diag, meta = score(op.read_private(files), rows_of, gold_of, final_response)
    summary = dict(metric="LongBench-v2 pred.py extract_answer judge; result.py aggregation (one decimal)",
                   answer_text=source, arms=summarize(official, diag, meta))
    op.write_outputs(prefix, official, secondary, summary)
    for label, per_ds in sorted(summary['arms'].items()):
        for ds, s in per_ds.items():
            for run, res in s['runs'].items():
                print(label, ds, run, ' '.join(f'{c}={res[c]}' for c in COLUMNS), f"n={res['items']} capped={s.get('capped', 0)} "
                      f"unparsed={s['unparsed']}")


if __name__ == '__main__':
    main()
