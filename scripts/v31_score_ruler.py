"""Score v31 private completions of RULER cells with the pinned official RULER metric (primary: per-sample score).

Rule: scripts/v31_official.py. Per completion: answer text = final_response(raw, row thinking) (thinking is off for
every RULER pool: the text after the last '<channel|>' if one is present, cut at the first end token, stripped), RULER's
postprocess_pred, then the official metric of the task base -- string_match_all (niah, variable_tracking,
common_words_extraction, freq_words_extraction) or string_match_part (qa) of RULER @ c3f5e3b
scripts/eval/synthetic/constants.py -- on [pred] vs the gold `outputs`. With the official prompt placement
(pools_v31_official/ruler_v33) the answer prefix is part of the prompt, so the completion is the continuation after
it, exactly RULER's `pred`; with the older pools (prefix at the end of the user message) the same text is scored.
Outputs (OUT_PREFIX):
  .official.json                       {arm: {cell: score 0-100}}  PRIMARY: RULER's per-sample metric (partial credit
                                       for multi-answer tasks), no finish requirement
  .strict_all_correct_finished.json    {arm: {cell: bool}}  SECONDARY: score == 100 AND a stop / eos finish (the boolean
                                       of the earlier McNemar analyses; not RULER's metric)
  .summary.json                        per arm: RULER's official aggregation (per length: mean per task, then the
                                       unweighted mean over tasks; overall = mean over lengths), cells, finished /
                                       capped (finish 'length') / other, nulls (empty predictions, RULER's 'Nulls');
                                       plus the RULER checkout identity and the answer-text source hash
A duplicate cell key raises; every cell's id must be the manifest row at its index. No text, ids or predictions are
written. Run with cwd / PYTHONPATH holding experiments/ (CPU only); the RULER checkout must be the pinned commit.
usage: python v31_score_ruler.py OUT_PREFIX RULER_ROOT MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR holds {dataset}_rowinfo.json or {dataset}_generation_manifest.json; GOLD_DIR {dataset}_gold.json
  ({id: [answer, ...]}); arm labels from file names <tag>_<label>.private.jsonl
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

STRICT = 'strict_all_correct_finished'


def score(records, metrics, task_base, rows_of, gold_of, final_response):
    """records [(label, key, record)] -> (official, {STRICT: booleans}, diagnostics)."""
    official, strict, diag = {}, {}, {}
    for label, key, r in records:
        row = op.check_cell(rows_of(r['dataset']), r['dataset'], r)
        pred = op.ruler_postprocess(final_response(r['completion'], bool(row.get('thinking', False))))
        value = float(metrics[row.get('task_base') or task_base[row['task']]]([pred], [gold_of(r['dataset'])[r['id']]]))
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, value)
        op.put(strict, label, key, value >= 100.0 - 1e-9 and fc == 'finished')
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['nulls'] += pred == ''
    return official, {STRICT: strict}, diag


def summarize(official, diag, rows_of):
    task_of = lambda k: rows_of(k.split('|')[0])[int(k.split('|')[1])]['task']
    length_of = lambda k: k.split('|')[0]
    return {label: dict(official=op.ruler_official_aggregate(cells, task_of, length_of), **dict(diag[label]))
            for label, cells in official.items()}


def main():
    prefix, ruler_root, man_dir, gold_dir, files = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
    checkout = op.ruler_verify_checkout(ruler_root)
    metrics = op.ruler_metrics(ruler_root)
    final_response, source = op.load_final_response()
    rows, golds = {}, {}

    def rows_of(ds):
        if ds not in rows:
            rows[ds] = op.load_rows(man_dir, ds, ('id', 'task', 'task_base', 'thinking'))
        return rows[ds]

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    records = op.read_private(files)
    task_base = {}
    if any('task_base' not in rows_of(r['dataset'])[r['index']] for _, _, r in records):
        task_base = op.ruler_task_bases(ruler_root)
    official, secondary, diag = score(records, metrics, task_base, rows_of, gold_of, final_response)
    summary = dict(metric='RULER per-sample score 0-100 (string_match_all / string_match_part), official aggregation',
                   ruler=checkout, answer_text=source, arms=summarize(official, diag, rows_of))
    op.write_outputs(prefix, official, secondary, summary)
    for label, s in sorted(summary['arms'].items()):
        per_len = {k: round(v['avg'], 2) for k, v in s['official'].items() if k != '_overall'}
        print(label, f"official={s['official']['_overall']:.2f}", per_len, f"cells={s['cells']} capped={s.get('capped', 0)} "
              f"nulls={s['nulls']} strict={sum(secondary[STRICT][label].values())}")


if __name__ == '__main__':
    main()
