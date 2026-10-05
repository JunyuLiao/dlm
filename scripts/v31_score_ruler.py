"""Score v31 private completions of RULER cells with the pinned official RULER metric (primary: per-sample score).

Rule: scripts/v31_official.py (planned cells, binding, unrounded aggregation). Per completion:
  - answer text: on a prefilled pool (ruler*_v33ofc / ruler32k_v33noop: the prompt already holds the answer prefix,
    rowinfo `prefilled`) the completion cut at the first end token only -- exactly RULER's `pred`; on older pools
    final_response(raw, row thinking) (thinking off: the text after the last '<channel|>' if present);
  - RULER's postprocess_pred, then the per-sample score of the task base's official metric -- string_match_all (niah,
    variable_tracking, common_words_extraction, freq_words_extraction) or string_match_part (qa) of RULER @ c3f5e3b --
    unrounded (asserted to round to exactly what the pinned metric function returns).
Outputs (OUT_PREFIX):
  .official.json                       {arm: {cell: score 0-100}}  PRIMARY, no finish requirement
  .strict_all_correct_finished.json    {arm: {cell: bool}}  SECONDARY: score == 100 AND a stop / eos finish
  .binding.json                        {arm: {cell: manifest / prompt sha256, budget, rng_seed, prompt_tokens,
                                       max_model_len, chunk, block_size}} for the comparison tools
  .summary.json                        per arm: RULER's official aggregation (per length: mean per task, then the
                                       unweighted mean over tasks; overall = mean over lengths), cells, finished / capped /
                                       other, nulls (empty predictions, RULER's 'Nulls'), coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids or predictions are written. Run with cwd / PYTHONPATH holding
experiments/ (CPU only); the RULER checkout must be the pinned commit.
usage: python v31_score_ruler.py OUT_PREFIX RULER_ROOT MANIFEST_DIR GOLD_DIR --cells CELLS.json [--cells ...]
           [--repeats N] [--allow-missing] [--legacy-unbound] PRIVATE.jsonl [...]
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

STRICT = 'strict_all_correct_finished'
FIELDS = ('task', 'task_base', 'thinking', 'prefilled')


def answer_text(row, raw, final_response):
    return op.cut_at_end(raw) if row.get('prefilled') else final_response(raw, bool(row.get('thinking', False)))


def score(records, metrics, task_base, pools, gold_of, final_response, legacy=False):
    """records [(label, key, record)] -> (official, {STRICT: booleans}, binding, diagnostics)."""
    official, strict, binding, diag = {}, {}, {}, {}
    for label, key, r in records:
        row, b, verified = op.bind(r, pools, legacy)
        pred = op.ruler_postprocess(answer_text(row, r['completion'], final_response))
        fn = metrics[row.get('task_base') or task_base[row['task']]]
        value = op.ruler_sample_score(fn, pred, gold_of(r['dataset'])[r['id']])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, value)
        op.put(strict, label, key, value >= 100.0 - 1e-9 and fc == 'finished')
        op.put(binding, label, key, b)
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['nulls'] += pred == ''
        d['verified' if verified else 'legacy_unbound'] += 1
    return official, {STRICT: strict}, binding, diag


def summarize(official, diag, pools):
    task_of = lambda k: pools(k.split('|')[0])[0][int(k.split('|')[1])]['task']
    length_of = lambda k: k.split('|')[0]
    return {label: dict(official=op.ruler_official_aggregate(cells, task_of, length_of), **dict(diag[label]))
            for label, cells in official.items()}


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'ruler_root', 'manifest_dir', 'gold_dir'))
    checkout = op.ruler_verify_checkout(a.ruler_root)
    metrics = op.ruler_metrics(a.ruler_root)
    final_response, source = op.load_final_response()
    pools, golds = op.Pools(a.manifest_dir, FIELDS), {}

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((Path(a.gold_dir) / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    records, coverage = op.planned_records(a)
    task_base = {}
    if any('task_base' not in pools(r['dataset'])[0][r['index']] for _, _, r in records):
        task_base = op.ruler_task_bases(a.ruler_root)
    official, secondary, binding, diag = score(records, metrics, task_base, pools, gold_of, final_response, a.legacy_unbound)
    summary = dict(metric='RULER per-sample score 0-100 (string_match_all / string_match_part, unrounded), official '
                          'aggregation', ruler=checkout, answer_text=source, coverage=coverage,
                   arms=summarize(official, diag, pools))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    for label, s in sorted(summary['arms'].items()):
        per_len = {k: round(v['avg'], 2) for k, v in s['official'].items() if k != '_overall'}
        print(label, f"official={s['official']['_overall']:.2f}", per_len, f"cells={s['cells']} capped={s.get('capped', 0)} "
              f"nulls={s['nulls']} strict={sum(secondary[STRICT][label].values())} unverified={s.get('legacy_unbound', 0)}")


if __name__ == '__main__':
    main()
