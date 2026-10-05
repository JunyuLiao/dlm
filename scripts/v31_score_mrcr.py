"""Score v31 private MRCR 2-needle completions with the official OpenAI MRCR grading (primary: ratio per sample).

Rule: scripts/v31_official.py (planned cells, binding). Per completion: answer text = the raw response the README grades
(`answer_unstripped`: the text after the last '<channel|>' if present -- the API's message content excludes the thought
channel --, cut at the first end token, NOT stripped), then the README's `grade` (openai/mrcr @ f4c69fae): 0 unless the
response starts with random_string_to_prepend, otherwise difflib.SequenceMatcher(None, response, answer).ratio() after
removing that prefix from both (difflib defaults, autojunk on, as the published numbers).
Outputs (OUT_PREFIX):
  .official.json                     {arm: {cell: ratio}}  PRIMARY (no finish requirement)
  .ratio_ge_099_and_finished.json    {arm: {cell: bool}}  SECONDARY: ratio >= 0.99 AND a stop / eos finish
  .binding.json                      per-cell run settings for the comparison tools (v31_paired_official.py mrcr)
  .summary.json                      per arm: the official MRCR number = mean ratio per bin (dataset), plus the mean
                                     over bins; cells, finished / capped / other, missing_prefix (ratio 0 because the
                                     response does not open with the random string), coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids or answers written. Run with cwd / PYTHONPATH holding experiments/.
usage: python v31_score_mrcr.py OUT_PREFIX MANIFEST_DIR GOLD_DIR --cells CELLS.json [--repeats N] [--allow-missing]
           [--legacy-unbound] PRIVATE.jsonl [...]
  GOLD_DIR holds {dataset}_gold.json ({id: {"answer", "random_string_to_prepend"}})
"""
import collections
import json
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

THRESHOLD = 0.99
SECONDARY = 'ratio_ge_099_and_finished'


def grade(response, answer, random_string_to_prepend) -> float:
    """openai/mrcr README `grade`, verbatim apart from returning 0.0 as a float."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def score(records, pools, gold_of, legacy=False):
    official, strict, binding, diag = {}, {}, {}, {}
    for label, key, r in records:
        row, b, verified = op.bind(r, pools, legacy)
        gold = gold_of(r['dataset'])[r['id']]
        text = op.answer_unstripped(r['completion'], bool(row['thinking']))
        ratio = grade(text, gold['answer'], gold['random_string_to_prepend'])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, ratio)
        op.put(strict, label, key, ratio >= THRESHOLD and fc == 'finished')
        op.put(binding, label, key, b)
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['missing_prefix'] += not text.startswith(gold['random_string_to_prepend'])
        d['verified' if verified else 'legacy_unbound'] += 1
    return official, {SECONDARY: strict}, binding, diag


def summarize(official, diag):
    out = {}
    for label, cells in official.items():
        per_bin = collections.defaultdict(list)
        for k, v in cells.items():
            per_bin[k.split('|')[0]].append(v)
        bins = {b: dict(mean_ratio=statistics.mean(v), cells=len(v)) for b, v in sorted(per_bin.items(), key=lambda x: op.natural_key(x[0]))}
        out[label] = dict(bins=bins, mean_over_bins=statistics.mean(b['mean_ratio'] for b in bins.values()), **dict(diag[label]))
    return out


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'manifest_dir', 'gold_dir'))
    golds = {}

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((Path(a.gold_dir) / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    records, coverage = op.planned_records(a)
    official, secondary, binding, diag = score(records, op.Pools(a.manifest_dir, ('thinking',)), gold_of, a.legacy_unbound)
    summary = dict(metric='openai/mrcr grade on the unstripped response: SequenceMatcher ratio with the random-prefix '
                          'check; mean per bin', coverage=coverage, arms=summarize(official, diag))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    for label, s in sorted(summary['arms'].items()):
        print(label, {b: round(v['mean_ratio'], 4) for b, v in s['bins'].items()}, f"cells={s['cells']} "
              f"capped={s.get('capped', 0)} missing_prefix={s['missing_prefix']}")


if __name__ == '__main__':
    main()
