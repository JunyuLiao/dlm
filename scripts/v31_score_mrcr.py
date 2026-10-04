"""Score v31 private MRCR 2-needle completions with the official OpenAI MRCR grading (primary: ratio per sample).

Rule: scripts/v31_official.py. Per completion: answer text = final_response(raw, row thinking) (OFF for every MRCR
pool: the text after the last '<channel|>' if one is present, cut at the first '<turn|>' / '<|endoftext|>' / '<eos>',
stripped), then the README's `grade` (openai/mrcr @ f4c69fae): 0 unless the response starts with
random_string_to_prepend, otherwise difflib.SequenceMatcher(None, response, answer).ratio() after removing that prefix
from both (difflib defaults, autojunk on, as the published numbers).
Outputs (OUT_PREFIX):
  .official.json                     {arm: {cell: ratio}}  PRIMARY (no finish requirement)
  .ratio_ge_099_and_finished.json    {arm: {cell: bool}}  SECONDARY: ratio >= 0.99 AND a stop / eos finish
  .summary.json                      per arm: the official MRCR number = mean ratio per bin (dataset), plus the mean over
                                     bins; cells, finished / capped / other, missing_prefix (ratio 0 because the
                                     response does not open with the random string)
A duplicate cell key raises; every cell's id must be the manifest row at its index. No text, ids or answers written.
Run with cwd / PYTHONPATH holding experiments/ (CPU only, no torch).
usage: python v31_score_mrcr.py OUT_PREFIX MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR / GOLD_DIR hold {dataset}_generation_manifest.json (or _rowinfo.json) and {dataset}_gold.json
  ({id: {"answer", "random_string_to_prepend"}}); arm labels from file names <tag>_<label>.private.jsonl
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


def score(records, rows_of, gold_of, final_response):
    official, strict, diag = {}, {}, {}
    for label, key, r in records:
        row = op.check_cell(rows_of(r['dataset']), r['dataset'], r)
        gold = gold_of(r['dataset'])[r['id']]
        text = final_response(r['completion'], bool(row['thinking']))
        ratio = grade(text, gold['answer'], gold['random_string_to_prepend'])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, ratio)
        op.put(strict, label, key, ratio >= THRESHOLD and fc == 'finished')
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['missing_prefix'] += not text.startswith(gold['random_string_to_prepend'])
    return official, {SECONDARY: strict}, diag


def summarize(official, diag):
    out = {}
    for label, cells in official.items():
        per_bin = collections.defaultdict(list)
        for k, v in cells.items():
            per_bin[k.split('|')[0]].append(v)
        bins = {b: dict(mean_ratio=statistics.mean(v), cells=len(v)) for b, v in sorted(per_bin.items())}
        out[label] = dict(bins=bins, mean_over_bins=statistics.mean(b['mean_ratio'] for b in bins.values()), **dict(diag[label]))
    return out


def main():
    prefix, man_dir, gold_dir, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    final_response, source = op.load_final_response()
    rows, golds = {}, {}

    def rows_of(ds):
        if ds not in rows:
            rows[ds] = op.load_rows(man_dir, ds, ('id', 'thinking'))
        return rows[ds]

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    official, secondary, diag = score(op.read_private(files), rows_of, gold_of, final_response)
    summary = dict(metric='openai/mrcr grade: SequenceMatcher ratio with the random-prefix check; mean per bin',
                   answer_text=source, arms=summarize(official, diag))
    op.write_outputs(prefix, official, secondary, summary)
    for label, s in sorted(summary['arms'].items()):
        print(label, {b: round(v['mean_ratio'], 4) for b, v in s['bins'].items()}, f"cells={s['cells']} "
              f"capped={s.get('capped', 0)} missing_prefix={s['missing_prefix']}")


if __name__ == '__main__':
    main()
