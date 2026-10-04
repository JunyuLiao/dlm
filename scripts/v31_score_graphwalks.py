"""Score v31 private GraphWalks completions with the official OpenAI GraphWalks extraction and set F1 (primary: F1).

Rule: scripts/v31_official.py. Per completion: answer text = final_response(raw, row thinking) (ON for every GraphWalks
pool: the text after the last '<channel|>', cut at the first end token, stripped; an unfinished thought gives ''), then
the README's `get_list` (openai/graphwalks @ be6cc6ec: the last line must contain 'Final Answer:', the greedy '[...]'
span on it is split on commas, items stripped, empty items dropped; otherwise unparsed) and its set precision / recall /
F1 against the gold answer nodes (unparsed -> 0; empty gold and empty prediction -> 1).
Outputs (OUT_PREFIX):
  .official.json               {arm: {cell: f1}}  PRIMARY (no finish requirement)
  .f1_eq_1_and_finished.json   {arm: {cell: bool}}  SECONDARY: F1 == 1 AND a stop / eos finish
  .summary.json                per arm: the official GraphWalks numbers = mean F1 per bin (dataset) per problem type
                               (bfs / parents), plus per bin; cells, finished / capped / other, unparsed,
                               no_final_response
A duplicate cell key raises; every cell's id must be the manifest row at its index. No text, ids or answers written.
Run with cwd / PYTHONPATH holding experiments/ (CPU only, no torch).
usage: python v31_score_graphwalks.py OUT_PREFIX MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR / GOLD_DIR hold {dataset}_generation_manifest.json (or _rowinfo.json: id, problem_type, thinking) and
  {dataset}_gold.json ({id: [node, ...]}); arm labels from file names <tag>_<label>.private.jsonl
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

SECONDARY = 'f1_eq_1_and_finished'


def get_list(response: str) -> tuple[list[str], bool]:
    """openai/graphwalks README extraction, verbatim (as a free function)."""
    # get the very last line of the response
    line = response.split("\n")[-1]
    # check if formatted correctly
    if "Final Answer:" not in line:
        return [], True
    list_part = re.search(r"\[.*\]", line)
    if list_part:
        result_list = list_part.group(0).strip("[]").split(",")
        # if the list was empty, then get [] not [""]
        result_list = [item.strip() for item in result_list if item.strip()]
        return result_list, False
    else:
        return [], True


def f1_score(response: str, answer_nodes) -> float:
    """openai/graphwalks README grading, verbatim (F1 returned as a float)."""
    sampled_list, failed_to_parse = get_list(response)
    sampled_set = set(sampled_list)
    truth_set = set(answer_nodes)
    n_golden = len(truth_set)
    n_sampled = len(sampled_set)
    if failed_to_parse:
        recall = 0.0
        precision = 0.0
        f1 = 0.0
    elif n_golden == n_sampled == 0:
        recall = 1.0
        precision = 1.0
        f1 = 1.0
    else:
        n_overlap = len(sampled_set & truth_set)
        recall = n_overlap / n_golden if n_golden > 0 else 0
        precision = n_overlap / n_sampled if n_sampled > 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return float(f1)


def score(records, rows_of, gold_of, final_response):
    official, strict, diag, types = {}, {}, {}, {}
    for label, key, r in records:
        row = op.check_cell(rows_of(r['dataset']), r['dataset'], r)
        text = final_response(r['completion'], bool(row['thinking']))
        f1 = f1_score(text, gold_of(r['dataset'])[r['id']])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, f1)
        op.put(strict, label, key, f1 == 1.0 and fc == 'finished')
        types[key] = row['problem_type']
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['unparsed'] += get_list(text)[1]
        d['no_final_response'] += text == ''
    return official, {SECONDARY: strict}, diag, types


def summarize(official, diag, types):
    out = {}
    for label, cells in official.items():
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, v in cells.items():
            per[k.split('|')[0]][types[k]].append(v)
            per[k.split('|')[0]]['all'].append(v)
        out[label] = dict(bins={b: {t: dict(mean_f1=statistics.mean(v), cells=len(v)) for t, v in sorted(by_type.items())}
                                for b, by_type in sorted(per.items())}, **dict(diag[label]))
    return out


def main():
    prefix, man_dir, gold_dir, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    final_response, source = op.load_final_response()
    rows, golds = {}, {}

    def rows_of(ds):
        if ds not in rows:
            rows[ds] = op.load_rows(man_dir, ds, ('id', 'problem_type', 'thinking'))
        return rows[ds]

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    official, secondary, diag, types = score(op.read_private(files), rows_of, gold_of, final_response)
    summary = dict(metric='openai/graphwalks set F1 of the final-answer list; mean per bin per problem type',
                   answer_text=source, arms=summarize(official, diag, types))
    op.write_outputs(prefix, official, secondary, summary)
    for label, s in sorted(summary['arms'].items()):
        print(label, {b: {t: round(v['mean_f1'], 4) for t, v in by.items()} for b, by in s['bins'].items()},
              f"cells={s['cells']} capped={s.get('capped', 0)} unparsed={s['unparsed']}")


if __name__ == '__main__':
    main()
