"""Score v31 private GraphWalks completions with the official OpenAI GraphWalks extraction and set F1 (primary: F1).

Rule: scripts/v31_official.py (planned cells, binding). Per completion: answer text = the raw response the README grades
(`answer_unstripped`: the text after the last '<channel|>' -- thinking is ON for every GraphWalks pool, an unfinished
thought gives '' --, cut at the first end token, NOT stripped), then the README's `get_list` (openai/graphwalks @
be6cc6ec: the last line must contain 'Final Answer:', the greedy '[...]' span on it is split on commas, items stripped,
empty items dropped; otherwise unparsed) and its set precision / recall / F1 against the gold answer nodes (unparsed -> 0;
empty gold and empty prediction -> 1).
Outputs (OUT_PREFIX):
  .official.json               {arm: {cell: f1}}  PRIMARY (no finish requirement)
  .f1_eq_1_and_finished.json   {arm: {cell: bool}}  SECONDARY: F1 == 1 AND a stop / eos finish
  .binding.json                per-cell run settings for the comparison tools (v31_paired_official.py graphwalks)
  .summary.json                per arm: the official GraphWalks numbers = mean F1 per bin (dataset) per problem type
                               (bfs / parents), plus per bin; cells, finished / capped / other, unparsed,
                               no_final_response, coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids or answers written. Run with cwd / PYTHONPATH holding experiments/.
usage: python v31_score_graphwalks.py OUT_PREFIX MANIFEST_DIR GOLD_DIR --cells CELLS.json [--repeats N] [--allow-missing]
           [--legacy-unbound] PRIVATE.jsonl [...]
  GOLD_DIR holds {dataset}_gold.json ({id: [node, ...]})
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


def score(records, pools, gold_of, legacy=False):
    official, strict, binding, diag, types = {}, {}, {}, {}, {}
    for label, key, r in records:
        row, b, verified = op.bind(r, pools, legacy)
        text = op.answer_unstripped(r['completion'], bool(row['thinking']))
        f1 = f1_score(text, gold_of(r['dataset'])[r['id']])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, f1)
        op.put(strict, label, key, f1 == 1.0 and fc == 'finished')
        op.put(binding, label, key, b)
        types[key] = row['problem_type']
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['unparsed'] += get_list(text)[1]
        d['no_final_response'] += text == ''
        d['verified' if verified else 'legacy_unbound'] += 1
    return official, {SECONDARY: strict}, binding, diag, types


def summarize(official, diag, types):
    out = {}
    for label, cells in official.items():
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, v in cells.items():
            per[k.split('|')[0]][types[k]].append(v)
            per[k.split('|')[0]]['all'].append(v)
        out[label] = dict(bins={b: {t: dict(mean_f1=statistics.mean(v), cells=len(v)) for t, v in sorted(by_type.items())}
                                for b, by_type in sorted(per.items(), key=lambda x: op.natural_key(x[0]))},
                          **dict(diag[label]))
    return out


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'manifest_dir', 'gold_dir'))
    golds = {}

    def gold_of(ds):
        if ds not in golds:
            golds[ds] = json.loads((Path(a.gold_dir) / f'{ds}_gold.json').read_text(encoding='utf-8'))
        return golds[ds]

    records, coverage = op.planned_records(a)
    official, secondary, binding, diag, types = score(records, op.Pools(a.manifest_dir, ('problem_type', 'thinking')),
                                                      gold_of, a.legacy_unbound)
    summary = dict(metric='openai/graphwalks set F1 of the final-answer list on the unstripped response; mean per bin per '
                          'problem type', coverage=coverage, arms=summarize(official, diag, types))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    for label, s in sorted(summary['arms'].items()):
        print(label, {b: {t: round(v['mean_f1'], 4) for t, v in by.items()} for b, by in s['bins'].items()},
              f"cells={s['cells']} capped={s.get('capped', 0)} unparsed={s['unparsed']}")


if __name__ == '__main__':
    main()
