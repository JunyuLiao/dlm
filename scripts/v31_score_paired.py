"""Score v31 private completions with the panels' unchanged LongBench-v2 scorer (v15 NeMo MCQ on the final channel,
strict EOS), on mpk where the long-LB gold and the pinned NeMo-Skills checkout live. Writes only booleans:
{arm_label: {"dataset|index|panel_seed|repeat": correct}} -- no text, no ids, no predictions.
Run in a panel deployment (cwd, PYTHONPATH=src:.) with the project's scorer environment.
usage: python v31_score_paired.py OUT_JSON GOLD_DIR32_64 GOLD_DIR96 PRIVATE.jsonl [PRIVATE.jsonl ...]
  arm labels come from the file names: <tag>_<label>.private.jsonl
"""
import json
import sys
from pathlib import Path

from scripts import v15_longbench_task as task


def main():
    out, gold64, gold96, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    golds = {}
    for ds, d in (('longbench_v2_32k', gold64), ('longbench_v2_64k', gold64), ('longbench_v2_96k', gold96)):
        p = d / f'{ds}_gold_scorer_only.json'
        if p.exists():
            golds[ds] = json.loads(p.read_text())
    result = {}
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        labels = [golds[r['dataset']][r['id']] for r in rows]
        scores = task.score([r['completion'] for r in rows], labels,
                            ['eos' if r['finish_reason'] in ('stop', 'eos') else 'length' for r in rows])
        result.setdefault(label, {})
        for r, s in zip(rows, scores):
            key = f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            if key in result[label]:
                raise ValueError(f'duplicate {label} {key}')
            result[label][key] = bool(s['task_correct'])
        print(label, len(rows), sum(result[label].values()), flush=True)
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
