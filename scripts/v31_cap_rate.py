"""Cap rate of a panel from its PUBLIC records: per arm and dataset, the share of requests that stopped at the
generation budget (finish_reason 'length'), with output-token statistics. No text.
usage: python v31_cap_rate.py PUBLIC.jsonl [...]   (labels from file names <tag>_<label>.jsonl; warm-up records skipped)
"""
import collections
import json
import statistics
import sys
from pathlib import Path


def main():
    groups = collections.defaultdict(list)
    for f in sys.argv[1:]:
        label = Path(f).name.split('.jsonl')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r.get('repeat', 0) >= 0:
                groups[(label, r['dataset'])].append(r)
    print('| arm | dataset | cells | capped (length) | cap rate | budget | output tokens median / p90 / max |')
    print('|---|---|---|---|---|---|---|')
    for (label, ds), rs in sorted(groups.items()):
        capped = sum(r['finish_reason'] == 'length' for r in rs)
        toks = sorted(r['output_tokens'] for r in rs)
        budgets = sorted({r['budget'] for r in rs})
        print(f"| {label} | {ds} | {len(rs)} | {capped} | {capped / len(rs):.3f} | {'/'.join(map(str, budgets))} | "
              f"{statistics.median(toks):.0f} / {toks[min(len(toks) - 1, int(0.9 * len(toks)))]} / {toks[-1]} |")


if __name__ == '__main__':
    main()
