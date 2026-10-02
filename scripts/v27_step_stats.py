"""v27: absolute denoising-step statistics per arm (numbers only, no text).

For each dataset and arm, over successful first outputs (attempt0, ok) in the given ledgers:
- total steps per request (decoder calls): mean, median;
- canvases (blocks of 256 tokens) per request: mean;
- steps per block: pooled (sum of calls / sum of canvases), mean and median over blocks, share of blocks that hit
  the 48-step cap (no native stop);
- paired ratios to the dense control within each (item, seed) cell, geometric mean with an item-clustered bootstrap
  95% CI: total steps N and steps per block (calls / canvases);
- the same means per host (a host diagnostic; hosts hold different cells, so host gaps are not hardware effects).
usage: python -m scripts.v27_step_stats OUT_MD OUT_CSV LEDGER [LEDGER ...]
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict

BASE = 'D_fa4_allkept'
CAP = 48


def load(ledgers):
    runs = {}
    for path in ledgers:
        with open(path, encoding='utf-8') as f:
            for line in f:
                e = json.loads(line) if line.strip() else {}
                if e.get('event') != 'run' or e.get('role', 'attempt0') != 'attempt0' or not e.get('ok'):
                    continue
                key = (e['dataset'], e['id'], str(e['seed']), e['arm'])
                if key in runs:
                    raise ValueError('duplicate first output; merge disjoint executions only')
                per = [int(x) for x in (e.get('per_canvas_calls') or [])]
                stops = e.get('per_canvas_stopping') or []
                capped = sum(1 for s in stops if isinstance(s, dict) and not s.get('native_stop'))
                runs[key] = dict(N=int(e['decoder_calls']), C=int(e.get('canvases') or len(per)), per=per,
                                 capped=capped, host=e.get('host'))
    return runs


def geo_ci(by_item, reps=4000, seed=7):
    items = sorted(by_item)
    allv = [x for i in items for x in by_item[i]]
    if not allv:
        return None, None, None
    point = math.exp(statistics.mean(allv))
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        vals = [x for i in (rng.choice(items) for _ in items) for x in by_item[i]]
        boots.append(math.exp(statistics.mean(vals)))
    boots.sort()
    return point, boots[int(.025 * reps)], boots[int(.975 * reps) - 1]


def main(argv=None):
    argv = argv or sys.argv[1:]
    out_md, out_csv, ledgers = argv[0], argv[1], argv[2:]
    runs = load(ledgers)
    rows, lines = [], ['| dataset | arm | requests | total steps mean / median | canvases mean | steps per block '
                       'pooled / mean / median | blocks at 48-step cap | N ratio [CI] | steps/block ratio [CI] | '
                       'total steps mean by host |', '|' + '---|' * 10]
    for dataset in sorted({k[0] for k in runs}):
        arms = sorted({k[3] for k in runs if k[0] == dataset}, key=lambda a: (a != BASE, a))
        for arm in arms:
            mine = {k: v for k, v in runs.items() if k[0] == dataset and k[3] == arm}
            ns = [v['N'] for v in mine.values()]
            cs = [v['C'] for v in mine.values()]
            blocks = [x for v in mine.values() for x in v['per']]
            capped = sum(v['capped'] for v in mine.values())
            by_n, by_b = defaultdict(list), defaultdict(list)
            for (d, i, s, a), v in mine.items():
                b = runs.get((d, i, s, BASE))
                if b and arm != BASE and b['N'] and v['N'] and b['C'] and v['C']:
                    by_n[i].append(math.log(v['N'] / b['N']))
                    by_b[i].append(math.log((v['N'] / v['C']) / (b['N'] / b['C'])))
            n_ratio = geo_ci(by_n) if arm != BASE else (1.0, 1.0, 1.0)
            b_ratio = geo_ci(by_b) if arm != BASE else (1.0, 1.0, 1.0)
            hosts = defaultdict(list)
            for v in mine.values():
                hosts[(v['host'] or '?')[-3:]].append(v['N'])
            row = dict(dataset=dataset, arm=arm, requests=len(mine), total_steps_mean=round(statistics.mean(ns), 2),
                       total_steps_median=statistics.median(ns), canvases_mean=round(statistics.mean(cs), 2),
                       steps_per_block_pooled=round(sum(ns) / max(sum(cs), 1), 2),
                       steps_per_block_mean=round(statistics.mean(blocks), 2) if blocks else None,
                       steps_per_block_median=statistics.median(blocks) if blocks else None,
                       blocks=len(blocks), blocks_at_cap=capped,
                       N_ratio=None if n_ratio[0] is None else round(n_ratio[0], 4),
                       N_ci='' if n_ratio[1] is None else f'[{n_ratio[1]:.3f},{n_ratio[2]:.3f}]',
                       block_ratio=None if b_ratio[0] is None else round(b_ratio[0], 4),
                       block_ci='' if b_ratio[1] is None else f'[{b_ratio[1]:.3f},{b_ratio[2]:.3f}]',
                       by_host='; '.join(f'{h}: {statistics.mean(v):.1f} (n={len(v)})' for h, v in sorted(hosts.items())))
            rows.append(row)
            lines.append(f"| {dataset} | {arm} | {row['requests']} | {row['total_steps_mean']} / {row['total_steps_median']} | "
                         f"{row['canvases_mean']} | {row['steps_per_block_pooled']} / {row['steps_per_block_mean']} / "
                         f"{row['steps_per_block_median']} | {capped}/{len(blocks)} | {row['N_ratio']} {row['N_ci']} | "
                         f"{row['block_ratio']} {row['block_ci']} | {row['by_host']} |")
    with open(out_md, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
