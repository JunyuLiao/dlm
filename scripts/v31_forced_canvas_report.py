"""Step inflation and fidelity on the SAME canvases (forced-canvas mode of v31_vllm_paired_bench.py).

Every arm runs with FORCE_REF = the reference run's committed tokens: canvas i starts from the reference prefix with
the same per-canvas noise seed, so the arm's denoising-step count for canvas i and the fraction of its converged tokens
equal to the reference's measure the effect of its attention alone, canvas by canvas (no trajectory divergence).
Per arm against the reference record (FORCE_RECORD run; the self-forced reference must reproduce it exactly):
  canvases / cells, forced outputs equal to the reference (must be all),
  step ratio = total arm steps / total reference steps on the paired canvases, 95% CI by a cell-clustered bootstrap,
  geometric mean of the per-canvas step ratio (CI likewise), canvases with more / equal / fewer steps,
  token agreement: mean over canvases, share of canvases fully equal.
Timing fields of forced records are not valid (the mode synchronizes) and are not read. Public records only.
usage: python v31_forced_canvas_report.py REF_LABEL RECORDS.jsonl [...]   (labels from <tag>_<label>.jsonl)
"""
import collections
import json
import math
import random
import sys
from pathlib import Path


def load(files):
    recs = collections.defaultdict(dict)
    for f in files:
        label = Path(f).name.split('.jsonl')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r.get('repeat', 0) < 0 or 'forced_canvas_steps' not in r:
                continue
            recs[label][(r['dataset'], r['index'], r['panel_seed'], r['repeat'])] = r
    return recs


def cluster_ci(cells, stat, reps=2000, seed=11):
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(cells) for _ in cells]) for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def main():
    ref_label, files = sys.argv[1], sys.argv[2:]
    recs = load(files)
    ref = recs.pop(ref_label)
    datasets = sorted({k[0] for k in ref})
    head = ['arm', 'cells', 'canvases', 'output = ref', 'step ratio (sum) [95% CI]', 'per-canvas geo mean [95% CI]',
            'more / equal / fewer', 'token agreement', 'canvases fully equal']
    for ds in datasets + ['all']:
        print(f'\n### {ds}: reference {ref_label}\n')
        print('| ' + ' | '.join(head) + ' |')
        print('|' + '---|' * len(head))
        for label in sorted(recs):
            pairs = []                                       # per cell: [(arm steps, ref steps, agree), ...]
            match = 0
            for k, r in ref.items():
                if (ds != 'all' and k[0] != ds) or k not in recs[label]:
                    continue
                a = recs[label][k]
                match += bool(a.get('forced_output_matches_ref'))
                sa, sr, ag = a['forced_canvas_steps'], r['forced_canvas_steps'], a.get('forced_agree') or []
                n = min(len(sa), len(sr))
                pairs.append([(sa[i], sr[i], ag[i] if i < len(ag) else None) for i in range(n)])
            if not pairs:
                continue
            flat = [c for cell in pairs for c in cell]

            def ratio_sum(cells):
                a = sum(c[0] for cell in cells for c in cell)
                b = sum(c[1] for cell in cells for c in cell)
                return a / b if b else float('nan')

            def geo(cells):
                logs = [math.log(c[0] / c[1]) for cell in cells for c in cell if c[0] > 0 and c[1] > 0]
                return math.exp(sum(logs) / len(logs)) if logs else float('nan')
            lo, hi = cluster_ci(pairs, ratio_sum)
            glo, ghi = cluster_ci(pairs, geo)
            more = sum(c[0] > c[1] for c in flat)
            fewer = sum(c[0] < c[1] for c in flat)
            agree = [c[2] for c in flat if c[2] is not None]
            row = [label, str(len(pairs)), str(len(flat)), f'{match}/{len(pairs)}',
                   f'{ratio_sum(pairs):.3f} [{lo:.3f}, {hi:.3f}]', f'{geo(pairs):.3f} [{glo:.3f}, {ghi:.3f}]',
                   f'{more} / {len(flat) - more - fewer} / {fewer}',
                   f'{sum(agree) / len(agree):.4f}' if agree else '-',
                   f'{sum(x == 1.0 for x in agree)}/{len(agree)}' if agree else '-']
            print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
