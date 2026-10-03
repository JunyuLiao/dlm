"""Trajectory stability of seed-paired arms: how much does an arm perturb each request's denoising trajectory?

With equal per-request seeds, two arms that differ only numerically (dense PIECEWISE vs dense FULL + fix) already
change a request's number of denoising forwards N by a large factor (log-ratio s.d. ~0.3), because a single changed
acceptance decision sends the trajectory elsewhere. Per arm and length bin, against the reference arm:
  sd       s.d. of the per-cell log N ratio, with an item-clustered bootstrap 95% CI
  excess   sqrt(max(0, sd^2 - sd_null^2)): the perturbation beyond numerical noise (null = NULL_LABEL)
  p90      90th percentile of the per-cell N ratio (tail inflation)
  P>1.5    share of cells whose N grew by more than 1.5x
  med      median per-cell N ratio
usage: python v31_trajectory_stability.py OUT_PREFIX REF_LABEL NULL_LABEL FILE.jsonl [...]
  labels from file names <tag>_<label>.jsonl (as v31_paired_summary.py); repeat 0 only
"""
import collections
import json
import math
import random
import statistics as st
import sys
from pathlib import Path


def load(files):
    rows = collections.defaultdict(dict)
    for f in files:
        label = Path(f).stem.split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r.get('repeat', 0) == 0:
                rows[(r['dataset'], r['index'], r['panel_seed'])][label] = r
    return rows


def stats(cells, null_sd=None):
    """cells: list of (item, log ratio)."""
    x = [v for _, v in cells]
    sd = st.pstdev(x)
    ratios = sorted(math.exp(v) for v in x)
    out = dict(n=len(x), sd=sd, med=math.exp(st.median(x)), p90=ratios[min(len(ratios) - 1, int(0.9 * len(ratios)))],
               p15=sum(r > 1.5 for r in ratios) / len(ratios))
    by_item = collections.defaultdict(list)
    for item, v in cells:
        by_item[item].append(v)
    items = list(by_item)
    rng = random.Random(7)
    boots = []
    for _ in range(2000):
        sample = [v for it in (rng.choice(items) for _ in items) for v in by_item[it]]
        boots.append(st.pstdev(sample))
    boots.sort()
    out['sd_lo'], out['sd_hi'] = boots[50], boots[1949]
    if null_sd is not None:
        out['excess'] = math.sqrt(max(0.0, sd * sd - null_sd * null_sd))
    return out


def main():
    out_prefix, ref, null, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    rows = load(files)
    labels = sorted({l for v in rows.values() for l in v} - {ref})
    lines = [f'Trajectory stability vs `{ref}` (null: `{null}`). Per-cell log N ratio, item-clustered bootstrap.', '']
    for ds in sorted({k[0] for k in rows}):
        def cells(lab):
            return [(k[1], math.log(v[lab]['denoise_forwards'] / v[ref]['denoise_forwards']))
                    for k, v in rows.items() if k[0] == ds and lab in v and ref in v]
        null_cells = cells(null) if null in labels else []
        null_sd = stats(null_cells)['sd'] if null_cells else None
        lines += [f'### {ds}', '', '| arm | cells | sd [95% CI] | excess | median | p90 | P(>1.5x) |', '|---|---:|---|---|---|---|---|']
        for lab in labels:
            c = cells(lab)
            if len(c) < 8:
                continue
            s = stats(c, null_sd)
            lines.append(f"| {lab} | {s['n']} | {s['sd']:.3f} [{s['sd_lo']:.3f}, {s['sd_hi']:.3f}] | "
                         f"{s.get('excess', float('nan')):.3f} | {s['med']:.3f} | {s['p90']:.3f} | {s['p15']:.2f} |")
        lines.append('')
    Path(out_prefix + '.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
