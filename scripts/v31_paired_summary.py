"""Paired summary of v31 seed-paired vLLM panels.

Pairs every arm with a reference arm on (dataset, index, panel seed, repeat); all arms of a cell ran on the same
host with the same per-request RNG seed. Per bin and arm, reports paired geometric-mean ratios with item-clustered
bootstrap 95% CIs for:
  W   request wall time (incl. prefill)        S   decode span (generation only, incl. commits)
  C   canvases (commits)                       N/C denoising forwards per canvas
  N   denoising forwards (= C x N/C)           S/N amortized per-forward decode cost
  T   output tokens
plus the share of cells whose output is token-identical to the reference, and accuracy when a scored file is given.
usage: python v31_paired_summary.py OUT_PREFIX REF_LABEL FILE.jsonl [FILE.jsonl ...] [--scores SCORES.json]
  arm labels come from the file names: <tag>_<label>.jsonl (label = everything after the first '_')
"""
import collections
import json
import math
import random
import sys
from pathlib import Path


def load(files):
    rows = collections.defaultdict(dict)
    for f in files:
        label = Path(f).stem.split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            key = (r['dataset'], r['index'], r['panel_seed'], r['repeat'])
            if label in rows[key]:
                raise ValueError(f'duplicate cell {key} for {label}')
            rows[key][label] = r
    return rows


def metrics(r):
    n = r['denoise_forwards']
    return dict(W=r['wall_s'], S=r['decode_s'], C=r['canvases'], NC=n / r['canvases'], N=n, SN=r['decode_s'] / n,
                T=r['output_tokens'])


def geo_ci(groups, reps=4000, seed=7):
    """groups: {item: [log ratios]} -> (geomean, lo, hi) with items resampled."""
    items = list(groups)
    flat = [x for v in groups.values() for x in v]
    point = math.exp(sum(flat) / len(flat))
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        pick = [x for i in (rng.choice(items) for _ in items) for x in groups[i]]
        boots.append(math.exp(sum(pick) / len(pick)))
    boots.sort()
    return point, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def main():
    args = sys.argv[1:]
    scores = None
    if '--scores' in args:
        i = args.index('--scores')
        scores = json.load(open(args[i + 1]))
        args = args[:i] + args[i + 2:]
    out, ref, files = args[0], args[1], args[2:]
    rows = load(files)
    labels = sorted({l for v in rows.values() for l in v})
    if ref not in labels:
        raise ValueError(f'reference {ref} not among {labels}')
    lines, csv = [], ['bin,arm,cells,metric,ratio,lo,hi']
    for ds in sorted({k[0] for k in rows}):
        lines.append(f'### {ds}\n')
        lines.append('| arm | cells | W [CI] | S [CI] | C | N/C [CI] | N | S/N [CI] | T | identical | correct (ref) |')
        lines.append('|---|---:|---|---|---:|---|---:|---|---:|---:|---|')
        for arm in labels:
            pairs = [(k, v[arm], v[ref]) for k, v in rows.items() if k[0] == ds and arm in v and ref in v]
            if not pairs:
                continue
            res = {}
            for m in ('W', 'S', 'C', 'NC', 'N', 'SN', 'T'):
                g = collections.defaultdict(list)
                for k, a, b in pairs:
                    g[k[1]].append(math.log(metrics(a)[m] / metrics(b)[m]))
                res[m] = geo_ci(g)
                csv.append(f'{ds},{arm},{len(pairs)},{m},{res[m][0]:.4f},{res[m][1]:.4f},{res[m][2]:.4f}')
            same = sum(a['output_hash'] == b['output_hash'] for _, a, b in pairs)
            acc = ''
            if scores is not None:
                ca = sum(bool(scores.get(arm, {}).get(f'{k[0]}|{k[1]}|{k[2]}|{k[3]}')) for k, _, _ in pairs)
                cb = sum(bool(scores.get(ref, {}).get(f'{k[0]}|{k[1]}|{k[2]}|{k[3]}')) for k, _, _ in pairs)
                acc = f'{ca} ({cb})'
            f = lambda m: f'{res[m][0]:.3f} [{res[m][1]:.3f}, {res[m][2]:.3f}]'
            lines.append(f"| {arm} | {len(pairs)} | {f('W')} | {f('S')} | {res['C'][0]:.3f} | {f('NC')} | "
                         f"{res['N'][0]:.3f} | {f('SN')} | {res['T'][0]:.3f} | {same}/{len(pairs)} | {acc} |")
        lines.append('')
    Path(out + '.md').write_text(f'Reference: `{ref}`. Paired geometric-mean ratios arm/ref, item-clustered 95% CI.\n\n'
                                 + '\n'.join(lines), encoding='utf-8')
    Path(out + '.csv').write_text('\n'.join(csv) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
